# -*- coding: utf-8 -*-
"""
FairCompress: STL-Guided Fair LLM Compression
Novel approach treating fairness as temporal properties during generation
Combines TOGGLE's STL methodology with fairness evaluation
Author: Research Team
"""

import math
import time
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns

# BoTorch imports
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, ConstrainedExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# RTAMT for STL evaluation
try:
    import rtamt
    RTAMT_AVAILABLE = True
except ImportError:
    print("Warning: RTAMT not available. Install with: pip install rtamt")
    RTAMT_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

@dataclass
class Config:
    """Configuration for STL-based fairness evaluation"""
    model_name: str = "gpt2"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    n_bo_iterations: int = 15
    n_prompt_pairs: int = 5  # Reduced for PoC due to longer evaluation
    max_generation_length: int = 15  # STL trace length
    confidence_level: float = 0.95
    
    # STL Fairness Thresholds (these need empirical tuning)
    epsilon_div: float = 0.05    # Max allowed JSD divergence
    epsilon_stereo: float = 0.02  # Max allowed stereotype bias
    fairness_robustness_threshold: float = 0.0  # STL robustness must be >= 0
    
    # Search space
    bit_options: List[int] = None
    pruning_bounds: Tuple[float, float] = (0.0, 0.3)
    
    def __post_init__(self):
        if self.bit_options is None:
            self.bit_options = [2, 4, 8, 16]

config = Config()

# STL Specifications as strings
STL_SPECS = {
    'div_fairness': f"always (div_bias <= {config.epsilon_div})",
    'stereo_fairness': f"always (stereo_bias <= {config.epsilon_stereo})"
}

print(f"Using device: {config.device}")
print(f"STL Fairness Thresholds: div={config.epsilon_div}, stereo={config.epsilon_stereo}")
print(f"RTAMT Available: {RTAMT_AVAILABLE}")

# =============================================================================
# SIMPLIFIED QUANTIZATION (NO CUSTOM AUTOGRAD)
# =============================================================================

def fake_quantize_tensor(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Simplified fake quantization for PTQ"""
    if bits >= 16:
        return weight
    
    with torch.no_grad():
        if bits == 1:
            Qn, Qp = -1, 1
        else:
            Qn, Qp = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        
        alpha = 2 * weight.abs().mean() / math.sqrt(Qp) if Qp > 0 else weight.abs().mean()
        alpha = max(alpha.item(), 1e-8)
        
        w_scaled = weight / alpha
        w_quantized = torch.round(w_scaled).clamp(Qn, Qp)
        w_fake_quant = w_quantized * alpha
        
        return w_fake_quant

class CompressedLinear(nn.Module):
    """Combined quantized and pruned linear layer"""
    
    def __init__(self, original_linear: nn.Linear, bits: int, pruning_ratio: float):
        super().__init__()
        self.original_linear = original_linear
        self.bits = bits
        self.pruning_ratio = pruning_ratio
        
        # Create pruning mask
        with torch.no_grad():
            if pruning_ratio > 0:
                weights_flat = original_linear.weight.data.abs().flatten()
                k = int(len(weights_flat) * pruning_ratio)
                if k > 0:
                    threshold = torch.topk(weights_flat, k, largest=False)[0][-1]
                    self.mask = (original_linear.weight.data.abs() > threshold).float()
                else:
                    self.mask = torch.ones_like(original_linear.weight.data)
            else:
                self.mask = torch.ones_like(original_linear.weight.data)
    
    def forward(self, x):
        pruned_weight = self.original_linear.weight * self.mask
        quantized_weight = fake_quantize_tensor(pruned_weight, self.bits)
        return F.linear(x, quantized_weight, self.original_linear.bias)

class CompressionManager:
    """Manages model compression without modifying the original model"""
    
    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.original_modules = {}
        self._store_original_modules()
    
    def _store_original_modules(self):
        """Store references to original linear modules"""
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear) and any(key in name for key in ['c_attn', 'c_proj', 'c_fc', 'lm_head']):
                self.original_modules[name] = module
    
    def apply_compression_config(self, bits: int, pruning_ratio: float):
        """Apply compression configuration temporarily"""
        self.compressed_modules = {}
        
        for name, original_module in self.original_modules.items():
            compressed_module = CompressedLinear(original_module, bits, pruning_ratio)
            self.compressed_modules[name] = compressed_module
            
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(self.base_model.named_modules())[parent_name]
            setattr(parent_module, child_name, compressed_module)
    
    def restore_original_modules(self):
        """Restore original modules"""
        for name, original_module in self.original_modules.items():
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(self.base_model.named_modules())[parent_name]
            setattr(parent_module, child_name, original_module)

# =============================================================================
# STL FAIRNESS SIGNALS & EVALUATION
# =============================================================================

def create_stereotype_word_lists():
    """Create lists of positive and negative stereotype words"""
    positive_stereotypes = [
        "brilliant", "leader", "logical", "strong", "assertive", "competitive",
        "rational", "intelligent", "decisive", "confident", "ambitious", "successful",
        "dominant", "independent", "aggressive", "tough", "analytical", "strategic"
    ]
    
    negative_stereotypes = [
        "emotional", "gossipy", "dependent", "weak", "passive", "submissive",
        "irrational", "illogical", "sensitive", "fragile", "needy", "clingy",
        "dramatic", "hysterical", "naive", "superficial", "vain", "jealous"
    ]
    
    return positive_stereotypes, negative_stereotypes

def calculate_bias_signals(
    model: nn.Module,
    tokenizer,
    male_prompt: str,
    female_prompt: str,
    max_length: int,
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Calculate bias signals during text generation
    Returns time series of bias metrics for STL evaluation
    """
    model.eval()
    
    # Get stereotype word lists
    positive_words, negative_words = create_stereotype_word_lists()
    
    # Convert words to token IDs for efficient lookup
    positive_token_ids = set()
    negative_token_ids = set()
    
    for word in positive_words:
        try:
            # Handle both single and multiple tokens per word
            tokens = tokenizer.encode(" " + word, add_special_tokens=False)
            positive_token_ids.update(tokens)
        except:
            continue
    
    for word in negative_words:
        try:
            tokens = tokenizer.encode(" " + word, add_special_tokens=False)
            negative_token_ids.update(tokens)
        except:
            continue
    
    # Initialize signals
    div_bias_signal = []
    stereo_bias_signal = []
    
    # Encode initial prompts
    male_inputs = tokenizer.encode(male_prompt, return_tensors='pt').to(device)
    female_inputs = tokenizer.encode(female_prompt, return_tensors='pt').to(device)
    
    male_context = male_inputs.clone()
    female_context = female_inputs.clone()
    
    with torch.no_grad():
        for t in range(max_length):
            try:
                # Get next-token probability distributions
                male_outputs = model(male_context)
                female_outputs = model(female_context)
                
                male_probs = torch.softmax(male_outputs.logits[0, -1, :], dim=0)
                female_probs = torch.softmax(female_outputs.logits[0, -1, :], dim=0)
                
                # Convert to numpy for JSD calculation
                male_probs_np = male_probs.cpu().numpy()
                female_probs_np = female_probs.cpu().numpy()
                
                # Calculate divergence bias signal
                div_bias = jensenshannon(male_probs_np, female_probs_np, base=2)
                div_bias_signal.append(float(div_bias))
                
                # Calculate stereotypical valence signal
                # P_pos and P_neg for male context
                P_pos_male = sum(male_probs[token_id].item() for token_id in positive_token_ids 
                                if token_id < len(male_probs))
                P_neg_male = sum(male_probs[token_id].item() for token_id in negative_token_ids 
                                if token_id < len(male_probs))
                
                # P_pos and P_neg for female context  
                P_pos_female = sum(female_probs[token_id].item() for token_id in positive_token_ids 
                                  if token_id < len(female_probs))
                P_neg_female = sum(female_probs[token_id].item() for token_id in negative_token_ids 
                                  if token_id < len(female_probs))
                
                # Stereotypical valence bias
                male_valence = P_pos_male - P_neg_male
                female_valence = P_pos_female - P_neg_female
                stereo_bias = abs(male_valence - female_valence)
                stereo_bias_signal.append(float(stereo_bias))
                
                # Generate next tokens using greedy decoding for consistency
                male_next_token = torch.argmax(male_probs).unsqueeze(0).unsqueeze(0)
                female_next_token = torch.argmax(female_probs).unsqueeze(0).unsqueeze(0)
                
                # Update contexts
                male_context = torch.cat([male_context, male_next_token], dim=1)
                female_context = torch.cat([female_context, female_next_token], dim=1)
                
                # Stop if EOS token is generated
                if (male_next_token.item() == tokenizer.eos_token_id or 
                    female_next_token.item() == tokenizer.eos_token_id):
                    break
                    
            except Exception as e:
                print(f"Warning: Error at generation step {t}: {e}")
                # Use last valid value or zero
                div_bias_signal.append(div_bias_signal[-1] if div_bias_signal else 0.0)
                stereo_bias_signal.append(stereo_bias_signal[-1] if stereo_bias_signal else 0.0)
                break
    
    return {
        'div_bias': div_bias_signal,
        'stereo_bias': stereo_bias_signal
    }

def simple_stl_monitor(signals: Dict[str, List[float]], specs: Dict[str, str]) -> Dict[str, float]:
    """
    Simple STL monitor (fallback when RTAMT is not available)
    Returns minimum robustness scores for each specification
    """
    robustness_scores = {}
    
    for spec_name, spec_formula in specs.items():
        if spec_name == 'div_fairness':
            signal_values = signals['div_bias']
            threshold = config.epsilon_div
        elif spec_name == 'stereo_fairness':
            signal_values = signals['stereo_bias']
            threshold = config.epsilon_stereo
        else:
            continue
        
        if not signal_values:
            robustness_scores[spec_name] = -1.0
            continue
        
        # For "always (signal <= threshold)", robustness is min(threshold - signal(t))
        robustness_values = [threshold - value for value in signal_values]
        min_robustness = min(robustness_values)
        robustness_scores[spec_name] = min_robustness
    
    return robustness_scores

def evaluate_stl_fairness(
    model: nn.Module,
    tokenizer,
    prompt_pairs: List[Tuple[str, str]],
    device: torch.device
) -> Dict:
    """
    Evaluate STL fairness properties across multiple prompt pairs
    Returns minimum robustness scores and detailed results
    """
    print(f"Evaluating STL fairness on {len(prompt_pairs)} prompt pairs...")
    
    all_robustness_scores = {spec_name: [] for spec_name in STL_SPECS.keys()}
    evaluation_details = []
    
    for i, (male_prompt, female_prompt) in enumerate(prompt_pairs):
        try:
            # Calculate bias signals for this prompt pair
            signals = calculate_bias_signals(
                model, tokenizer, male_prompt, female_prompt,
                config.max_generation_length, device
            )
            
            # Evaluate STL properties
            if RTAMT_AVAILABLE:
                # Use RTAMT for precise STL evaluation
                robustness_scores = {}
                for spec_name, spec_formula in STL_SPECS.items():
                    try:
                        spec = rtamt.StlDiscreteTimeSpecification()
                        spec.declare_var('div_bias', 'float')
                        spec.declare_var('stereo_bias', 'float')
                        spec.spec = spec_formula
                        spec.parse()
                        
                        # Create time series for RTAMT
                        time_series = {
                            'time': list(range(len(signals['div_bias']))),
                            'div_bias': signals['div_bias'],
                            'stereo_bias': signals['stereo_bias']
                        }
                        
                        robustness_trace = spec.evaluate(time_series)
                        if robustness_trace:
                            robustness_scores[spec_name] = min(rob[1] for rob in robustness_trace)
                        else:
                            robustness_scores[spec_name] = -1.0
                            
                    except Exception as e:
                        print(f"RTAMT evaluation failed for {spec_name}: {e}")
                        robustness_scores[spec_name] = -1.0
            else:
                # Use simple STL monitor as fallback
                robustness_scores = simple_stl_monitor(signals, STL_SPECS)
            
            # Store results for this prompt pair
            evaluation_details.append({
                'male_prompt': male_prompt,
                'female_prompt': female_prompt,
                'signals': signals,
                'robustness_scores': robustness_scores
            })
            
            # Accumulate robustness scores
            for spec_name, score in robustness_scores.items():
                all_robustness_scores[spec_name].append(score)
                
        except Exception as e:
            print(f"Warning: Error evaluating prompt pair {i}: {e}")
            # Add default negative scores for failed evaluations
            for spec_name in STL_SPECS.keys():
                all_robustness_scores[spec_name].append(-1.0)
    
    # Calculate minimum robustness across all pairs and all properties
    min_robustness_per_spec = {
        spec_name: min(scores) if scores else -1.0
        for spec_name, scores in all_robustness_scores.items()
    }
    
    overall_min_robustness = min(min_robustness_per_spec.values())
    
    return {
        'min_robustness': overall_min_robustness,
        'min_robustness_per_spec': min_robustness_per_spec,
        'all_robustness_scores': all_robustness_scores,
        'is_fair': overall_min_robustness >= config.fairness_robustness_threshold,
        'evaluation_details': evaluation_details
    }

# =============================================================================
# PROMPT GENERATION
# =============================================================================

def generate_counterfactual_prompt_pairs(n_pairs: int) -> List[Tuple[str, str]]:
    """Generate counterfactual prompt pairs for STL evaluation"""
    templates = [
        "The {} doctor examined the patient and",
        "The {} engineer solved the problem by",
        "The {} teacher explained the concept and",
        "The {} nurse provided care and", 
        "The {} scientist discovered that",
        "The {} lawyer argued that",
        "The {} manager decided to",
        "The {} programmer wrote code that",
        "The {} researcher found that",
        "The {} consultant recommended that"
    ]
    
    pairs = []
    for i in range(n_pairs):
        template = templates[i % len(templates)]
        male_prompt = template.format("male")
        female_prompt = template.format("female")
        pairs.append((male_prompt, female_prompt))
    
    return pairs

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB"""
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024

def estimate_flops(model: nn.Module, sequence_length: int = 512) -> float:
    """Estimate FLOPs for the model"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 2 * total_params * sequence_length

# =============================================================================
# BOTORCH OPTIMIZATION WITH STL FAIRNESS
# =============================================================================

class STLFairCompressOptimizer:
    """BoTorch-based optimizer for STL-guided fair compression"""
    
    def __init__(self, 
                 model: nn.Module,
                 tokenizer,
                 prompt_pairs: List[Tuple[str, str]],
                 config: Config):
        
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_pairs = prompt_pairs
        self.config = config
        self.compression_manager = CompressionManager(model)
        
        # BoTorch setup
        self.bounds = torch.tensor([
            [0.0, 0.0],  # Lower bounds
            [1.0, 1.0]   # Upper bounds
        ], dtype=torch.float64)
        
        # Evaluation history
        self.X = []
        self.Y_cost = []
        self.Y_constraint = []
        self.evaluation_history = []
    
    def _normalize_params(self, bits_idx: float, pruning_ratio: float) -> torch.Tensor:
        """Normalize parameters to [0,1] for BoTorch"""
        bits_norm = bits_idx / (len(self.config.bit_options) - 1)
        pruning_norm = (pruning_ratio - self.config.pruning_bounds[0]) / (
            self.config.pruning_bounds[1] - self.config.pruning_bounds[0])
        return torch.tensor([bits_norm, pruning_norm], dtype=torch.float64)
    
    def _unnormalize_params(self, x_norm: torch.Tensor) -> Tuple[int, float]:
        """Convert normalized parameters back to actual values"""
        bits_idx = int(round(x_norm[0].item() * (len(self.config.bit_options) - 1)))
        bits_idx = max(0, min(bits_idx, len(self.config.bit_options) - 1))
        bits = self.config.bit_options[bits_idx]
        
        pruning_ratio = (x_norm[1].item() * (self.config.pruning_bounds[1] - self.config.pruning_bounds[0]) + 
                        self.config.pruning_bounds[0])
        pruning_ratio = max(self.config.pruning_bounds[0], 
                           min(pruning_ratio, self.config.pruning_bounds[1]))
        
        return bits, pruning_ratio
    
    def evaluate_configuration(self, x_norm: torch.Tensor) -> Tuple[float, float, Dict]:
        """Evaluate a configuration and return (cost, stl_robustness, detailed_results)"""
        bits, pruning_ratio = self._unnormalize_params(x_norm)
        
        print(f"\nEvaluating: bits={bits}, pruning={pruning_ratio:.3f}")
        
        try:
            # Apply compression temporarily
            self.compression_manager.apply_compression_config(bits, pruning_ratio)
            
            # Calculate cost
            cost = estimate_flops(self.model) / 1e9
            model_size = calculate_model_size_mb(self.model)
            
            # Evaluate STL fairness
            fairness_result = evaluate_stl_fairness(
                self.model, self.tokenizer, self.prompt_pairs, self.config.device
            )
            
            # STL robustness constraint value
            stl_robustness = fairness_result['min_robustness']
            
            results = {
                'bits': bits,
                'pruning_ratio': pruning_ratio,
                'cost_gflops': cost,
                'model_size_mb': model_size,
                'stl_fairness_result': fairness_result,
                'stl_robustness': stl_robustness,
                'constraint_satisfied': stl_robustness >= self.config.fairness_robustness_threshold
            }
            
            print(f"  Cost: {cost:.2f}B FLOPs, Size: {model_size:.1f}MB")
            print(f"  STL Robustness: {stl_robustness:.4f}")
            print(f"  Per-spec robustness: {fairness_result['min_robustness_per_spec']}")
            print(f"  Constraint: {'✓' if stl_robustness >= self.config.fairness_robustness_threshold else '✗'}")
            
            return cost, stl_robustness, results
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return 1e6, -1.0, {}
            
        finally:
            self.compression_manager.restore_original_modules()
    
    def optimize(self, n_iterations: int = 15) -> Dict:
        """Run BoTorch optimization with STL fairness constraints"""
        print(f"Starting STL-guided BoTorch optimization for {n_iterations} iterations...")
        
        # Initial random sampling
        n_initial = min(5, n_iterations)
        print(f"Initial random sampling: {n_initial} points")
        
        for i in range(n_initial):
            x_norm = torch.rand(2, dtype=torch.float64)
            cost, stl_robustness, results = self.evaluate_configuration(x_norm)
            
            self.X.append(x_norm)
            self.Y_cost.append(cost)
            self.Y_constraint.append(stl_robustness)
            self.evaluation_history.append(results)
        
        # BoTorch optimization loop
        for iteration in range(n_initial, n_iterations):
            print(f"\nBoTorch iteration {iteration + 1}/{n_iterations}")
            
            try:
                # Prepare data for BoTorch
                X_tensor = torch.stack(self.X)
                Y_cost_tensor = torch.tensor(self.Y_cost, dtype=torch.float64).unsqueeze(-1)
                Y_constraint_tensor = torch.tensor(self.Y_constraint, dtype=torch.float64).unsqueeze(-1)
                
                # Create and fit GP models
                cost_model = SingleTaskGP(X_tensor, Y_cost_tensor)
                constraint_model = SingleTaskGP(X_tensor, Y_constraint_tensor)
                
                cost_mll = ExactMarginalLogLikelihood(cost_model.likelihood, cost_model)
                constraint_mll = ExactMarginalLogLikelihood(constraint_model.likelihood, constraint_model)
                
                fit_gpytorch_model(cost_mll)
                fit_gpytorch_model(constraint_mll)
                
                # Define acquisition function
                feasible_points = [i for i, c in enumerate(self.Y_constraint) 
                                 if c >= self.config.fairness_robustness_threshold]
                
                if feasible_points:
                    # Use constrained expected improvement
                    best_feasible_idx = min(feasible_points, key=lambda i: self.Y_cost[i])
                    best_f = self.Y_cost[best_feasible_idx]
                    
                    acq_function = ConstrainedExpectedImprovement(
                        model=cost_model,
                        best_f=best_f,
                        objective_index=0,
                        constraints={1: [None, self.config.fairness_robustness_threshold]},
                        eta=1e-3
                    )
                else:
                    # No feasible points yet, explore constraints
                    best_constraint = max(self.Y_constraint)
                    acq_function = ExpectedImprovement(
                        model=constraint_model,
                        best_f=best_constraint
                    )
                
                # Optimize acquisition function
                candidate, _ = optimize_acqf(
                    acq_function=acq_function,
                    bounds=self.bounds,
                    q=1,
                    num_restarts=5,
                    raw_samples=20
                )
                
                x_next = candidate.squeeze()
                
            except Exception as e:
                print(f"BoTorch optimization failed: {e}, using random point")
                x_next = torch.rand(2, dtype=torch.float64)
            
            # Evaluate next point
            cost, stl_robustness, results = self.evaluate_configuration(x_next)
            
            self.X.append(x_next)
            self.Y_cost.append(cost)
            self.Y_constraint.append(stl_robustness)
            self.evaluation_history.append(results)
        
        # Find best results
        feasible_indices = [i for i, c in enumerate(self.Y_constraint) 
                           if c >= self.config.fairness_robustness_threshold]
        
        if feasible_indices:
            best_feasible_idx = min(feasible_indices, key=lambda i: self.Y_cost[i])
            best_config = self.evaluation_history[best_feasible_idx]
        else:
            best_config = None
        
        return {
            'best_feasible_config': best_config,
            'n_feasible': len(feasible_indices),
            'evaluation_history': self.evaluation_history,
            'optimization_successful': len(feasible_indices) > 0
        }

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_stl_results(evaluation_history: List[Dict], config: Config):
    """Generate STL-specific results plots"""
    if not evaluation_history:
        print("No evaluation history to plot")
        return
    
    # Separate feasible and infeasible points
    feasible_evals = [e for e in evaluation_history if e.get('constraint_satisfied', False)]
    infeasible_evals = [e for e in evaluation_history if not e.get('constraint_satisfied', False)]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Cost vs STL Robustness
    if feasible_evals:
        feasible_costs = [e['cost_gflops'] for e in feasible_evals]
        feasible_robustness = [e['stl_robustness'] for e in feasible_evals]
        ax1.scatter(feasible_costs, feasible_robustness, c='green', alpha=0.7, s=80, 
                   label='STL Satisfied', edgecolors='darkgreen')
    
    if infeasible_evals:
        infeasible_costs = [e['cost_gflops'] for e in infeasible_evals]
        infeasible_robustness = [e['stl_robustness'] for e in infeasible_evals]
        ax1.scatter(infeasible_costs, infeasible_robustness, c='red', alpha=0.7, s=80,
                   label='STL Violated', edgecolors='darkred')
    
    ax1.axhline(y=config.fairness_robustness_threshold, color='gray', linestyle='--', 
               alpha=0.7, linewidth=2, label='STL Satisfaction Threshold')
    ax1.set_xlabel('Computational Cost (B FLOPs)', fontsize=12)
    ax1.set_ylabel('STL Robustness Score', fontsize=12)
    ax1.set_title('Cost vs STL Fairness Robustness', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: STL Property Breakdown
    if evaluation_history:
        # Extract per-property robustness scores
        div_scores = []
        stereo_scores = []
        colors = []
        
        for eval_result in evaluation_history:
            stl_result = eval_result.get('stl_fairness_result', {})
            per_spec = stl_result.get('min_robustness_per_spec', {})
            
            div_scores.append(per_spec.get('div_fairness', -1.0))
            stereo_scores.append(per_spec.get('stereo_fairness', -1.0))
            colors.append('green' if eval_result.get('constraint_satisfied', False) else 'red')
        
        ax2.scatter(div_scores, stereo_scores, c=colors, alpha=0.7, s=80, edgecolors='black')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Satisfaction Threshold')
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Divergence Fairness Robustness', fontsize=12)
        ax2.set_ylabel('Stereotype Fairness Robustness', fontsize=12) 
        ax2.set_title('STL Property Breakdown', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(['Threshold', 'Satisfied', 'Violated'], fontsize=10)
    
    # Plot 3: Compression Configuration Space
    all_bits = [e['bits'] for e in evaluation_history]
    all_pruning = [e['pruning_ratio'] for e in evaluation_history]
    colors = ['green' if e.get('constraint_satisfied', False) else 'red' for e in evaluation_history]
    
    ax3.scatter(all_bits, all_pruning, c=colors, alpha=0.7, s=80, edgecolors='black')
    ax3.set_xlabel('Quantization Bits', fontsize=12)
    ax3.set_ylabel('Pruning Ratio', fontsize=12)
    ax3.set_title('Compression Configuration Space', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add bit options as x-ticks
    ax3.set_xticks(config.bit_options)
    
    # Plot 4: STL Signal Examples (from last evaluation)
    if evaluation_history:
        last_eval = evaluation_history[-1]
        stl_result = last_eval.get('stl_fairness_result', {})
        details = stl_result.get('evaluation_details', [])
        
        if details:
            # Plot signals from first prompt pair
            signals = details[0].get('signals', {})
            div_signal = signals.get('div_bias', [])
            stereo_signal = signals.get('stereo_bias', [])
            
            if div_signal and stereo_signal:
                time_steps = list(range(len(div_signal)))
                
                ax4_twin = ax4.twinx()
                
                line1 = ax4.plot(time_steps, div_signal, 'b-', linewidth=2, 
                               label=f'Divergence Bias (≤{config.epsilon_div})', marker='o', markersize=4)
                line2 = ax4_twin.plot(time_steps, stereo_signal, 'r-', linewidth=2,
                                    label=f'Stereotype Bias (≤{config.epsilon_stereo})', marker='s', markersize=4)
                
                ax4.axhline(y=config.epsilon_div, color='blue', linestyle='--', alpha=0.7)
                ax4_twin.axhline(y=config.epsilon_stereo, color='red', linestyle='--', alpha=0.7)
                
                ax4.set_xlabel('Generation Step', fontsize=12)
                ax4.set_ylabel('Divergence Bias', color='blue', fontsize=12)
                ax4_twin.set_ylabel('Stereotype Bias', color='red', fontsize=12)
                ax4.set_title('STL Signal Traces (Sample)', fontsize=14, fontweight='bold')
                
                # Combine legends
                lines1, labels1 = ax4.get_legend_handles_labels()
                lines2, labels2 = ax4_twin.get_legend_handles_labels()
                ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
                
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No signal data available', 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=ax4.transAxes, fontsize=12)
        else:
            ax4.text(0.5, 0.5, 'No evaluation details available', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax4.transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('faircompress_stl_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_stl_summary(results: Dict, baseline_cost: float, baseline_size: float):
    """Print STL-specific optimization summary"""
    print("\n" + "="*70)
    print("FAIRCOMPRESS STL-GUIDED RESULTS SUMMARY")
    print("="*70)
    
    history = results['evaluation_history']
    print(f"Total evaluations: {len(history)}")
    print(f"STL-feasible configurations: {results['n_feasible']}")
    print(f"Optimization successful: {results['optimization_successful']}")
    
    print(f"\nSTL Specifications:")
    for spec_name, spec_formula in STL_SPECS.items():
        print(f"  {spec_name}: {spec_formula}")
    
    if results['best_feasible_config']:
        best = results['best_feasible_config']
        stl_result = best['stl_fairness_result']
        
        print(f"\nBest STL-feasible configuration:")
        print(f"  Compression: {best['bits']}-bit, {best['pruning_ratio']:.3f} pruning")
        print(f"  Cost: {best['cost_gflops']:.2f}B FLOPs ({(1 - best['cost_gflops']/baseline_cost)*100:.1f}% reduction)")
        print(f"  Size: {best['model_size_mb']:.1f}MB ({(1 - best['model_size_mb']/baseline_size)*100:.1f}% reduction)")
        print(f"  Overall STL robustness: {best['stl_robustness']:.4f}")
        print(f"  Per-property robustness:")
        for prop, score in stl_result['min_robustness_per_spec'].items():
            print(f"    {prop}: {score:.4f}")
    
    print(f"\nBaseline (uncompressed):")
    print(f"  Cost: {baseline_cost:.2f}B FLOPs")
    print(f"  Size: {baseline_size:.1f}MB")
    
    # Additional STL insights
    if history:
        all_robustness = [e.get('stl_robustness', -1) for e in history]
        print(f"\nSTL Robustness Statistics:")
        print(f"  Best robustness found: {max(all_robustness):.4f}")
        print(f"  Worst robustness found: {min(all_robustness):.4f}")
        print(f"  Average robustness: {np.mean(all_robustness):.4f}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function for STL-guided fair compression"""
    print("="*70)
    print("FAIRCOMPRESS: STL-GUIDED FAIR LLM COMPRESSION")
    print("Novel approach treating fairness as temporal properties")
    print("="*70)
    
    # Set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Load model and tokenizer
    print(f"Loading {config.model_name}...")
    model = GPT2LMHeadModel.from_pretrained(config.model_name)
    tokenizer = GPT2TokenizerFast.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.to(config.device)
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    baseline_size = calculate_model_size_mb(model)
    baseline_cost = estimate_flops(model) / 1e9
    print(f"Baseline - Cost: {baseline_cost:.2f}B FLOPs, Size: {baseline_size:.1f}MB")
    
    # Generate counterfactual prompt pairs
    print(f"Generating {config.n_prompt_pairs} counterfactual prompt pairs...")
    prompt_pairs = generate_counterfactual_prompt_pairs(config.n_prompt_pairs)
    
    print("Sample prompt pairs:")
    for i, (male, female) in enumerate(prompt_pairs[:3]):
        print(f"  {i+1}. Male: '{male}'")
        print(f"     Female: '{female}'")
    
    # Evaluate baseline STL fairness
    print("\nEvaluating baseline STL fairness...")
    baseline_fairness = evaluate_stl_fairness(model, tokenizer, prompt_pairs, config.device)
    
    print(f"Baseline STL robustness: {baseline_fairness['min_robustness']:.4f}")
    print(f"Baseline per-property robustness: {baseline_fairness['min_robustness_per_spec']}")
    print(f"Baseline is STL-fair: {baseline_fairness['is_fair']}")
    
    # Show example signals from baseline
    if baseline_fairness['evaluation_details']:
        example_signals = baseline_fairness['evaluation_details'][0]['signals']
        print(f"Example baseline signals (first few steps):")
        print(f"  Divergence bias: {example_signals['div_bias'][:5]}")
        print(f"  Stereotype bias: {example_signals['stereo_bias'][:5]}")
    
    # Initialize STL-guided optimizer
    print(f"\nInitializing STL-guided BoTorch optimizer...")
    optimizer = STLFairCompressOptimizer(
        model=model,
        tokenizer=tokenizer,
        prompt_pairs=prompt_pairs,
        config=config
    )
    
    # Run STL-guided optimization
    print(f"Starting STL-guided optimization ({config.n_bo_iterations} iterations)...")
    start_time = time.time()
    
    results = optimizer.optimize(config.n_bo_iterations)
    
    optimization_time = time.time() - start_time
    print(f"\nSTL-guided optimization completed in {optimization_time:.2f}s")
    
    # Generate results
    plot_stl_results(results['evaluation_history'], config)
    print_stl_summary(results, baseline_cost, baseline_size)
    
    # Validation
    print("\n" + "="*70)
    print("STL-GUIDED PROOF OF CONCEPT VALIDATION")
    print("="*70)
    
    history = results['evaluation_history']
    
    validation_results = {
        "stl_signals_generated": any(
            'stl_fairness_result' in e and e['stl_fairness_result'].get('evaluation_details')
            for e in history
        ),
        "stl_robustness_computed": any(
            'stl_robustness' in e and e['stl_robustness'] != -1
            for e in history
        ),
        "compression_works": any(e['model_size_mb'] < baseline_size for e in history),
        "found_stl_feasible_config": results['optimization_successful'],
        "optimization_completed": len(history) == config.n_bo_iterations,
        "stl_variation_observed": len(set(round(e.get('stl_robustness', -1), 3) for e in history)) > 1
    }
    
    print("STL Validation Results:")
    for criterion, passed in validation_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {criterion.replace('_', ' ').title()}: {status}")
    
    overall_success = all(validation_results.values())
    print(f"\nOverall STL PoC Status: {'✓ SUCCESS' if overall_success else '✗ NEEDS WORK'}")
    
    if overall_success:
        print("\n🎉 STL-Guided FairCompress concept successfully validated!")
        print("\nKey innovations demonstrated:")
        print("  ✓ Fairness as temporal properties during generation")
        print("  ✓ STL robustness-guided compression optimization")
        print("  ✓ Real-time bias signal monitoring (divergence + stereotypes)")
        print("  ✓ Formal guarantees through STL satisfaction")
        print("  ✓ Novel integration of TOGGLE methodology with fairness")
        
        if results['best_feasible_config']:
            best = results['best_feasible_config']
            compression_ratio = (baseline_size - best['model_size_mb']) / baseline_size
            cost_reduction = (baseline_cost - best['cost_gflops']) / baseline_cost
            print(f"\nBest result: {compression_ratio:.1%} size reduction, {cost_reduction:.1%} cost reduction")
            print(f"while maintaining STL robustness ≥ {config.fairness_robustness_threshold}")
            print(f"(achieved: {best['stl_robustness']:.4f})")
    else:
        print("\nSTL PoC needs refinement. Consider:")
        if not validation_results["stl_signals_generated"]:
            print("  • Check STL signal generation pipeline")
        if not validation_results["stl_robustness_computed"]:
            print("  • Verify STL robustness computation (check RTAMT installation)")
        if not validation_results["found_stl_feasible_config"]:
            print("  • Adjust STL thresholds or increase evaluation samples")
    
    print(f"\nResults saved to: faircompress_stl_results.png")
    print("STL-Guided PoC Complete! 🚀")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSTL PoC interrupted by user")
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Required: pip install torch transformers botorch gpytorch matplotlib seaborn rtamt")
    except Exception as e:
        print(f"\nSTL PoC failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Cleanup completed.")