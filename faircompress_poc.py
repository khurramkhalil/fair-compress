# -*- coding: utf-8 -*-
"""
FairCompress: STL-Guided Fair LLM Compression
Version 1.0 - Research-Grade Script

This framework compresses LLMs by guiding a Bayesian Optimizer with formal
fairness properties defined using Signal Temporal Logic (STL). It co-optimizes
for computational efficiency (FLOPs) and certified temporal fairness.

Author: Research Team
"""

import math
import time
import warnings
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Core ML/Scientific Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from scipy.spatial.distance import jensenshannon

# BoTorch for Bayesian Optimization
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ConstrainedExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

# RTAMT for STL Evaluation
import rtamt

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=botorch.exceptions.warnings.BadInitialCandidatesWarning)

# =============================================================================
# FRAMEWORK CONFIGURATION
# =============================================================================

@dataclass
class FairCompressConfig:
    """Configuration for the STL-guided fair compression framework."""
    # Model and Environment
    model_name: str = "gpt2"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42

    # Bayesian Optimization
    n_bo_iterations: int = 25  # Total BO iterations
    n_initial_random: int = 5  # Number of initial random points for BO

    # Fairness Evaluation
    n_prompt_pairs: int = 10  # Number of counterfactual pairs to evaluate per config
    max_generation_length: int = 20  # STL trace length

    # STL Fairness Specifications & Thresholds
    epsilon_div: float = 0.05    # Max allowed JSD divergence
    epsilon_stereo: float = 0.02  # Max allowed stereotype bias
    stl_robustness_threshold: float = 0.0  # STL robustness must be >= 0

    # Compression Search Space
    bit_options: List[int] = None
    pruning_bounds: Tuple[float, float] = (0.0, 0.5)
    
    def __post_init__(self):
        if self.bit_options is None:
            self.bit_options = [4, 8, 16] # Removed 2-bit as it's often unstable without advanced techniques

config = FairCompressConfig()

# Define STL Specifications using the configured thresholds
STL_SPECS = {
    'div_fairness': f"always (div_bias <= {config.epsilon_div})",
    'stereo_fairness': f"always (stereo_bias <= {config.epsilon_stereo})"
}

# =============================================================================
# STATELESS COMPRESSION VIA PYTORCH HOOKS
# =============================================================================

def fake_quantize_tensor(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Simplified post-training fake quantization for inference."""
    if bits >= 16:
        return weight
    
    # Detach to prevent gradients from flowing through this heuristic process
    w = weight.detach()
    
    if bits == 1:
        return torch.sign(w) * w.abs().mean()
        
    Qn, Qp = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    
    # Min-max scaling for weights
    alpha = (w.max() - w.min()) / (Qp - Qn)
    alpha = alpha.clamp(min=1e-8)
    
    w_quantized = torch.round(w / alpha).clamp(Qn, Qp)
    w_fake_quant = w_quantized * alpha
    
    return w_fake_quant

class CompressionHook:
    """A forward hook to apply temporary compression to a linear layer."""
    def __init__(self):
        self.bits = 16
        self.pruning_ratio = 0.0
        self.mask = None
        self.original_weight = None

    def setup(self, module: nn.Linear, bits: int, pruning_ratio: float):
        self.bits = bits
        self.pruning_ratio = pruning_ratio
        self.original_weight = module.weight.data
        self.mask = self.create_pruning_mask(self.original_weight, self.pruning_ratio)
        self.mask = self.mask.to(self.original_weight.device)

    def create_pruning_mask(self, weight: torch.Tensor, ratio: float) -> torch.Tensor:
        if ratio <= 0:
            return torch.ones_like(weight)
        
        weights_flat = weight.flatten().abs()
        k = int(len(weights_flat) * ratio)
        if k == 0:
            return torch.ones_like(weight)
        
        threshold = torch.kthvalue(weights_flat, k).values
        return (weight.abs() > threshold).float()

    def __call__(self, module, input, output):
        # Apply pruning
        pruned_weight = self.original_weight * self.mask
        # Apply quantization
        quantized_weight = fake_quantize_tensor(pruned_weight, self.bits)
        # Recompute output with compressed weight
        return F.linear(input[0], quantized_weight, module.bias)

class CompressionManager:
    """Manages applying and removing compression hooks from a model."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = {}
        self._identify_target_modules()

    def _identify_target_modules(self):
        """Finds linear layers to apply compression to."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(key in name for key in ['c_attn', 'c_proj', 'c_fc', 'lm_head']):
                self.hooks[name] = CompressionHook()

    def apply_compression_config(self, bits: int, pruning_ratio: float):
        """Register forward hooks with the given compression config."""
        for name, hook in self.hooks.items():
            module = dict(self.model.named_modules())[name]
            hook.setup(module, bits, pruning_ratio)
            handle = module.register_forward_hook(hook)
            hook.handle = handle  # Store handle to remove it later

    def restore_original_model(self):
        """Remove all compression hooks."""
        for hook in self.hooks.values():
            if hasattr(hook, 'handle'):
                hook.handle.remove()

# =============================================================================
# STL FAIRNESS EVALUATION
# =============================================================================

def get_stereotype_token_ids(tokenizer) -> Tuple[set, set]:
    """Pre-computes token IDs for stereotype words."""
    positive_words, negative_words = [
        "brilliant", "leader", "logical", "strong", "assertive", "intelligent", "decisive"
    ], [
        "emotional", "gossipy", "dependent", "weak", "passive", "sensitive", "fragile"
    ]
    
    pos_ids, neg_ids = set(), set()
    for word in positive_words:
        pos_ids.update(tokenizer.encode(" " + word, add_special_tokens=False))
    for word in negative_words:
        neg_ids.update(tokenizer.encode(" " + word, add_special_tokens=False))
        
    return pos_ids, neg_ids

def calculate_bias_signals(
    model: nn.Module,
    tokenizer,
    prompt_pair: Tuple[str, str],
    stereotype_ids: Tuple[set, set],
    config: FairCompressConfig
) -> Dict[str, List[float]]:
    """Calculates time-series bias signals for one prompt pair."""
    male_prompt, female_prompt = prompt_pair
    pos_ids, neg_ids = stereotype_ids
    
    div_bias_signal, stereo_bias_signal = [], []
    male_input_ids = tokenizer.encode(male_prompt, return_tensors='pt').to(config.device)
    female_input_ids = tokenizer.encode(female_prompt, return_tensors='pt').to(config.device)

    with torch.no_grad():
        for _ in range(config.max_generation_length):
            male_logits = model(male_input_ids).logits[:, -1, :]
            female_logits = model(female_input_ids).logits[:, -1, :]
            
            male_probs = F.softmax(male_logits, dim=-1).squeeze()
            female_probs = F.softmax(female_logits, dim=-1).squeeze()
            
            # 1. Divergence Bias Signal
            div_bias = jensenshannon(male_probs.cpu().numpy(), female_probs.cpu().numpy(), base=2)
            div_bias_signal.append(float(div_bias) if not np.isnan(div_bias) else 1.0)

            # 2. Stereotype Bias Signal
            male_valence = sum(male_probs[i] for i in pos_ids) - sum(male_probs[i] for i in neg_ids)
            female_valence = sum(female_probs[i] for i in pos_ids) - sum(female_probs[i] for i in neg_ids)
            stereo_bias = abs(male_valence - female_valence)
            stereo_bias_signal.append(stereo_bias.item())
            
            # Append next token (greedy) to continue generation
            male_next_token = torch.argmax(male_probs).unsqueeze(0)
            female_next_token = torch.argmax(female_probs).unsqueeze(0)
            
            male_input_ids = torch.cat([male_input_ids, male_next_token.unsqueeze(0)], dim=1)
            female_input_ids = torch.cat([female_input_ids, female_next_token.unsqueeze(0)], dim=1)
            
            if male_next_token == tokenizer.eos_token_id and female_next_token == tokenizer.eos_token_id:
                break
    
    return {'div_bias': div_bias_signal, 'stereo_bias': stereo_bias_signal}

def evaluate_stl_fairness(
    model: nn.Module,
    tokenizer,
    prompt_pairs: List[Tuple[str, str]],
    stereotype_ids: Tuple[set, set],
    parsed_stl_specs: Dict[str, Any],
    config: FairCompressConfig
) -> Dict:
    """Evaluates STL fairness and returns the minimum robustness score."""
    all_robustness_scores = {spec_name: [] for spec_name in parsed_stl_specs}
    
    for prompt_pair in prompt_pairs:
        signals = calculate_bias_signals(model, tokenizer, prompt_pair, stereotype_ids, config)
        
        for spec_name, spec_obj in parsed_stl_specs.items():
            # RTAMT requires dict of lists of [time, value] pairs
            time_series = {'time': list(range(len(signals['div_bias'])))}
            time_series.update({k: [[i, v] for i, v in enumerate(vals)] for k, vals in signals.items()})
            
            try:
                robustness = spec_obj.evaluate(time_series)
                # Robustness of an 'always' formula is the minimum value in its trace
                min_rob_for_trace = min(r[1] for r in robustness) if robustness else -1.0
            except Exception as e:
                print(f"Warning: RTAMT evaluation failed for {spec_name}: {e}")
                min_rob_for_trace = -1.0
            
            all_robustness_scores[spec_name].append(min_rob_for_trace)

    min_robustness_per_spec = {
        spec: min(scores) if scores else -1.0
        for spec, scores in all_robustness_scores.items()
    }
    
    overall_min_robustness = min(min_robustness_per_spec.values()) if min_robustness_per_spec else -1.0
    
    return {
        'min_robustness': overall_min_robustness,
        'min_robustness_per_spec': min_robustness_per_spec
    }

# =============================================================================
# BOTORCH OPTIMIZATION
# =============================================================================

class FairCompressOptimizer:
    """Manages the full BoTorch optimization loop."""
    def __init__(self, model, tokenizer, prompt_pairs, stereotype_ids, config):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_pairs = prompt_pairs
        self.stereotype_ids = stereotype_ids
        self.config = config
        self.compression_manager = CompressionManager(model)
        
        # Search space boundaries, normalized to [0, 1]
        self.bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=config.device, dtype=torch.double)
        
        # Pre-parse STL specs once for efficiency
        self.parsed_stl_specs = self._parse_stl_specs()
        
        # History
        self.X_observed = torch.empty((0, 2), device=config.device, dtype=torch.double)
        self.Y_observed = torch.empty((0, 2), device=config.device, dtype=torch.double) # (cost, robustness)
        self.evaluation_history = []
        
    def _parse_stl_specs(self):
        specs = {}
        for name, formula in STL_SPECS.items():
            spec = rtamt.StlDiscreteTimeSpecification()
            spec.declare_var('div_bias', 'float')
            spec.declare_var('stereo_bias', 'float')
            spec.spec = formula
            spec.parse()
            specs[name] = spec
        return specs

    def _unnormalize_params(self, x_norm: torch.Tensor) -> Tuple[int, float]:
        """Converts normalized [0,1] parameters to actual values."""
        bits_idx = int(round(x_norm[0].item() * (len(self.config.bit_options) - 1)))
        bits = self.config.bit_options[bits_idx]
        
        pruning_ratio = x_norm[1].item() * (self.config.pruning_bounds[1] - self.config.pruning_bounds[0]) + self.config.pruning_bounds[0]
        return bits, pruning_ratio

    def evaluate_configuration(self, x_norm_tensor: torch.Tensor):
        """The objective function for BoTorch."""
        x_norm = x_norm_tensor.squeeze()
        bits, pruning_ratio = self._unnormalize_params(x_norm)
        
        print(f"  Evaluating: bits={bits}, pruning={pruning_ratio:.3f}...")
        
        try:
            self.compression_manager.apply_compression_config(bits, pruning_ratio)
            cost = estimate_flops(self.model) / 1e12  # Normalize to TFLOPs for better GP scaling
            stl_result = evaluate_stl_fairness(self.model, self.tokenizer, self.prompt_pairs, self.stereotype_ids, self.parsed_stl_specs, self.config)
            robustness = stl_result['min_robustness']
            
            # Store full results for later analysis
            full_result = {'bits': bits, 'pruning_ratio': pruning_ratio, 'cost_tflops': cost, 'stl_robustness': robustness}
            self.evaluation_history.append(full_result)
            
        except Exception as e:
            print(f"    ERROR during evaluation: {e}")
            cost, robustness = 1e6, -1.0 # Return high cost, failed constraint
        finally:
            self.compression_manager.restore_original_model()
            
        return torch.tensor([cost, robustness], device=self.config.device, dtype=torch.double)

    def optimize(self):
        """Run the main optimization loop."""
        # Initial random sampling
        initial_x = torch.rand(self.config.n_initial_random, 2, device=self.config.device, dtype=torch.double)
        for x_pt in initial_x:
            observed_y = self.evaluate_configuration(x_pt)
            self.X_observed = torch.cat([self.X_observed, x_pt.unsqueeze(0)], dim=0)
            self.Y_observed = torch.cat([self.Y_observed, observed_y.unsqueeze(0)], dim=0)

        # BoTorch optimization loop
        for iteration in range(self.config.n_initial_random, self.config.n_bo_iterations):
            print(f"\n--- BoTorch Iteration {iteration + 1}/{self.config.n_bo_iterations} ---")
            
            # Standardize outputs for better GP fitting
            standardized_Y = Standardize(m=2)(self.Y_observed)[0]
            
            # Create and fit GP models for objective and constraint
            cost_model = SingleTaskGP(self.X_observed, standardized_Y[:, 0].unsqueeze(-1))
            constraint_model = SingleTaskGP(self.X_observed, standardized_Y[:, 1].unsqueeze(-1))
            
            model_list = ModelListGP(cost_model, constraint_model)
            mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
            fit_gpytorch_model(mll)

            # Define acquisition function
            acq_function = ConstrainedExpectedImprovement(
                model=model_list,
                best_f=standardized_Y[:, 0].min(),
                objective_index=0, # Index 0 is cost
                constraints={1: [self.config.stl_robustness_threshold, None]} # Index 1 is robustness
            )
            
            # Optimize acquisition function
            candidate, _ = optimize_acqf(
                acq_function=acq_function,
                bounds=self.bounds,
                q=1,
                num_restarts=5,
                raw_samples=20
            )
            x_next = candidate.squeeze(0)
            
            # Evaluate next point and update history
            observed_y = self.evaluate_configuration(x_next)
            self.X_observed = torch.cat([self.X_observed, x_next.unsqueeze(0)], dim=0)
            self.Y_observed = torch.cat([self.Y_observed, observed_y.unsqueeze(0)], dim=0)
            
        return self.evaluation_history

# =============================================================================
# VISUALIZATION & MAIN EXECUTION
# =============================================================================

# (Visualization and main function from previous PoC can be adapted here. 
# They will need to be updated to handle the new data format from STLFairCompressOptimizer.
# For brevity in this response, I'm omitting the plotting code, but it would be very similar
# to the one in the previous response, using the 'evaluation_history' from the optimizer.)

def main():
    """Main execution function."""
    print("="*70)
    print("FairCompress: STL-GUIDED FAIR LLM COMPRESSION (V1.0)")
    print("="*70)

    # Setup
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = GPT2LMHeadModel.from_pretrained(config.model_name).to(config.device)
    tokenizer = GPT2TokenizerFast.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    prompt_pairs = generate_counterfactual_prompt_pairs(config.n_prompt_pairs)
    stereotype_ids = get_stereotype_token_ids(tokenizer)

    # Evaluate baseline
    baseline_cost = estimate_flops(model) / 1e12
    print(f"Baseline Cost: {baseline_cost:.3f} TFLOPs")
    baseline_fairness = evaluate_stl_fairness(model, tokenizer, prompt_pairs, stereotype_ids, STLFairCompressOptimizer(model, tokenizer, [], [], config).parsed_stl_specs, config)
    print(f"Baseline STL Robustness: {baseline_fairness['min_robustness']:.4f}")

    # Run optimizer
    optimizer = STLFairCompressOptimizer(model, tokenizer, prompt_pairs, stereotype_ids, config)
    results = optimizer.optimize()

    # Plot and summarize
    # plot_stl_results(results, config)  # You would call your updated plotting function here
    # print_stl_summary(results, baseline_cost, ...)

    print("\nOptimization Complete.")
    
if __name__ == "__main__":
    if not RTAMT_AVAILABLE:
        print("Error: RTAMT library is required for this script. Please install it.")
    else:
        try:
            main()
        except Exception as e:
            import traceback
            print(f"\nAn unexpected error occurred: {e}")
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("\nScript finished.")
