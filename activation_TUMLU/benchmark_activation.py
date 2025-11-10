"""
LAPE Neuron Analysis on TUMLU Benchmark
Combines neuron activation tracking with benchmark quality evaluation.
"""

import json
import os
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============ CONFIGURATION ============
MODEL_NAME = "Tweeties/tweety-tatar-base-7b-2024-v1"
BENCHMARK_PATH = "../LAPE/data/TUMLU/crimean-tatar/biology.jsonl"

# Additional settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
BATCH_SIZE = 1  # Process one question at a time for accurate tracking

# LAPE parameters
FILTER_RATE = 0.95  # Top percentile for filtering significant neurons
TOP_RATE = 0.01  # Bottom 1% by entropy = language-specific
ACTIVATION_BAR_RATIO = 0.95  # Minimum activation probability threshold


# ============ HELPER FUNCTIONS ============


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    """Load TUMLU benchmark questions from JSONL file."""
    print(f"[1/8] Loading benchmark from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print(f"  Loaded {len(questions)} questions")
    return questions


def detect_mlp_kind(mlp) -> str:
    """Detect MLP architecture type: 'gated' (Mistral/LLaMA) or 'bloom'."""
    if (
        hasattr(mlp, "gate_proj")
        and hasattr(mlp, "up_proj")
        and hasattr(mlp, "down_proj")
    ):
        return "gated"
    if hasattr(mlp, "dense_h_to_4h") and hasattr(mlp, "dense_4h_to_h"):
        return "bloom"
    raise RuntimeError("Unknown MLP type (expected gated or bloom).")


def get_layers_and_intermediate(model) -> Tuple[List[Any], int]:
    """Get model layers and intermediate size."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = list(model.model.layers)
        first_mlp = layers[0].mlp
        kind = detect_mlp_kind(first_mlp)
        if kind == "gated":
            inter = first_mlp.gate_proj.out_features
        else:
            inter = first_mlp.dense_h_to_4h.out_features
        return layers, inter
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = list(model.transformer.h)
        first_mlp = layers[0].mlp
        kind = detect_mlp_kind(first_mlp)
        if kind == "gated":
            inter = first_mlp.gate_proj.out_features
        else:
            inter = first_mlp.dense_h_to_4h.out_features
        return layers, inter
    else:
        raise RuntimeError(
            "Cannot find model layers (.model.layers or .transformer.h)."
        )


def make_patchers(
    layers: List[Any], over_zero_ref: Dict[str, torch.Tensor]
) -> List[Any]:
    """Create patched forward functions to track neuron activations."""
    patched = []
    for li, layer in enumerate(layers):
        mlp = layer.mlp
        kind = detect_mlp_kind(mlp)

        if kind == "gated":
            gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
            act = mlp.act_fn  # SiLU

            def fwd(x, gate=gate, up=up, down=down, act=act, li=li):
                g = gate(x)
                a = act(g)
                ctr = over_zero_ref["tensor"]
                if ctr is not None:
                    pos = (a > 0).sum(dim=(0, 1))
                    ctr[li].add_(pos.to(ctr.dtype))
                u = up(x)
                y = down(a * u)
                return y

        else:  # bloom
            d14h, d4h1 = mlp.dense_h_to_4h, mlp.dense_4h_to_h
            gelu = mlp.gelu_impl

            def fwd(x, d14h=d14h, d4h1=d4h1, gelu=gelu, li=li):
                z = d14h(x)
                a = gelu(z)
                ctr = over_zero_ref["tensor"]
                if ctr is not None:
                    pos = (a > 0).sum(dim=(0, 1))
                    ctr[li].add_(pos.to(ctr.dtype))
                y = d4h1(a)
                return y

        patched.append((mlp, mlp.forward, fwd))
    return patched


def apply_patches(patched: List[Any]) -> None:
    """Apply patched forward functions."""
    for mlp, _orig, fwd in patched:
        mlp.forward = fwd


def restore_patches(patched: List[Any]) -> None:
    """Restore original forward functions."""
    for mlp, orig, _fwd in patched:
        mlp.forward = orig


def format_question_choices(question: Dict[str, Any]) -> str:
    """Format question with multiple choice options."""
    q_text = question["question"]
    choices = question["choices"]
    choice_labels = ["A", "B", "C", "D"]

    formatted = f"{q_text}\n"
    for label, choice in zip(choice_labels, choices):
        formatted += f"{label}. {choice}\n"
    formatted += "Answer:"

    return formatted


def get_predicted_answer(
    model, tokenizer, input_ids, attention_mask, debug=False
) -> Tuple[str, Dict[str, float]]:
    """Get predicted answer and log probabilities for each choice (A, B, C, D)."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]  # Last token logits
        log_probs = torch.log_softmax(logits, dim=-1)

    # Get tokens for A, B, C, D - try multiple variations
    answer_tokens = {}
    token_map = {}  # For debugging
    
    for letter in ["A", "B", "C", "D"]:
        # Try different tokenization approaches
        variations = [
            f" {letter}",
            letter,
            f"\n{letter}",
            f" {letter}.",
        ]
        
        for var in variations:
            tokens = tokenizer.encode(var, add_special_tokens=False)
            if tokens and len(tokens) == 1:
                answer_tokens[letter] = tokens[0]
                token_map[letter] = (var, tokens[0])
                break
        
        # If still not found, use the first token
        if letter not in answer_tokens:
            tokens = tokenizer.encode(letter, add_special_tokens=False)
            if tokens:
                answer_tokens[letter] = tokens[0]
                token_map[letter] = (letter, tokens[0])

    # Debug: print token mapping for first call
    if debug and len(token_map) > 0:
        print("\n[DEBUG] Answer token mapping:")
        for letter, (var, token_id) in token_map.items():
            decoded = tokenizer.decode([token_id])
            print(f"  {letter}: '{var}' -> token {token_id} ('{decoded}')")

    # Get log probs for each answer
    answer_logprobs = {}
    for letter, token_id in answer_tokens.items():
        answer_logprobs[letter] = float(log_probs[0, token_id].item())

    # Get predicted answer (highest logprob)
    predicted_answer = max(answer_logprobs, key=answer_logprobs.get)
    
    return predicted_answer, answer_logprobs


@torch.no_grad()
def evaluate_benchmark(
    questions: List[Dict[str, Any]],
    tokenizer,
    model,
    device: torch.device,
    over_zero_all: torch.Tensor,
    over_zero_correct: torch.Tensor,
    over_zero_incorrect: torch.Tensor,
    over_zero_ref: Dict[str, torch.Tensor],
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """
    Evaluate model on benchmark while tracking neuron activations.
    Returns detailed results, tokens for all, tokens for correct, tokens for incorrect.
    """
    model.eval()
    results = []
    tokens_all = 0
    tokens_correct = 0
    tokens_incorrect = 0
    correct = 0

    print(f"[4/9] Evaluating on benchmark ({len(questions)} questions)...")

    for i, question in enumerate(questions):
        # Format question
        formatted = format_question_choices(question)

        # Tokenize
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        num_tokens = int(attention_mask.sum().item())
        tokens_all += num_tokens

        # Check correctness first (without activations tracking)
        correct_answer = question["answer"]
        
        # Temporarily disable tracking
        over_zero_ref["tensor"] = None
        predicted_answer, answer_logprobs = get_predicted_answer(
            model, tokenizer, input_ids, attention_mask, debug=(i == 0)
        )
        
        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct += 1
            tokens_correct += num_tokens
        else:
            tokens_incorrect += num_tokens
        
        # Run with tracking for ALL questions
        over_zero_ref["tensor"] = over_zero_all
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Run with tracking for correct/incorrect
        if is_correct:
            over_zero_ref["tensor"] = over_zero_correct
        else:
            over_zero_ref["tensor"] = over_zero_incorrect
        
        # Run model again with tracking enabled
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

        # Store result
        result_entry = {
            "question_idx": i,
            "question": question["question"],
            "choices": question["choices"],
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "subject": question.get("subject", "unknown"),
        }
        
        # Only include logprobs if they're actually different (useful for debugging)
        unique_logprobs = len(set(answer_logprobs.values()))
        if unique_logprobs > 1:
            result_entry["answer_logprobs"] = answer_logprobs
        
        results.append(result_entry)

        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            accuracy = (correct / (i + 1)) * 100
            print(
                f"    Progress: {i+1}/{len(questions)} | Accuracy: {accuracy:.2f}% | Tokens: {tokens_all}"
            )

    return results, tokens_all, tokens_correct, tokens_incorrect


def layerwise_indices_from_mask(mask_tensor: torch.Tensor) -> List[List[int]]:
    """Convert mask tensor to layerwise indices."""
    out: List[List[int]] = []
    for li in range(mask_tensor.size(0)):
        idx = torch.nonzero(mask_tensor[li], as_tuple=False).squeeze(-1)
        out.append([int(i.item()) for i in idx])
    return out


def compute_overall_statistics(
    over_zero: torch.Tensor,
    total_tokens: int,
    num_layers: int,
    intermediate_size: int,
) -> Dict[str, Any]:
    """
    Compute overall neuron activation statistics (without categories).
    """
    print(f"[5/9] Computing overall neuron statistics...")
    
    # Activation probabilities
    activation_probs = over_zero.float() / max(1, total_tokens)
    
    # Active neurons
    active_mask = over_zero > 0
    num_active = int(active_mask.sum().item())
    
    # Layer-wise statistics
    layer_stats = []
    for li in range(num_layers):
        layer_active = int((over_zero[li] > 0).sum().item())
        layer_mean_prob = float(activation_probs[li].mean().item())
        layer_max_prob = float(activation_probs[li].max().item())
        
        layer_stats.append({
            "layer": li,
            "active_neurons": layer_active,
            "mean_activation_prob": layer_mean_prob,
            "max_activation_prob": layer_max_prob,
        })
    
    # Top neurons by activation probability
    flat_probs = activation_probs.view(-1)
    top_k = min(200, flat_probs.numel())
    top_vals, top_idx = torch.topk(flat_probs, k=top_k, largest=True, sorted=True)
    
    top_neurons = []
    for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
        layer = idx // intermediate_size
        neuron = idx % intermediate_size
        top_neurons.append({
            "layer": int(layer),
            "neuron": int(neuron),
            "activation_prob": float(val),
            "activation_count": int(over_zero.view(-1)[idx].item()),
        })
    
    print(f"  Total active neurons: {num_active:,} / {num_layers * intermediate_size:,}")
    
    return {
        "total_active_neurons": num_active,
        "total_neurons": num_layers * intermediate_size,
        "activation_rate": num_active / (num_layers * intermediate_size),
        "top_neurons_by_activation": top_neurons,
        "layer_statistics": layer_stats,
    }


def compute_lape_analysis(
    over_zero_1: torch.Tensor,
    over_zero_2: torch.Tensor,
    tokens_1: int,
    tokens_2: int,
    num_layers: int,
    intermediate_size: int,
    label_1: str = "correct",
    label_2: str = "incorrect",
) -> Dict[str, Any]:
    """
    Compute LAPE-style analysis to find category-specific neurons.
    
    Args:
        over_zero_1: Activation counts for category 1 (e.g., correct answers)
        over_zero_2: Activation counts for category 2 (e.g., incorrect answers)
        tokens_1: Total tokens for category 1
        tokens_2: Total tokens for category 2
        num_layers: Number of layers
        intermediate_size: Neurons per layer
        label_1: Name for category 1
        label_2: Name for category 2
    
    Returns:
        Dictionary with LAPE analysis results
    """
    print(f"[6/9] Running LAPE analysis ({label_1} vs {label_2})...")
    
    eps = 1e-12
    
    # Calculate activation probabilities
    p1 = over_zero_1.float() / max(1, tokens_1)
    p2 = over_zero_2.float() / max(1, tokens_2)
    
    # Stack and normalize (L1 normalization)
    probs = torch.stack([p1, p2], dim=-1)  # (L, I, 2)
    probs_sum = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    probs_norm = probs / probs_sum
    
    # Compute entropy (LAPE)
    entropy = -(probs_norm * (probs_norm.clamp_min(eps).log())).sum(dim=-1)  # (L, I)
    
    # Filter significant neurons (top FILTER_RATE by max probability)
    flat_probs = probs.view(-1)
    k_filter = max(1, int(round(len(flat_probs) * FILTER_RATE)))
    top_prob_value = torch.kthvalue(flat_probs, k_filter).values.item()
    max_over_cats = probs.max(dim=-1).values  # (L, I)
    entropy_filtered = entropy.clone()
    mask_keep = max_over_cats > top_prob_value
    entropy_filtered[~mask_keep] = float("inf")  # Exclude weak neurons
    
    # Select bottom TOP_RATE by entropy (lowest entropy = most specific)
    flat_entropy = entropy_filtered.view(-1)
    k_top = max(1, int(round(len(flat_entropy) * TOP_RATE)))
    top_vals, top_idx = torch.topk(-flat_entropy, k=k_top, largest=True)  # min entropy
    row_index = top_idx // entropy.size(1)  # layers
    col_index = top_idx % entropy.size(1)  # neurons
    selected_probs = probs[row_index, col_index]  # (k_top, 2)
    
    # Threshold for assigning category to neuron
    k_bar = max(1, int(round(len(flat_probs) * ACTIVATION_BAR_RATIO)))
    activation_bar = torch.kthvalue(flat_probs, k_bar).values.item()
    
    # Assign neurons to categories based on activation_bar threshold
    selected_probs_T = selected_probs.t()  # (2, k_top)
    cat_idx, neu_idx = torch.where(selected_probs_T > activation_bar)  # cat ∈ {0,1}
    
    # Collect indices by category and layer
    merged = torch.stack((row_index, col_index), dim=-1)  # (k_top, 2)
    by_category: List[List[torch.LongTensor]] = []
    for cat_id in [0, 1]:
        sel = merged[neu_idx[cat_idx == cat_id]]
        layer_lists = [[] for _ in range(num_layers)]
        for l, h in sel.tolist():
            layer_lists[l].append(h)
        layer_tensors = [torch.tensor(sorted(v), dtype=torch.long) for v in layer_lists]
        by_category.append(layer_tensors)
    
    # Basic statistics
    active1_any = over_zero_1 > 0
    active2_any = over_zero_2 > 0
    only1 = active1_any & (~active2_any)
    only2 = active2_any & (~active1_any)
    both = active1_any & active2_any
    
    # Top neurons by rate difference
    diff_rate = (p1 - p2).view(-1)
    flat_p1 = p1.view(-1)
    flat_p2 = p2.view(-1)
    top_k_diff = min(200, diff_rate.numel())
    top_vals_diff, top_idx_diff = torch.topk(diff_rate, k=top_k_diff, largest=True, sorted=True)
    top_list = []
    for val, idx in zip(top_vals_diff.tolist(), top_idx_diff.tolist()):
        layer = idx // intermediate_size
        neuron = idx % intermediate_size
        top_list.append({
            "layer": int(layer),
            "neuron": int(neuron),
            f"{label_1}_rate": float(flat_p1[idx].item()),
            f"{label_2}_rate": float(flat_p2[idx].item()),
            "rate_diff": float(val),
        })
    
    count_cat1 = sum(len(layer) for layer in by_category[0])
    count_cat2 = sum(len(layer) for layer in by_category[1])
    
    print(f"  Found {count_cat1} {label_1}-specific neurons")
    print(f"  Found {count_cat2} {label_2}-specific neurons")
    
    return {
        "tokens_processed": {
            label_1: int(tokens_1),
            label_2: int(tokens_2),
        },
        "neuron_totals": {
            "total_neurons": int(num_layers * intermediate_size),
            f"{label_1}_active_any": int(active1_any.sum().item()),
            f"{label_2}_active_any": int(active2_any.sum().item()),
            "both_active": int(both.sum().item()),
            f"only_{label_1}": int(only1.sum().item()),
            f"only_{label_2}": int(only2.sum().item()),
        },
        "thresholds": {
            "filter_rate": FILTER_RATE,
            "top_rate": TOP_RATE,
            "activation_bar_ratio": ACTIVATION_BAR_RATIO,
            "activation_bar_value": float(activation_bar),
            "top_prob_value": float(top_prob_value),
        },
        "layerwise_indices_any": {
            f"only_{label_1}": layerwise_indices_from_mask(only1),
            f"only_{label_2}": layerwise_indices_from_mask(only2),
            "both": layerwise_indices_from_mask(both),
        },
        "category_specific_masks": {
            label_1: [v.tolist() for v in by_category[0]],
            label_2: [v.tolist() for v in by_category[1]],
        },
        f"top_{label_1}_specific_by_rate_diff": top_list,
    }


def main():
    print("=" * 60)
    print("LAPE Neuron Analysis on TUMLU Benchmark")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDevice: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Benchmark: {BENCHMARK_PATH}")
    print()

    # Load benchmark
    questions = load_benchmark(BENCHMARK_PATH)

    # Load model and tokenizer
    print(f"[2/8] Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[3/8] Loading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32),
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    layers, intermediate_size = get_layers_and_intermediate(model)
    num_layers = len(layers)
    total_neurons = num_layers * intermediate_size

    print(f"  Model loaded: {num_layers} layers, {intermediate_size} neurons/layer")
    print(f"  Total neurons: {total_neurons}")

    # Initialize activation counters for all, correct and incorrect answers
    over_zero_all = torch.zeros(
        (num_layers, intermediate_size), dtype=torch.int64, device=device
    )
    over_zero_correct = torch.zeros(
        (num_layers, intermediate_size), dtype=torch.int64, device=device
    )
    over_zero_incorrect = torch.zeros(
        (num_layers, intermediate_size), dtype=torch.int64, device=device
    )

    # Patch model to track activations
    over_zero_ref = {"tensor": None}
    patched = make_patchers(layers, over_zero_ref)
    apply_patches(patched)

    # Run evaluation
    eval_results, tokens_all, tokens_correct, tokens_incorrect = evaluate_benchmark(
        questions, tokenizer, model, device, 
        over_zero_all, over_zero_correct, over_zero_incorrect, over_zero_ref
    )

    # Restore original forward
    restore_patches(patched)

    # Compute overall statistics (all questions together)
    overall_stats = compute_overall_statistics(
        over_zero_all,
        tokens_all,
        num_layers,
        intermediate_size,
    )

    # Compute LAPE analysis (correct vs incorrect)
    lape_stats = compute_lape_analysis(
        over_zero_correct,
        over_zero_incorrect,
        tokens_correct,
        tokens_incorrect,
        num_layers,
        intermediate_size,
        label_1="correct",
        label_2="incorrect",
    )

    # Calculate accuracy
    correct = sum(1 for r in eval_results if r["is_correct"])
    accuracy = (correct / len(questions)) * 100

    print("[7/9] Computing final statistics...")
    print(f"\n  Accuracy: {accuracy:.2f}% ({correct}/{len(questions)})")
    print(f"  Total tokens: {tokens_all}")
    print(f"  Tokens for correct answers: {tokens_correct}")
    print(f"  Tokens for incorrect answers: {tokens_incorrect}")
    
    print(f"  Overall active neurons: {overall_stats['total_active_neurons']:,}")
    correct_neurons = lape_stats["neuron_totals"]["correct_active_any"]
    incorrect_neurons = lape_stats["neuron_totals"]["incorrect_active_any"]
    print(f"  Neurons active on correct: {correct_neurons:,}")
    print(f"  Neurons active on incorrect: {incorrect_neurons:,}")

    # Prepare output
    output = {
        "model": MODEL_NAME,
        "benchmark": BENCHMARK_PATH,
        "device": str(device),
        "architecture": {
            "num_layers": num_layers,
            "intermediate_size": intermediate_size,
            "total_neurons": total_neurons,
        },
        "benchmark_results": {
            "total_questions": len(questions),
            "correct": correct,
            "incorrect": len(questions) - correct,
            "accuracy": accuracy,
            "tokens_all": tokens_all,
            "tokens_correct": tokens_correct,
            "tokens_incorrect": tokens_incorrect,
        },
        "overall_statistics": overall_stats,
        "lape_analysis": lape_stats,
        "detailed_results": eval_results,
    }

    # Save results
    print("[8/9] Saving results...")

    # Determine output filename based on benchmark
    benchmark_name = os.path.basename(BENCHMARK_PATH).replace(".jsonl", "")
    model_short = MODEL_NAME.split("/")[-1]
    output_dir = "activation_TUMLU_results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_json = f"{output_dir}/benchmark_lape_{model_short}_{benchmark_name}.json"
    output_pth = f"{output_dir}/benchmark_lape_{model_short}_{benchmark_name}.pth"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Save activation tensors
    torch.save(
        {
            "over_zero_all": over_zero_all.cpu(),
            "over_zero_correct": over_zero_correct.cpu(),
            "over_zero_incorrect": over_zero_incorrect.cpu(),
            "activation_probs_all": (over_zero_all.float() / max(1, tokens_all)).cpu(),
            "activation_probs_correct": (over_zero_correct.float() / max(1, tokens_correct)).cpu(),
            "activation_probs_incorrect": (over_zero_incorrect.float() / max(1, tokens_incorrect)).cpu(),
            "correct_specific_masks": [v for v in lape_stats["category_specific_masks"]["correct"]],
            "incorrect_specific_masks": [v for v in lape_stats["category_specific_masks"]["incorrect"]],
        },
        output_pth,
    )

    print("[9/9] Done!")
    print(f"\n{'=' * 60}")
    print("LAPE ANALYSIS COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Results saved to: {output_json}")
    print(f"Activations saved to: {output_pth}")
    print("\nSummary:")
    print(f"  - Accuracy: {accuracy:.2f}%")
    print(f"  - Overall active neurons: {overall_stats['total_active_neurons']:,}")
    print(f"  - Correct-specific neurons: {sum(len(l) for l in lape_stats['category_specific_masks']['correct']):,}")
    print(f"  - Incorrect-specific neurons: {sum(len(l) for l in lape_stats['category_specific_masks']['incorrect']):,}")
    print(f"  - Tokens processed: {tokens_all} (correct: {tokens_correct}, incorrect: {tokens_incorrect})")


if __name__ == "__main__":
    main()
