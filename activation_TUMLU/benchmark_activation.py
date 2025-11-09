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
BENCHMARK_PATH = "/Users/galyukshev/Desktop/LAPE/data/TUMLU/crimean-tatar/biology.jsonl"

# Additional settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
BATCH_SIZE = 1  # Process one question at a time for accurate tracking


# ============ HELPER FUNCTIONS ============


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    """Load TUMLU benchmark questions from JSONL file."""
    print(f"[1/7] Loading benchmark from {path}...")
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


def get_answer_logprobs(
    model, tokenizer, input_ids, attention_mask
) -> Dict[str, float]:
    """Get log probabilities for each answer choice (A, B, C, D)."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]  # Last token logits
        log_probs = torch.log_softmax(logits, dim=-1)

    # Get tokens for A, B, C, D
    answer_tokens = {}
    for letter in ["A", "B", "C", "D"]:
        tokens = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if tokens:
            answer_tokens[letter] = tokens[0]

    # Get log probs for each answer
    answer_logprobs = {}
    for letter, token_id in answer_tokens.items():
        answer_logprobs[letter] = float(log_probs[0, token_id].item())

    return answer_logprobs


@torch.no_grad()
def evaluate_benchmark(
    questions: List[Dict[str, Any]],
    tokenizer,
    model,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Evaluate model on benchmark while tracking neuron activations.
    Returns detailed results and total token count.
    """
    model.eval()
    results = []
    total_tokens = 0
    correct = 0

    print(f"[4/7] Evaluating on benchmark ({len(questions)} questions)...")

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

        # Get answer probabilities
        answer_logprobs = get_answer_logprobs(
            model, tokenizer, input_ids, attention_mask
        )
        predicted_answer = max(answer_logprobs, key=answer_logprobs.get)

        # Check correctness
        correct_answer = question["answer"]
        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct += 1

        # Track tokens
        total_tokens += int(attention_mask.sum().item())

        # Store result
        results.append(
            {
                "question_idx": i,
                "question": question["question"],
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "answer_logprobs": answer_logprobs,
                "subject": question.get("subject", "unknown"),
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            accuracy = (correct / (i + 1)) * 100
            print(
                f"    Progress: {i+1}/{len(questions)} | Accuracy: {accuracy:.2f}% | Tokens: {total_tokens}"
            )

    return results, total_tokens


def compute_neuron_statistics(
    over_zero: torch.Tensor,
    total_tokens: int,
    num_layers: int,
    intermediate_size: int,
) -> Dict[str, Any]:
    """Compute neuron activation statistics."""
    print("[5/7] Computing neuron statistics...")

    activation_probs = over_zero.float() / max(1, total_tokens)

    # Active neurons (activated at least once)
    active_mask = over_zero > 0
    num_active = int(active_mask.sum().item())

    # Top neurons by activation probability
    flat_probs = activation_probs.view(-1)
    top_k = flat_probs.numel()
    top_vals, top_idx = torch.topk(flat_probs, k=top_k, largest=True, sorted=True)

    top_neurons = []
    for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
        layer = idx // intermediate_size
        neuron = idx % intermediate_size
        top_neurons.append(
            {
                "layer": int(layer),
                "neuron": int(neuron),
                "activation_prob": float(val),
                "activation_count": int(over_zero.view(-1)[idx].item()),
            }
        )

    # Layer-wise statistics
    layer_stats = []
    for li in range(num_layers):
        layer_probs = activation_probs[li]
        layer_active = int((over_zero[li] > 0).sum().item())
        layer_mean_prob = float(layer_probs.mean().item())
        layer_max_prob = float(layer_probs.max().item())

        layer_stats.append(
            {
                "layer": li,
                "active_neurons": layer_active,
                "mean_activation_prob": layer_mean_prob,
                "max_activation_prob": layer_max_prob,
            }
        )

    return {
        "total_active_neurons": num_active,
        "total_neurons": num_layers * intermediate_size,
        "activation_rate": num_active / (num_layers * intermediate_size),
        "top_neurons": top_neurons,
        "layer_statistics": layer_stats,
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
    print(f"[2/7] Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[3/7] Loading model {MODEL_NAME}...")
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

    # Initialize activation counter
    over_zero = torch.zeros(
        (num_layers, intermediate_size), dtype=torch.int64, device=device
    )

    # Patch model to track activations
    over_zero_ref = {"tensor": over_zero}
    patched = make_patchers(layers, over_zero_ref)
    apply_patches(patched)

    # Run evaluation
    eval_results, total_tokens = evaluate_benchmark(questions, tokenizer, model, device)

    # Restore original forward
    restore_patches(patched)

    # Compute statistics
    neuron_stats = compute_neuron_statistics(
        over_zero, total_tokens, num_layers, intermediate_size
    )

    # Calculate accuracy
    correct = sum(1 for r in eval_results if r["is_correct"])
    accuracy = (correct / len(questions)) * 100

    print("[6/7] Computing final statistics...")
    print(f"\n  Accuracy: {accuracy:.2f}% ({correct}/{len(questions)})")
    print(f"  Total tokens processed: {total_tokens}")
    print(
        f"  Active neurons: {neuron_stats['total_active_neurons']}/{neuron_stats['total_neurons']}"
    )
    print(f"  Activation rate: {neuron_stats['activation_rate']*100:.2f}%")

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
            "accuracy": accuracy,
            "total_tokens": total_tokens,
        },
        "neuron_statistics": neuron_stats,
        "detailed_results": eval_results,
    }

    # Save results
    print("[7/7] Saving results...")

    # Determine output filename based on benchmark
    benchmark_name = os.path.basename(BENCHMARK_PATH).replace(".jsonl", "")
    model_short = MODEL_NAME.split("/")[-1]
    output_json = f"benchmark_activation_{model_short}_{benchmark_name}.json"
    output_pth = f"benchmark_activation_{model_short}_{benchmark_name}.pth"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Save activation tensor
    torch.save(
        {
            "over_zero": over_zero.cpu(),
            "activation_probs": (over_zero.float() / max(1, total_tokens)).cpu(),
        },
        output_pth,
    )

    print(f"\n{'=' * 60}")
    print("DONE!")
    print(f"{'=' * 60}")
    print(f"Results saved to: {output_json}")
    print(f"Activations saved to: {output_pth}")
    print("\nSummary:")
    print(f"  - Accuracy: {accuracy:.2f}%")
    print(
        f"  - Active neurons: {neuron_stats['total_active_neurons']} ({neuron_stats['activation_rate']*100:.2f}%)"
    )
    print(f"  - Tokens processed: {total_tokens}")


if __name__ == "__main__":
    main()
