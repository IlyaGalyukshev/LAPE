import warnings

warnings.filterwarnings("ignore")

import os
import json
import csv
import torch
from datetime import datetime
from typing import Dict, Any, List

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()


# -----------------------------
# CONFIG
# -----------------------------
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3-12b-it")
MODEL_PATH = "/hf_models"

DATA_ROOT = "/work/benchmarks/TUMLU"
OUTPUT_BASE = "/work/benchmarks/uncertainty_metrics"

LANGS = [
    "azerbaijani",
    "crimean-tatar",
    "crimean-tatar-cyrillic",
    "en",
    "karakalpak",
    "kazakh",
    "kazakh-latin",
    "ru",
    "tatar",
    "turkish",
    "uyghur",
    "uyghur-latin",
    "uzbek",
    "uzbek-cyrillic",
]

CHUNK_SIZE = 128

TOP_RATE = 0.01
FILTER_RATE = 0.95
ACTIVATION_BAR_RATIO = 0.95

PROMPTS = {
    "azerbaijani": """Sual: {question}\n{choices}\n\nCavabı bir cümlə ilə yazın.\nCavab: """,
    "crimean-tatar": """Sual: {question}\n{choices}\n\nCevapnı bir cümle ile yazıñ.\nCevap: """,
    "crimean-tatar-cyrillic": """Суал: {question}\n{choices}\n\nДжевапны бир джумле иле языныз.\nДжевап: """,
    "en": """Question: {question}\n{choices}\n\nAnswer in one sentence.\nAnswer: """,
    "karakalpak": """Soraw: {question}\n{choices}\n\nJuwaptı bir sózlem menen jazıń.\nJuwap: """,
    "kazakh": """Сұрақ: {question}\n{choices}\n\nЖауапты бір сөйлеммен жазыңыз.\nЖауап: """,
    "kazakh-latin": """Suraq: {question}\n{choices}\n\nJawapty bir sóılemmen jazyńyz.\nJawap: """,
    "ru": """Вопрос: {question}\n{choices}\n\nОтветьте одним предложением.\nОтвет: """,
    "tatar": """Сорау: {question}\n{choices}\n\nҖавапны бер җөмлә белән языгыз.\nҖавап: """,
    "turkish": """Soru: {question}\n{choices}\n\nCevabı tek bir cümleyle yazın.\nCevap: """,
    "uyghur": """سوئال: {question}\n{choices}\n\nجاۋابنى بىر جۈملە بىلەن يېزىڭ.\nجاۋاب: """,
    "uyghur-latin": """Soal: {question}\n{choices}\n\nJawabni bir jumle bilen yezing.\nJawab: """,
    "uzbek": """Savol: {question}\n{choices}\n\nJavobni bitta gap bilan yozing.\nJavob: """,
    "uzbek-cyrillic": """Савол: {question}\n{choices}\n\nЖавобни битта гап билан ёзинг.\nЖавоб: """,
}


# -----------------------------
# Helpers
# -----------------------------
def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def safe_model_id(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_").replace(" ", "_")


def format_choices(choices: list) -> str:
    labels = ["A", "B", "C", "D"]
    out = []
    for i, c in enumerate(choices):
        if i < len(labels):
            out.append(f"{labels[i]}. {c}")
    return "\n".join(out)


def make_user_prompt(lang: str, obj: Dict[str, Any]) -> str:
    formatted_choices = format_choices(obj["choices"])
    prompt_template = PROMPTS[lang]
    return prompt_template.format(question=obj["question"], choices=formatted_choices)


# -----------------------------
# LAPE
# -----------------------------
def main() -> None:
    torch.set_grad_enabled(False)

    safe_name = safe_model_id(MODEL_ID)
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, safe_name, "lape")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_progress("=" * 80)
    log_progress(f"Starting LAPE analysis for model: {MODEL_ID}")
    log_progress(f"MODEL_PATH: {MODEL_PATH}")
    log_progress(f"DATA_ROOT: {DATA_ROOT}")
    log_progress(f"OUTPUT_DIR: {OUTPUT_DIR}")
    log_progress(f"Number of languages: {len(LANGS)}")
    log_progress("=" * 80)

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    log_progress(f"Using device: {device}")

    if has_cuda:
        torch.cuda.empty_cache()
        try:
            props = torch.cuda.get_device_properties(0)
            log_progress(f"GPU[0] total memory: {props.total_memory / 1e9:.2f} GB")
        except Exception:
            pass

    log_progress("Loading tokenizer (local_files_only=True)...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log_progress("Loading model (local_files_only=True)...")
    if has_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            local_files_only=True,
        )
        input_device = model.get_input_embeddings().weight.device
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=None,
            local_files_only=True,
        )
        model.to(device)
        input_device = device

    model.eval()
    log_progress("Model loaded and ready!")

    # DEBUG: print model structure to determine correct layer access path
    print(model)
    print(f"\nmodel_type = {model.config.model_type}")
    print(f"type(model) = {type(model)}")
    if hasattr(model, "model"):
        print(f"type(model.model) = {type(model.model)}")
        if hasattr(model.model, "language_model"):
            print(
                f"type(model.model.language_model) = {type(model.model.language_model)}"
            )
            print(
                f"dir(model.model.language_model) = {[a for a in dir(model.model.language_model) if not a.startswith('_')]}"
            )

    # -----------------------------
    # Pass 1: count tokens per language and find minimum
    # -----------------------------
    lang_token_counts: Dict[str, int] = {}
    min_tokens = None

    log_progress("\nLoading and counting tokens for all languages (pass 1)...")
    for idx, lang in enumerate(LANGS, 1):
        path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")
        log_progress(f"[{idx}/{len(LANGS)}] Scanning {lang}: {path}")

        total = 0
        n_questions = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = make_user_prompt(lang, obj)
                ids = tokenizer(text, add_special_tokens=True)["input_ids"]
                total += len(ids)
                n_questions += 1

        lang_token_counts[lang] = total
        log_progress(f"  - Loaded {n_questions} questions")
        log_progress(f"  - Tokenized to {total} tokens")

        if min_tokens is None or total < min_tokens:
            min_tokens = total

    common_tokens = int(min_tokens or 0)
    log_progress(f"\nMinimum tokens across all languages: {common_tokens}")

    # -----------------------------
    # Model architecture mapping
    # -----------------------------
    model_type = model.config.model_type

    if model_type in ("llama", "mistral", "qwen2", "gemma2", "gemma3_text"):
        layers = model.model.layers
    elif model_type == "gemma3":
        layers = model.model.language_model.layers
    elif model_type == "gpt2":
        layers = model.transformer.h
    elif model_type == "bloom":
        layers = model.transformer.h
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    num_layers = len(layers)
    log_progress(f"\nModel architecture: {model_type}")
    log_progress(f"Number of layers: {num_layers}")

    if model_type in ("llama", "mistral", "qwen2", "gemma2", "gemma3", "gemma3_text"):
        sample_mlp = layers[0].mlp
        intermediate_size = sample_mlp.gate_proj.out_features

    elif model_type == "gpt2":
        sample_mlp = layers[0].mlp
        intermediate_size = sample_mlp.c_fc.weight.shape[1]
    elif model_type == "bloom":
        sample_mlp = layers[0].mlp
        intermediate_size = sample_mlp.dense_h_to_4h.out_features
    else:
        raise ValueError(
            f"Unsupported model_type (for intermediate_size): {model_type}"
        )

    log_progress(f"Intermediate size (neurons per layer): {intermediate_size}")
    log_progress(f"Total neurons in model: {num_layers * intermediate_size:,}")

    over_zero = torch.zeros(
        num_layers, intermediate_size, len(LANGS), dtype=torch.long, device="cpu"
    )
    token_counts = torch.zeros(len(LANGS), dtype=torch.long, device="cpu")
    current_lang_index = 0

    log_progress("\nRegistering forward hooks...")
    if model_type in ("llama", "mistral", "qwen2", "gemma2", "gemma3", "gemma3_text"):

        def make_gate_hook(layer_idx, act_fn):

            def hook(module, input, output):
                nonlocal current_lang_index
                activation = act_fn(output.to(torch.float32))
                active = (activation > 0).sum(dim=(0, 1))  # (intermediate_size,)
                over_zero[layer_idx, :, current_lang_index] += active.to(
                    dtype=torch.long
                ).cpu()

            return hook

        for layer_idx, layer in enumerate(layers):
            mlp = layer.mlp
            mlp.gate_proj.register_forward_hook(make_gate_hook(layer_idx, mlp.act_fn))

    elif model_type == "gpt2":

        def make_fc_hook(layer_idx, act_fn):
            def hook(module, input, output):
                nonlocal current_lang_index
                activation = act_fn(output.to(torch.float32))
                active = (activation > 0).sum(dim=(0, 1))
                over_zero[layer_idx, :, current_lang_index] += active.to(
                    dtype=torch.long
                ).cpu()

            return hook

        for layer_idx, layer in enumerate(layers):
            mlp = layer.mlp
            mlp.c_fc.register_forward_hook(make_fc_hook(layer_idx, mlp.act))

    elif model_type == "bloom":

        def make_gelu_hook(layer_idx):
            def hook(module, input, output):
                nonlocal current_lang_index
                activation = output.to(torch.float32)
                active = (activation > 0).sum(dim=(0, 1))
                over_zero[layer_idx, :, current_lang_index] += active.to(
                    dtype=torch.long
                ).cpu()

            return hook

        for layer_idx, layer in enumerate(layers):
            mlp = layer.mlp
            mlp.gelu_impl.register_forward_hook(make_gelu_hook(layer_idx))

    log_progress(f"Hooks registered for all {num_layers} layers")

    # -----------------------------
    # Pass 2: stream tokens
    # -----------------------------
    log_progress("\n" + "=" * 80)
    log_progress("Collecting neuron activations (pass 2)...")
    log_progress("=" * 80)

    checkpoint_path = os.path.join(OUTPUT_DIR, "lape_checkpoint.pt")

    completed_lang_indices: List[int] = []
    if os.path.exists(checkpoint_path):
        log_progress(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, weights_only=False)
        over_zero.copy_(ckpt["over_zero"])
        token_counts.copy_(ckpt["token_counts"])
        completed_lang_indices = ckpt["completed_lang_indices"]
        log_progress(
            f"Resumed: {len(completed_lang_indices)} languages already completed"
        )

    with torch.no_grad():
        for lang_idx, lang in enumerate(LANGS):
            if lang_idx in completed_lang_indices:
                log_progress(
                    f"\n[{lang_idx+1}/{len(LANGS)}] Skipping {lang} (already completed)"
                )
                continue

            current_lang_index = lang_idx
            path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")

            tokens_used = 0
            buffer: List[int] = []

            total_chunks = (common_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
            log_progress(
                f"\n[{lang_idx+1}/{len(LANGS)}] Processing {lang} ({common_tokens} tokens, {total_chunks} chunks)..."
            )

            def flush_full_chunks():
                nonlocal tokens_used, buffer
                chunk_num_local = 0
                while len(buffer) >= CHUNK_SIZE and tokens_used < common_tokens:
                    take = CHUNK_SIZE
                    chunk = buffer[:take]
                    buffer = buffer[take:]

                    input_ids = (
                        torch.tensor(chunk, dtype=torch.long)
                        .unsqueeze(0)
                        .to(input_device)
                    )
                    _ = model(input_ids=input_ids, use_cache=False)

                    tokens_used += take
                    chunk_num_local += 1

                    if (
                        tokens_used // CHUNK_SIZE
                    ) % 10 == 0 or tokens_used >= common_tokens:
                        done_chunks = min(tokens_used // CHUNK_SIZE, total_chunks)
                        progress_pct = (done_chunks / total_chunks) * 100.0
                        log_progress(
                            f"  - Progress: ~{done_chunks}/{total_chunks} chunks ({progress_pct:.1f}%)"
                        )

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if tokens_used >= common_tokens:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = make_user_prompt(lang, obj)
                    ids = tokenizer(text, add_special_tokens=True)["input_ids"]

                    remaining = common_tokens - (tokens_used + len(buffer))
                    if remaining <= 0:
                        break

                    if len(ids) > remaining:
                        ids = ids[:remaining]

                    buffer.extend(ids)
                    flush_full_chunks()

            # flush remainder (<= CHUNK_SIZE) if any tokens still needed
            if tokens_used < common_tokens and len(buffer) > 0:
                remaining = common_tokens - tokens_used
                if remaining > 0:
                    chunk = buffer[:remaining]
                    input_ids = (
                        torch.tensor(chunk, dtype=torch.long)
                        .unsqueeze(0)
                        .to(input_device)
                    )
                    _ = model(input_ids=input_ids, use_cache=False)
                    tokens_used += len(chunk)
                buffer = []

            token_counts[lang_idx] = tokens_used

            if tokens_used != common_tokens:
                log_progress(
                    f"WARNING: {lang} used_tokens={tokens_used} != common_tokens={common_tokens}"
                )

            if has_cuda:
                torch.cuda.empty_cache()

            # Save checkpoint after each language
            completed_lang_indices.append(lang_idx)
            torch.save(
                {
                    "over_zero": over_zero.clone(),
                    "token_counts": token_counts.clone(),
                    "completed_lang_indices": completed_lang_indices,
                },
                checkpoint_path,
            )
            log_progress(
                f"Checkpoint saved ({len(completed_lang_indices)}/{len(LANGS)} languages)"
            )

    log_progress("\nActivation collection completed!")
    if has_cuda:
        torch.cuda.empty_cache()

    lang_num = len(LANGS)
    n = token_counts.to(torch.float32)  # CPU

    log_progress("\n" + "=" * 80)
    log_progress("Computing activation probabilities and entropy...")
    log_progress("=" * 80)

    activation_probs = over_zero.to(torch.float32) / n.view(1, 1, lang_num)
    log_progress("Computed raw activation probabilities")

    normed_activation_probs = activation_probs / activation_probs.sum(
        dim=-1, keepdim=True
    )
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0.0
    log_progress("Normalized activation probabilities")

    log_probs = torch.where(
        normed_activation_probs > 0,
        normed_activation_probs.log(),
        torch.zeros_like(normed_activation_probs),
    )
    entropy = -(normed_activation_probs * log_probs).sum(dim=-1)
    log_progress("Computed entropy for all neurons")

    if torch.isnan(entropy).any():
        raise RuntimeError("NaN values in entropy")

    log_progress("\n" + "=" * 80)
    log_progress("Identifying language-specific neurons...")
    log_progress("=" * 80)
    log_progress(
        f"TOP_RATE: {TOP_RATE}, FILTER_RATE: {FILTER_RATE}, ACTIVATION_BAR_RATIO: {ACTIVATION_BAR_RATIO}"
    )

    flattened_probs = activation_probs.flatten()
    k_filter = int(round(len(flattened_probs) * FILTER_RATE))
    if k_filter < 1:
        k_filter = 1
    top_prob_value = flattened_probs.kthvalue(k_filter).values.item()
    log_progress(
        f"Filtering threshold (top {FILTER_RATE*100}% activations): {top_prob_value:.6f}"
    )

    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy = entropy.clone()
    entropy[top_position == 0] = float("inf")

    flattened_entropy = entropy.flatten()
    k_top = int(round(len(flattened_entropy) * TOP_RATE))
    if k_top < 1:
        k_top = 1

    log_progress(
        f"Selecting top {k_top:,} neurons with lowest entropy (top {TOP_RATE*100}%)..."
    )
    _, top_index = flattened_entropy.topk(k_top, largest=False)
    log_progress(f"Selected {len(top_index):,} candidate neurons")

    row_index = top_index // intermediate_size
    col_index = top_index % intermediate_size

    selected_probs = activation_probs[row_index, col_index]  # (k_top, lang_num)
    selected_probs_t = selected_probs.transpose(0, 1)  # (lang_num, k_top)

    k_bar = int(round(len(flattened_probs) * ACTIVATION_BAR_RATIO))
    if k_bar < 1:
        k_bar = 1
    activation_bar = flattened_probs.kthvalue(k_bar).values.item()

    lang_index_tensor, selected_idx_in_selected = torch.where(
        selected_probs_t > activation_bar
    )
    log_progress(f"Applying activation bar threshold: {activation_bar:.6f}")
    log_progress(
        f"Found {len(lang_index_tensor):,} language-specific neuron assignments"
    )

    log_progress("\nBuilding language-specific neuron masks...")
    final_lang_neurons = []

    for lang_id in range(lang_num):
        mask = lang_index_tensor == lang_id
        idx_for_lang = selected_idx_in_selected[mask]
        layer_to_neurons = [[] for _ in range(num_layers)]
        for pos in idx_for_lang.tolist():
            layer = int(row_index[pos])
            neuron = int(col_index[pos])
            layer_to_neurons[layer].append(neuron)

        lang_layer_tensors = []
        for layer_neurons in layer_to_neurons:
            if len(layer_neurons) == 0:
                lang_layer_tensors.append(torch.empty(0, dtype=torch.long))
            else:
                layer_neurons_sorted = sorted(set(layer_neurons))
                lang_layer_tensors.append(
                    torch.tensor(layer_neurons_sorted, dtype=torch.long)
                )
        final_lang_neurons.append(lang_layer_tensors)

    print("\n" + "=" * 80)
    print("Language-specific neurons statistics:")
    print("=" * 80)

    total_neurons = num_layers * intermediate_size
    for lang_id, lang in enumerate(LANGS):
        neuron_count = sum(
            len(layer_neurons) for layer_neurons in final_lang_neurons[lang_id]
        )
        percentage = (neuron_count / total_neurons) * 100
        print(f"{lang:25s}: {neuron_count:6d} neurons ({percentage:6.3f}%)")

    all_neurons_set = set()
    for lang_neurons in final_lang_neurons:
        for layer_idx, layer_neurons in enumerate(lang_neurons):
            for neuron in layer_neurons.tolist():
                all_neurons_set.add((layer_idx, neuron))
    total_unique_neurons = len(all_neurons_set)
    unique_percentage = (total_unique_neurons / total_neurons) * 100

    print("-" * 80)
    print(
        f"{'Total unique neurons':25s}: {total_unique_neurons:6d} neurons ({unique_percentage:6.3f}%)"
    )
    print(f"{'Total neurons in model':25s}: {total_neurons:6d} neurons")
    print("=" * 80 + "\n")

    param_count = 0
    for p in model.parameters():
        param_count += p.numel()

    model_info = {
        "model_name": MODEL_ID,
        "model_path": MODEL_PATH,
        "model_type": model_type,
        "num_layers": int(num_layers),
        "hidden_size": int(
            model.config.text_config.hidden_size
            if model_type == "gemma3"
            else model.config.hidden_size
        ),
        "intermediate_size": int(intermediate_size),
        "vocab_size": int(
            model.config.text_config.vocab_size
            if model_type == "gemma3"
            else model.config.vocab_size
        ),
        "num_parameters": int(param_count),
    }

    activation_data = {
        "model": model_info,
        "langs": LANGS,
        "tokens_per_lang": token_counts.clone(),
        "over_zero": over_zero.clone(),
        "activation_probs": activation_probs.clone(),
        "entropy": entropy.clone(),
        "top_rate": TOP_RATE,
        "filter_rate": FILTER_RATE,
        "activation_bar_ratio": ACTIVATION_BAR_RATIO,
    }

    mask_data = {
        "model": model_info,
        "langs": LANGS,
        "tokens_per_lang": token_counts.clone(),
        "lang_neurons": final_lang_neurons,
        "top_rate": TOP_RATE,
        "filter_rate": FILTER_RATE,
        "activation_bar_ratio": ACTIVATION_BAR_RATIO,
        "top_prob_value": float(top_prob_value),
        "activation_bar": float(activation_bar),
    }

    log_progress("\n" + "=" * 80)
    log_progress("Saving results...")
    log_progress("=" * 80)

    stats_path = os.path.join(OUTPUT_DIR, "activation_stats.pt")
    mask_path = os.path.join(OUTPUT_DIR, "lang_specific_neurons.pt")

    log_progress(f"Saving activation statistics to: {stats_path}")
    torch.save(activation_data, stats_path)

    log_progress(f"Saving language-specific neurons to: {mask_path}")
    torch.save(mask_data, mask_path)

    # Save TSV summary
    summary_path = os.path.join(OUTPUT_DIR, "lape_summary.tsv")
    log_progress(f"Saving summary to: {summary_path}")

    with open(summary_path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(
            [
                "language",
                "tokens_used",
                "lang_specific_neurons",
                "lang_specific_neurons_pct",
            ]
        )
        for lang_id, lang in enumerate(LANGS):
            neuron_count = sum(
                len(layer_neurons) for layer_neurons in final_lang_neurons[lang_id]
            )
            percentage = (neuron_count / total_neurons) * 100
            writer.writerow(
                [
                    lang,
                    int(token_counts[lang_id].item()),
                    neuron_count,
                    f"{percentage:.3f}",
                ]
            )

    # Remove checkpoint after successful save
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        log_progress("Checkpoint removed (run completed successfully)")

    log_progress("\n" + "=" * 80)
    log_progress("LAPE analysis completed successfully!")
    log_progress("=" * 80)
    print("\nActivation stats saved to:", stats_path)
    print("Language-specific neurons saved to:", mask_path)
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()
