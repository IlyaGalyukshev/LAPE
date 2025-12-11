import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

def log_progress(message):
    """Print progress message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

MODEL = "meta-llama/Meta-Llama-3.1-8B"
MODEL = "Tweeties/tweety-tatar-base-7b-2024-v1"
# MODEL = 'mistralai/Mistral-7B-Instruct-v0.2'
MODEL = "ai-forever/mGPT-1.3B-tatar"
# MODEL = 'ai-forever/mGPT'
MODEL = "Qwen/Qwen2.5-7B-Instruct"
# MODEL = 'google/gemma-2-9b'
# MODEL = 'bigscience/bloomz-7b1-mt'
# MODEL = 'bigscience/bloomz-7b1'
DATA_ROOT = "data/TUMLU"

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

CHUNK_SIZE = 128  # Reduced from 512 to save GPU memory
OUTPUT_DIR = "activation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

log_progress("="*80)
log_progress(f"Starting LAPE analysis for model: {MODEL}")
log_progress(f"Number of languages: {len(LANGS)}")
log_progress("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_progress(f"Using device: {device}")

# Clear GPU cache before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    log_progress(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

log_progress("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

log_progress("Loading model...")
# Load model directly to GPU to avoid OOM when moving from CPU to GPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL, 
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
log_progress("Moving model to device...")
model.to(device)
model.eval()
torch.set_grad_enabled(False)
log_progress("Model loaded and ready!")

lang_tokens = {}
min_tokens = None

log_progress("\nLoading and tokenizing data for all languages...")
for idx, lang in enumerate(LANGS, 1):
    log_progress(f"[{idx}/{len(LANGS)}] Processing {lang}...")
    path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["question"])
    log_progress(f"  - Loaded {len(texts)} questions")
    ids_list = []
    for text in texts:
        enc = tokenizer(text, add_special_tokens=True, return_tensors="pt")
        ids_list.append(enc["input_ids"][0])
    ids = torch.cat(ids_list, dim=0)
    lang_tokens[lang] = ids
    length = ids.size(0)
    log_progress(f"  - Tokenized to {length} tokens")
    if min_tokens is None or length < min_tokens:
        min_tokens = length

log_progress(f"\nMinimum tokens across all languages: {min_tokens}")

model_type = model.config.model_type

if model_type in ("llama", "mistral", "qwen2", "gemma2"):
    layers = model.model.layers
elif model_type == "gpt2":
    layers = model.transformer.h
elif model_type == "bloom":
    layers = model.transformer.h
else:
    raise ValueError(f"Unsupported model_type: {model_type}")

num_layers = len(layers)
log_progress(f"\nModel architecture: {model_type}")
log_progress(f"Number of layers: {num_layers}")

if model_type in ("llama", "mistral", "qwen2", "gemma2"):
    sample_mlp = layers[0].mlp
    intermediate_size = sample_mlp.gate_proj.out_features
elif model_type == "gpt2":
    sample_mlp = layers[0].mlp
    intermediate_size = sample_mlp.c_fc.weight.shape[1]
elif model_type == "bloom":
    sample_mlp = layers[0].mlp
    intermediate_size = sample_mlp.dense_h_to_4h.out_features

log_progress(f"Intermediate size (neurons per layer): {intermediate_size}")
log_progress(f"Total neurons in model: {num_layers * intermediate_size:,}")

over_zero = torch.zeros(num_layers, intermediate_size, len(LANGS), dtype=torch.long, device=device)
token_counts = torch.zeros(len(LANGS), dtype=torch.long)
current_lang_index = 0

log_progress("\nRegistering forward hooks...")
if model_type in ("llama", "mistral", "qwen2", "gemma2"):
    def make_gate_hook(layer_idx, act_fn):
        def hook(module, input, output):
            activation = act_fn(output.to(torch.float32))
            active = (activation > 0).sum(dim=(0, 1))
            over_zero[layer_idx, :, current_lang_index] += active.to(over_zero.dtype)
        return hook

    for layer_idx, layer in enumerate(layers):
        mlp = layer.mlp
        mlp.gate_proj.register_forward_hook(make_gate_hook(layer_idx, mlp.act_fn))

elif model_type == "gpt2":
    def make_fc_hook(layer_idx, act_fn):
        def hook(module, input, output):
            activation = act_fn(output.to(torch.float32))
            active = (activation > 0).sum(dim=(0, 1))
            over_zero[layer_idx, :, current_lang_index] += active.to(over_zero.dtype)
        return hook

    for layer_idx, layer in enumerate(layers):
        mlp = layer.mlp
        mlp.c_fc.register_forward_hook(make_fc_hook(layer_idx, mlp.act))

elif model_type == "bloom":
    def make_gelu_hook(layer_idx):
        def hook(module, input, output):
            activation = output.to(torch.float32)
            active = (activation > 0).sum(dim=(0, 1))
            over_zero[layer_idx, :, current_lang_index] += active.to(over_zero.dtype)
        return hook

    for layer_idx, layer in enumerate(layers):
        mlp = layer.mlp
        mlp.gelu_impl.register_forward_hook(make_gelu_hook(layer_idx))

common_tokens = min_tokens
log_progress(f"Hooks registered for all {num_layers} layers")

log_progress("\n" + "="*80)
log_progress("Collecting neuron activations...")
log_progress("="*80)

with torch.no_grad():
    for lang_idx, lang in enumerate(LANGS):
        current_lang_index = lang_idx
        ids = lang_tokens[lang][:common_tokens]
        token_counts[lang_idx] = ids.size(0)
        pos = 0
        total_chunks = (ids.size(0) + CHUNK_SIZE - 1) // CHUNK_SIZE
        log_progress(f"\n[{lang_idx+1}/{len(LANGS)}] Processing {lang} ({ids.size(0)} tokens, {total_chunks} chunks)...")
        chunk_num = 0
        while pos < ids.size(0):
            chunk = ids[pos:pos + CHUNK_SIZE]
            input_ids = chunk.unsqueeze(0).to(device)
            _ = model(input_ids=input_ids, use_cache=False)
            pos += CHUNK_SIZE
            chunk_num += 1
            if chunk_num % 10 == 0 or chunk_num == total_chunks:
                progress_pct = (chunk_num / total_chunks) * 100
                log_progress(f"  - Progress: {chunk_num}/{total_chunks} chunks ({progress_pct:.1f}%)")
        
        # Clear GPU cache after each language to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

log_progress("\nActivation collection completed!")
del lang_tokens
if torch.cuda.is_available():
    torch.cuda.empty_cache()

lang_num = len(LANGS)
n = token_counts.to(torch.float32).to(device)

log_progress("\n" + "="*80)
log_progress("Computing activation probabilities and entropy...")
log_progress("="*80)

activation_probs = over_zero.to(torch.float32) / n.view(1, 1, lang_num)
log_progress("Computed raw activation probabilities")

normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
normed_activation_probs[torch.isnan(normed_activation_probs)] = 0.0
log_progress("Normalized activation probabilities")

log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), torch.zeros_like(normed_activation_probs))
entropy = -(normed_activation_probs * log_probs).sum(dim=-1)
log_progress("Computed entropy for all neurons")

if torch.isnan(entropy).any():
    raise RuntimeError("NaN values in entropy")

TOP_RATE = 0.01
FILTER_RATE = 0.95
ACTIVATION_BAR_RATIO = 0.95

log_progress("\n" + "="*80)
log_progress("Identifying language-specific neurons...")
log_progress("="*80)
log_progress(f"TOP_RATE: {TOP_RATE}, FILTER_RATE: {FILTER_RATE}, ACTIVATION_BAR_RATIO: {ACTIVATION_BAR_RATIO}")

flattened_probs = activation_probs.flatten()
k_filter = int(round(len(flattened_probs) * FILTER_RATE))
if k_filter < 1:
    k_filter = 1
top_prob_value = flattened_probs.kthvalue(k_filter).values.item()
log_progress(f"Filtering threshold (top {FILTER_RATE*100}% activations): {top_prob_value:.6f}")

top_position = (activation_probs > top_prob_value).sum(dim=-1)
entropy = entropy.clone()
entropy[top_position == 0] = float("inf")

flattened_entropy = entropy.flatten()
k_top = int(round(len(flattened_entropy) * TOP_RATE))
if k_top < 1:
    k_top = 1

log_progress(f"Selecting top {k_top:,} neurons with lowest entropy (top {TOP_RATE*100}%)...")
_, top_index = flattened_entropy.topk(k_top, largest=False)
log_progress(f"Selected {len(top_index):,} candidate neurons")
row_index = top_index // intermediate_size
col_index = top_index % intermediate_size

selected_probs = activation_probs[row_index, col_index]
selected_probs_t = selected_probs.transpose(0, 1)

k_bar = int(round(len(flattened_probs) * ACTIVATION_BAR_RATIO))
if k_bar < 1:
    k_bar = 1
activation_bar = flattened_probs.kthvalue(k_bar).values.item()

lang_index_tensor, selected_idx_in_selected = torch.where(selected_probs_t > activation_bar)
log_progress(f"Applying activation bar threshold: {activation_bar:.6f}")
log_progress(f"Found {len(lang_index_tensor):,} language-specific neuron assignments")

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
            lang_layer_tensors.append(torch.tensor(layer_neurons_sorted, dtype=torch.long))
    final_lang_neurons.append(lang_layer_tensors)

print("\n" + "="*80)
print("Language-specific neurons statistics:")
print("="*80)

total_neurons = num_layers * intermediate_size
for lang_id, lang in enumerate(LANGS):
    neuron_count = sum(len(layer_neurons) for layer_neurons in final_lang_neurons[lang_id])
    percentage = (neuron_count / total_neurons) * 100
    print(f"{lang:25s}: {neuron_count:6d} neurons ({percentage:6.3f}%)")

total_unique_neurons = 0
all_neurons_set = set()
for lang_neurons in final_lang_neurons:
    for layer_idx, layer_neurons in enumerate(lang_neurons):
        for neuron in layer_neurons.tolist():
            all_neurons_set.add((layer_idx, neuron))
total_unique_neurons = len(all_neurons_set)
unique_percentage = (total_unique_neurons / total_neurons) * 100

print("-"*80)
print(f"{'Total unique neurons':25s}: {total_unique_neurons:6d} neurons ({unique_percentage:6.3f}%)")
print(f"{'Total neurons in model':25s}: {total_neurons:6d} neurons")
print("="*80 + "\n")

param_count = 0
for p in model.parameters():
    param_count += p.numel()

model_info = {
    "model_name": MODEL,
    "model_type": model_type,
    "num_layers": int(num_layers),
    "hidden_size": int(model.config.hidden_size),
    "intermediate_size": int(intermediate_size),
    "vocab_size": int(model.config.vocab_size),
    "num_parameters": int(param_count),
}

activation_data = {
    "model": model_info,
    "langs": LANGS,
    "tokens_per_lang": token_counts.cpu(),
    "over_zero": over_zero.cpu(),
    "activation_probs": activation_probs.cpu(),
    "entropy": entropy.cpu(),
    "top_rate": TOP_RATE,
    "filter_rate": FILTER_RATE,
    "activation_bar_ratio": ACTIVATION_BAR_RATIO,
}

mask_data = {
    "model": model_info,
    "langs": LANGS,
    "tokens_per_lang": token_counts.cpu(),
    "lang_neurons": final_lang_neurons,
    "top_rate": TOP_RATE,
    "filter_rate": FILTER_RATE,
    "activation_bar_ratio": ACTIVATION_BAR_RATIO,
    "top_prob_value": top_prob_value,
    "activation_bar": activation_bar,
}

log_progress("\n" + "="*80)
log_progress("Saving results...")
log_progress("="*80)

safe_model_name = MODEL.replace("/", "_")
stats_path = os.path.join(OUTPUT_DIR, safe_model_name + "_activation_stats.pt")
mask_path = os.path.join(OUTPUT_DIR, safe_model_name + "_lang_specific_neurons.pt")

log_progress(f"Saving activation statistics to: {stats_path}")
torch.save(activation_data, stats_path)
log_progress(f"Saving language-specific neurons to: {mask_path}")
torch.save(mask_data, mask_path)

log_progress("\n" + "="*80)
log_progress("LAPE analysis completed successfully!")
log_progress("="*80)
print("\nActivation stats saved to:", stats_path)
print("Language-specific neurons saved to:", mask_path)
