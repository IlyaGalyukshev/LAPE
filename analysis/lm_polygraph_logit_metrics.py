import warnings
warnings.filterwarnings("ignore")
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

import os
import json
import csv
from datetime import datetime
import math
import torch

from transformers import AutoTokenizer
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import Perplexity, MaximumSequenceProbability, MeanTokenEntropy
from lm_polygraph.utils.manager import estimate_uncertainty

MODEL = "meta-llama/Meta-Llama-3.1-8B"
# MODEL = "Tweeties/tweety-tatar-base-7b-2024-v1"
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# MODEL = "ai-forever/mGPT-1.3B-tatar"
# MODEL = "ai-forever/mGPT"
# MODEL = "Qwen/Qwen2.5-7B-Instruct"
# MODEL = "google/gemma-2-9b"
# MODEL = "bigscience/bloomz-7b1-mt"
# MODEL = "bigscience/bloomz-7b1"

DATA_ROOT = "data/TUMLU"
MAX_MEMORY = {0: "13GiB"}

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

PROMPTS = {
    "azerbaijani": """Sual: {question}\n{choices}\n\nCavab: """,
    "crimean-tatar": """Sual: {question}\n{choices}\n\nCevap: """,
    "crimean-tatar-cyrillic": """Суал: {question}\n{choices}\n\nДжевап: """,
    "en": """Question: {question}\n{choices}\n\nAnswer: """,
    "karakalpak": """Soraw: {question}\n{choices}\n\nJuwap: """,
    "kazakh": """Сұрақ: {question}\n{choices}\n\nЖауап: """,
    "kazakh-latin": """Suraq: {question}\n{choices}\n\nJawap: """,
    "ru": """Вопрос: {question}\n{choices}\n\nОтвет: """,
    "tatar": """Сорау: {question}\n{choices}\n\nҖавап: """,
    "turkish": """Soru: {question}\n{choices}\n\nCevap: """,
    "uyghur": """سوئال: {question}\n{choices}\n\nجاۋاب: """,
    "uyghur-latin": """Soal: {question}\n{choices}\n\nJawab: """,
    "uzbek": """Savol: {question}\n{choices}\n\nJavob: """,
    "uzbek-cyrillic": """Савол: {question}\n{choices}\n\nЖавоб: """,
}

OUTPUT_DIR = "uncertainty_metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def format_choices(choices: list) -> str:
    """Format choices with A. B. C. D. labels."""
    labels = ["A", "B", "C", "D"]
    formatted = []
    for i, choice in enumerate(choices):
        if i < len(labels):
            formatted.append(f"{labels[i]}. {choice}")
    return "\n".join(formatted)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def safe_model_id(model_name: str) -> str:
    return (
        model_name.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


log("=" * 80)
log(f"Processing model: {MODEL}")
log("=" * 80)
log(f"Data root: {DATA_ROOT}")
log(f"Output directory: {OUTPUT_DIR}")
log(f"Languages to process: {len(LANGS)}")
log("")

log("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    use_fast=True
)

if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

vocab_size = len(tokenizer)
log(f"Vocab size: {vocab_size}")

log("Loading whitebox model (LM-Polygraph)...")

whitebox_model = WhiteboxModel.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    max_memory=MAX_MEMORY,
)

ppl_estimator = Perplexity()
msp_estimator = MaximumSequenceProbability()
mte_estimator = MeanTokenEntropy()

log("\nFirst pass: counting total tokens per language (full corpora)...")
total_tokens_raw = {}

for lang in LANGS:
    path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")
    log(f"[{lang}] scanning {path}")

    total_tokens = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            
            formatted_choices = format_choices(obj["choices"])
            prompt_template = PROMPTS[lang]
            text = prompt_template.format(
                question=obj["question"],
                choices=formatted_choices
            )
            
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(token_ids)

    total_tokens_raw[lang] = total_tokens
    log(f"  total_tokens_raw = {total_tokens}")

common_tokens = min(total_tokens_raw.values())
log(f"\nCommon token budget per language (min over langs): {common_tokens}")

log("\nSecond pass: estimating uncertainty metrics (up to common_tokens per language)...")
stats_per_lang = {
    lang: {
        "n_examples": 0,
        "used_tokens": 0,
        "ppl_values": [],
        "msp_values": [],
        "mte_values": [],
    }
    for lang in LANGS
}

for lang in LANGS:
    path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")
    log(f"[{lang}] processing {path} for uncertainty metrics")
    log(f"[{lang}] target token budget: {common_tokens}")

    used_tokens = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if used_tokens >= common_tokens:
                break

            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            
            formatted_choices = format_choices(obj["choices"])
            prompt_template = PROMPTS[lang]
            text = prompt_template.format(
                question=obj["question"],
                choices=formatted_choices
            )

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            text_tokens = len(token_ids)
            
            log(f"[{lang}] processing example: {text_tokens} tokens, cumulative: {used_tokens}/{common_tokens}")

            try:
                ppl_out = estimate_uncertainty(
                    whitebox_model,
                    ppl_estimator,
                    input_text=text,
                )
                msp_out = estimate_uncertainty(
                    whitebox_model,
                    msp_estimator,
                    input_text=text,
                )
                mte_out = estimate_uncertainty(
                    whitebox_model,
                    mte_estimator,
                    input_text=text,
                )
                used_tokens += text_tokens
                stats = stats_per_lang[lang]
                stats["n_examples"] += 1
                stats["used_tokens"] = used_tokens
                stats["ppl_values"].append(float(ppl_out.uncertainty))
                stats["msp_values"].append(float(msp_out.uncertainty))
                stats["mte_values"].append(float(mte_out.uncertainty))

                log(f"[{lang}] ppl_out.uncertainty = {ppl_out.uncertainty}")
                log(f"[{lang}] msp_out.uncertainty = {msp_out.uncertainty}")
                log(f"[{lang}] mte_out.uncertainty = {mte_out.uncertainty}")
                if used_tokens + text_tokens > common_tokens:
                    break
            except Exception as e:
                log(f"  [WARN] UE estimation failed for {lang}: {e}")
                torch.cuda.empty_cache()
                continue



    stats = stats_per_lang[lang]
    log(f"[{lang}] done: n_examples={stats['n_examples']}, used_tokens={stats['used_tokens']}")
    log(f"[{lang}] avg PPL={sum(stats['ppl_values'])/len(stats['ppl_values']) if stats['ppl_values'] else 0:.4f}, "
        f"avg MSP={sum(stats['msp_values'])/len(stats['msp_values']) if stats['msp_values'] else 0:.4f}, "
        f"avg MTE={sum(stats['mte_values'])/len(stats['mte_values']) if stats['mte_values'] else 0:.4f}")
    log("")


def mean_and_std(values):
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


model_id_safe = safe_model_id(MODEL)
output_path = os.path.join(
    OUTPUT_DIR,
    f"{model_id_safe}_uncertainty_summary.tsv",
)

log(f"\nSaving aggregated uncertainty metrics to {output_path}")

with open(output_path, "w", encoding="utf-8", newline="") as out_f:
    writer = csv.writer(out_f, delimiter="\t")
    writer.writerow(
        [
            "language",
            "n_examples",
            "used_tokens",
            "perplexity_mean",
            "perplexity_std",
            "maximum_sequence_probability_mean",
            "maximum_sequence_probability_std",
            "mean_token_entropy_mean",
            "mean_token_entropy_std",
        ]
    )

    for lang in LANGS:
        stats = stats_per_lang[lang]
        ppl_mean, ppl_std = mean_and_std(stats["ppl_values"])
        msp_mean, msp_std = mean_and_std(stats["msp_values"])
        mte_mean, mte_std = mean_and_std(stats["mte_values"])

        writer.writerow(
            [
                lang,
                stats["n_examples"],
                stats["used_tokens"],
                f"{ppl_mean:.6f}" if not math.isnan(ppl_mean) else "nan",
                f"{ppl_std:.6f}" if not math.isnan(ppl_std) else "nan",
                f"{msp_mean:.6f}" if not math.isnan(msp_mean) else "nan",
                f"{msp_std:.6f}" if not math.isnan(msp_std) else "nan",
                f"{mte_mean:.6f}" if not math.isnan(mte_mean) else "nan",
                f"{mte_std:.6f}" if not math.isnan(mte_std) else "nan",
            ]
        )

log(f"\nSaved summary statistics to: {output_path}")


stats_json_path = os.path.join(
    OUTPUT_DIR,
    f"{model_id_safe}_stats_per_lang.json",
)

log(f"Saving raw stats_per_lang to: {stats_json_path}")

with open(stats_json_path, "w", encoding="utf-8") as f:
    json.dump(stats_per_lang, f, indent=2, ensure_ascii=False)

log(f"Saved stats for {len(stats_per_lang)} languages")
log("Done.")
