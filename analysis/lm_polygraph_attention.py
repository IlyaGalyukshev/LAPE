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
from lm_polygraph.estimators import RAUQ, Focus
from lm_polygraph.utils.manager import estimate_uncertainty

MODEL = "meta-llama/Meta-Llama-3.1-8B"
# MODEL = "Tweeties/tweety-tatar-base-7b-2024-v1"
# MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# MODEL = "ai-forever/mGPT-1.3B-tatar"
# MODEL = "ai-forever/mGPT"
# MODEL = "Qwen/Qwen2.5-7B-Instruct"
# MODEL = "google/gemma-2-9b"
# MODEL = "bigscience/bloomz-7b1-mt"
# MODEL = "bigscience/bloomz-7b1"

DATA_ROOT = "data/TUMLU"

MAX_MEMORY = {
    0: "13GiB", 
}

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

OUTPUT_DIR = "uncertainty_metrics_attention"
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


def mean_and_std(values):
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


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
    use_fast=True,
)

if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

vocab_size = len(tokenizer)
log(f"Vocab size: {vocab_size}")

log("Loading whitebox model (LM-Polygraph, CUDA)...")

torch.set_grad_enabled(False)

whitebox_model = WhiteboxModel.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    max_memory=MAX_MEMORY,
)

rauq_estimator = RAUQ()    
focus_estimator = Focus()  


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
                choices=formatted_choices,
            )

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(token_ids)

    total_tokens_raw[lang] = total_tokens
    log(f"  total_tokens_raw = {total_tokens}")

common_tokens = min(total_tokens_raw.values())
log(f"\nCommon token budget per language (min over langs): {common_tokens}")


log("\nSecond pass: estimating RAUQ and Focus (up to common_tokens per language)...")

stats_per_lang = {
    lang: {
        "n_examples": 0,
        "used_tokens": 0,
        "rauq_values": [],
        "focus_values": [],
    }
    for lang in LANGS
}

for lang in LANGS:
    path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")
    log(f"[{lang}] processing {path} for RAUQ & Focus")
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
                choices=formatted_choices,
            )

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            text_tokens = len(token_ids)

            if used_tokens + text_tokens > common_tokens:
                log(
                    f"[{lang}] next example would exceed budget "
                    f"({used_tokens} + {text_tokens} > {common_tokens}), stopping."
                )
                break

            log(
                f"[{lang}] processing example: {text_tokens} tokens, "
                f"cumulative: {used_tokens}/{common_tokens}"
            )

            try:
                with torch.no_grad():
                    rauq_out = estimate_uncertainty(
                        whitebox_model,
                        rauq_estimator,
                        input_text=text,
                    )
                    focus_out = estimate_uncertainty(
                        whitebox_model,
                        focus_estimator,
                        input_text=text,
                    )
            except Exception as e:
                log(f"  [WARN] UE estimation failed for {lang}: {e}")
                torch.cuda.empty_cache()
                continue

            used_tokens += text_tokens

            stats = stats_per_lang[lang]
            stats["n_examples"] += 1
            stats["used_tokens"] = used_tokens
            stats["rauq_values"].append(float(rauq_out.uncertainty))
            stats["focus_values"].append(float(focus_out.uncertainty))

            log(f"[{lang}] RAUQ  uncertainty = {rauq_out.uncertainty}")
            log(f"[{lang}] Focus uncertainty = {focus_out.uncertainty}")

    stats = stats_per_lang[lang]
    if stats["n_examples"] > 0:
        rauq_mean, _ = mean_and_std(stats["rauq_values"])
        focus_mean, _ = mean_and_std(stats["focus_values"])
        log(
            f"[{lang}] done: n_examples={stats['n_examples']}, "
            f"used_tokens={stats['used_tokens']}"
        )
        log(
            f"[{lang}] avg RAUQ={rauq_mean:.4f}, "
            f"avg Focus={focus_mean:.4f}"
        )
    else:
        log(
            f"[{lang}] done: n_examples=0, used_tokens={stats['used_tokens']} "
            f"(no successful UE computations)"
        )
    log("")



model_id_safe = safe_model_id(MODEL)
summary_path = os.path.join(
    OUTPUT_DIR,
    f"{model_id_safe}_attention_uncertainty_summary.tsv",
)

log(f"\nSaving aggregated RAUQ & Focus metrics to {summary_path}")

with open(summary_path, "w", encoding="utf-8", newline="") as out_f:
    writer = csv.writer(out_f, delimiter="\t")
    writer.writerow(
        [
            "language",
            "n_examples",
            "used_tokens",
            "rauq_mean",
            "rauq_std",
            "focus_mean",
            "focus_std",
        ]
    )

    for lang in LANGS:
        stats = stats_per_lang[lang]
        rauq_mean, rauq_std = mean_and_std(stats["rauq_values"])
        focus_mean, focus_std = mean_and_std(stats["focus_values"])

        writer.writerow(
            [
                lang,
                stats["n_examples"],
                stats["used_tokens"],
                f"{rauq_mean:.6f}" if not math.isnan(rauq_mean) else "nan",
                f"{rauq_std:.6f}" if not math.isnan(rauq_std) else "nan",
                f"{focus_mean:.6f}" if not math.isnan(focus_mean) else "nan",
                f"{focus_std:.6f}" if not math.isnan(focus_std) else "nan",
            ]
        )

log(f"Saved summary statistics to: {summary_path}")

stats_json_path = os.path.join(
    OUTPUT_DIR,
    f"{model_id_safe}_attention_stats_per_lang.json",
)

log(f"Saving raw stats_per_lang to: {stats_json_path}")

with open(stats_json_path, "w", encoding="utf-8") as f:
    json.dump(stats_per_lang, f, indent=2, ensure_ascii=False)

log(f"Saved stats for {len(stats_per_lang)} languages")
log("Done.")
