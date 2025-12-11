import os
import json
import csv
from datetime import datetime

from transformers import AutoTokenizer

MODEL = "meta-llama/Meta-Llama-3.1-8B"
MODEL = "Tweeties/tweety-tatar-base-7b-2024-v1"
MODEL = 'mistralai/Mistral-7B-Instruct-v0.2'
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

MAJOR_LANGS = ["en", "ru", "turkish"]

OUTPUT_DIR = "tokenizer_min_tokens_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


log("=" * 80)
log(f"Processing model: {MODEL}")
log("=" * 80)

log("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    use_fast=True,
    add_prefix_space=True
)

if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

vocab_size = len(tokenizer)
log(f"Vocab size: {vocab_size}")

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
            text = obj["question"]
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(token_ids)

    total_tokens_raw[lang] = total_tokens
    log(f"  total_tokens_raw = {total_tokens}")

common_tokens = min(total_tokens_raw.values())
log(f"\nCommon token budget per language (min over langs): {common_tokens}")

log("\nSecond pass: collecting truncated stats (up to common_tokens per language)...")

tokens_per_lang = {}
words_per_lang = {}
unique_tokens_per_lang = {}
chars_per_lang = {}

for lang in LANGS:
    path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")
    log(f"[{lang}] processing with truncation from {path}")

    used_tokens = 0
    words_seen = set()
    token_ids_set = set()
    char_set = set()
    word_offset = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if used_tokens >= common_tokens:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["question"]

            for ch in text:
                char_set.add(ch)

            words = text.strip().split()
            if not words:
                word_offset += 0
                continue

            encoded = tokenizer(
                words,
                is_split_into_words=True,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            ids = encoded["input_ids"]
            word_ids = encoded.word_ids()

            for tid, wid in zip(ids, word_ids):
                if used_tokens >= common_tokens:
                    break
                token_ids_set.add(tid)
                used_tokens += 1
                if wid is not None:
                    global_wid = word_offset + wid
                    words_seen.add(global_wid)

            word_offset += len(words)

    tokens_per_lang[lang] = used_tokens
    words_per_lang[lang] = len(words_seen)
    unique_tokens_per_lang[lang] = token_ids_set
    chars_per_lang[lang] = char_set

    log(
        f"  used_tokens={used_tokens}, words_used={len(words_seen)}, "
        f"unique_tokens={len(token_ids_set)}, unique_chars={len(char_set)}"
    )

log("\nBuilding major language token sets...")
major_sets = {}
for mlang in MAJOR_LANGS:
    if mlang in unique_tokens_per_lang:
        major_sets[mlang] = unique_tokens_per_lang[mlang]
    else:
        major_sets[mlang] = set()

rows = []

log("\nComputing metrics per language...")
for lang in LANGS:
    total_tokens = tokens_per_lang.get(lang, 0)
    total_words = words_per_lang.get(lang, 0)
    token_ids_set = unique_tokens_per_lang.get(lang, set())
    char_set = chars_per_lang.get(lang, set())

    if total_tokens > 0 and total_words > 0:
        fertility = float(total_tokens) / float(total_words)
    else:
        fertility = 0.0

    unique_tokens = len(token_ids_set)
    if vocab_size > 0:
        unique_token_fraction = unique_tokens / float(vocab_size)
    else:
        unique_token_fraction = 0.0

    non_ascii_chars = sorted([c for c in char_set if ord(c) > 127])
    num_unique_chars = len(char_set)
    num_non_ascii_chars = len(non_ascii_chars)
    non_ascii_chars_str = "".join(non_ascii_chars)

    if unique_tokens > 0:
        shared_en_count = len(token_ids_set & major_sets["en"])
        shared_en_fraction = shared_en_count / float(unique_tokens)

        shared_ru_count = len(token_ids_set & major_sets["ru"])
        shared_ru_fraction = shared_ru_count / float(unique_tokens)

        shared_tk_count = len(token_ids_set & major_sets["turkish"])
        shared_tk_fraction = shared_tk_count / float(unique_tokens)
    else:
        shared_en_count = 0
        shared_en_fraction = 0.0
        shared_ru_count = 0
        shared_ru_fraction = 0.0
        shared_tk_count = 0
        shared_tk_fraction = 0.0

    row = {
        "model": MODEL,
        "language": lang,
        "vocab_size": vocab_size,
        "total_tokens_used": total_tokens,
        "total_words_used": total_words,
        "common_tokens_budget": common_tokens,
        "fertility_tokens_per_word": fertility,
        "unique_tokens": unique_tokens,
        "unique_token_fraction": unique_token_fraction,
        "num_unique_chars": num_unique_chars,
        "num_non_ascii_chars": num_non_ascii_chars,
        "non_ascii_chars": non_ascii_chars_str,
        "shared_en_token_count": shared_en_count,
        "shared_en_token_fraction": shared_en_fraction,
        "shared_ru_token_count": shared_ru_count,
        "shared_ru_token_fraction": shared_ru_fraction,
        "shared_turkish_token_count": shared_tk_count,
        "shared_turkish_token_fraction": shared_tk_fraction,
    }
    rows.append(row)

    log(
        f"[{lang}] used_tokens={total_tokens}, "
        f"fert={fertility:.3f}, uniq_frac={unique_token_fraction:.4f}, "
        f"shared_en={shared_en_fraction:.4f}, "
        f"shared_ru={shared_ru_fraction:.4f}, "
        f"shared_turkish={shared_tk_fraction:.4f}"
    )

safe_model_name = MODEL.replace("/", "_")
csv_path = os.path.join(OUTPUT_DIR, safe_model_name + "_tokenization_fertility_metrics.csv")

fieldnames = [
    "model",
    "language",
    "vocab_size",
    "total_tokens_used",
    "total_words_used",
    "common_tokens_budget",
    "fertility_tokens_per_word",
    "unique_tokens",
    "unique_token_fraction",
    "num_unique_chars",
    "num_non_ascii_chars",
    "non_ascii_chars",
    "shared_en_token_count",
    "shared_en_token_fraction",
    "shared_ru_token_count",
    "shared_ru_token_fraction",
    "shared_turkish_token_count",
    "shared_turkish_token_fraction",
]

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

log("=" * 80)
log(f"Done. CSV saved to: {csv_path}")
log("=" * 80)
