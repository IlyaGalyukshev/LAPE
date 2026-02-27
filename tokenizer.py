import warnings

warnings.filterwarnings("ignore")

import os
import json
import csv
from datetime import datetime
from typing import Dict, Set, List, Any

from transformers import AutoTokenizer
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

MAJOR_LANGS = ["en", "ru", "turkish"]


# -----------------------------
# Helpers
# -----------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


def mean_std(values: List[float]) -> (float, float):
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0, 0.0
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return float(mean), 0.0
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    return float(mean), float(var**0.5)


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    safe_name = safe_model_id(MODEL_ID)
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, safe_name, "tokenizer")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("=" * 80)
    log(f"Processing model: {MODEL_ID}")
    log(f"MODEL_PATH: {MODEL_PATH}")
    log(f"DATA_ROOT: {DATA_ROOT}")
    log(f"OUTPUT_DIR: {OUTPUT_DIR}")
    log("=" * 80)

    log("Loading tokenizer (local_files_only=True)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            use_fast=True,
            add_prefix_space=True,
            local_files_only=True,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            use_fast=True,
            local_files_only=True,
        )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    log(f"Vocab size: {vocab_size}")

    log("\nFirst pass: counting total tokens per language (full corpora)...")
    total_tokens_raw: Dict[str, int] = {}

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
                text = make_user_prompt(lang, obj)
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(token_ids)

        total_tokens_raw[lang] = total_tokens
        log(f"  total_tokens_raw = {total_tokens}")

    common_tokens = min(total_tokens_raw.values()) if total_tokens_raw else 0
    log(f"\nCommon token budget per language (min over langs): {common_tokens}")

    log(
        "\nSecond pass: collecting truncated stats (up to common_tokens per language)..."
    )

    checkpoint_path = os.path.join(OUTPUT_DIR, "tokenizer_checkpoint.json")

    tokens_per_lang: Dict[str, int] = {}
    words_per_lang: Dict[str, int] = {}
    unique_tokens_per_lang: Dict[str, Set[int]] = {}
    chars_per_lang: Dict[str, Set[str]] = {}
    chars_per_token_values_per_lang: Dict[str, List[float]] = {}

    # Load checkpoint if exists
    completed_langs: List[str] = []
    if os.path.exists(checkpoint_path):
        log(f"Loading checkpoint from {checkpoint_path}...")
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as ckf:
                ckpt = json.load(ckf)
            completed_langs = ckpt.get("completed_langs", [])
            for cl in completed_langs:
                tokens_per_lang[cl] = ckpt["tokens_per_lang"][cl]
                words_per_lang[cl] = ckpt["words_per_lang"][cl]
                unique_tokens_per_lang[cl] = set(ckpt["unique_tokens_per_lang"][cl])
                chars_per_lang[cl] = set(ckpt["chars_per_lang"][cl])
                chars_per_token_values_per_lang[cl] = ckpt[
                    "chars_per_token_values_per_lang"
                ][cl]
            log(f"Resumed: {len(completed_langs)} languages already completed")
        except Exception as e:
            log(f"Warning: Could not load checkpoint: {e}")
            completed_langs = []

    for lang in LANGS:
        if lang in completed_langs:
            log(f"[{lang}] skipping (already completed)")
            continue

        path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")
        log(f"[{lang}] processing with truncation from {path}")

        used_tokens = 0
        words_seen = set()
        token_ids_set: Set[int] = set()
        char_set: Set[str] = set()
        word_offset = 0

        cpt_values: List[float] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if used_tokens >= common_tokens:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = make_user_prompt(lang, obj)

                for ch in text:
                    char_set.add(ch)

                words = text.strip().split()
                if not words:
                    continue

                encoded = tokenizer(
                    words,
                    is_split_into_words=True,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )

                ids = encoded["input_ids"]
                word_ids = encoded.word_ids()

                tok_strs = tokenizer.convert_ids_to_tokens(ids)

                for tid, wid, tstr in zip(ids, word_ids, tok_strs):
                    if used_tokens >= common_tokens:
                        break

                    token_ids_set.add(int(tid))
                    used_tokens += 1

                    cpt_values.append(float(len(tstr)))

                    if wid is not None:
                        global_wid = word_offset + int(wid)
                        words_seen.add(global_wid)

                word_offset += len(words)

        tokens_per_lang[lang] = used_tokens
        words_per_lang[lang] = len(words_seen)
        unique_tokens_per_lang[lang] = token_ids_set
        chars_per_lang[lang] = char_set
        chars_per_token_values_per_lang[lang] = cpt_values

        log(
            f"  used_tokens={used_tokens}, words_used={len(words_seen)}, "
            f"unique_tokens={len(token_ids_set)}, unique_chars={len(char_set)}"
        )

        # Save checkpoint after each language
        completed_langs.append(lang)
        ckpt_data = {
            "completed_langs": completed_langs,
            "tokens_per_lang": {l: tokens_per_lang[l] for l in completed_langs},
            "words_per_lang": {l: words_per_lang[l] for l in completed_langs},
            "unique_tokens_per_lang": {
                l: sorted(unique_tokens_per_lang[l]) for l in completed_langs
            },
            "chars_per_lang": {l: sorted(chars_per_lang[l]) for l in completed_langs},
            "chars_per_token_values_per_lang": {
                l: chars_per_token_values_per_lang[l] for l in completed_langs
            },
        }
        with open(checkpoint_path, "w", encoding="utf-8") as ckf:
            json.dump(ckpt_data, ckf, ensure_ascii=False)
        log(f"  Checkpoint saved ({len(completed_langs)}/{len(LANGS)} languages)")

    log("\nBuilding major language token sets...")
    major_sets: Dict[str, Set[int]] = {}
    for mlang in MAJOR_LANGS:
        major_sets[mlang] = unique_tokens_per_lang.get(mlang, set())

    rows: List[Dict[str, Any]] = []

    log("\nComputing metrics per language...")
    for lang in LANGS:
        total_tokens = tokens_per_lang.get(lang, 0)
        total_words = words_per_lang.get(lang, 0)
        token_ids_set = unique_tokens_per_lang.get(lang, set())
        char_set = chars_per_lang.get(lang, set())

        fertility = (
            (float(total_tokens) / float(total_words))
            if (total_tokens > 0 and total_words > 0)
            else 0.0
        )

        unique_tokens = len(token_ids_set)
        unique_token_fraction = (
            (unique_tokens / float(vocab_size)) if vocab_size > 0 else 0.0
        )

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

        cpt_vals = chars_per_token_values_per_lang.get(lang, [])
        cpt_mean, cpt_std = mean_std(cpt_vals)

        row = {
            "model": MODEL_ID,
            "model_path": MODEL_PATH,
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
            "chars_per_token_mean": cpt_mean,
            "chars_per_token_std": cpt_std,
        }
        rows.append(row)

        log(
            f"[{lang}] used_tokens={total_tokens}, "
            f"fert={fertility:.3f}, uniq_frac={unique_token_fraction:.4f}, "
            f"shared_en={shared_en_fraction:.4f}, "
            f"shared_ru={shared_ru_fraction:.4f}, "
            f"shared_turkish={shared_tk_fraction:.4f}, "
            f"cpt_mean={cpt_mean:.3f}, cpt_std={cpt_std:.3f}"
        )

    tsv_path = os.path.join(OUTPUT_DIR, "tokenizer_summary.tsv")
    log(f"Saving summary to {tsv_path}")

    fieldnames = [
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
        "shared_en_token_count",
        "shared_en_token_fraction",
        "shared_ru_token_count",
        "shared_ru_token_fraction",
        "shared_turkish_token_count",
        "shared_turkish_token_fraction",
        "chars_per_token_mean",
        "chars_per_token_std",
    ]

    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            tsv_row = {k: row[k] for k in fieldnames}
            writer.writerow(tsv_row)

    # Remove checkpoint after successful save
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        log("Checkpoint removed (run completed successfully)")

    log("=" * 80)
    log(f"Done. TSV saved to: {tsv_path}")
    log("=" * 80)


if __name__ == "__main__":
    main()
