import warnings

warnings.filterwarnings("ignore")

import os
import re
import json
import csv
import random
import zlib
import gc
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()


# -----------------------------
# CONFIG
# -----------------------------
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3-12b-it")
MODEL_PATH = "/hf_models"  # mounted from model_registry

DATA_ROOT = "/work/benchmarks/TUMLU"  # /work/benchmarks is storage mount
OUTPUT_BASE = "/work/benchmarks/uncertainty_metrics"

MAX_NEW_TOKENS = 512

RNG_SEED = 42

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

JSON_INSTRUCTIONS = """Answer the following multiple choice question. Provide your response as a JSON object with two fields:
- "reasoning": your step-by-step reasoning process (from 1 to 3 sentences)
- "answer": a single letter (A, B, C, or D) representing your final answer

RETURN ONLY JSON (NO PROSE, NO MARKDOWN, NO COMMENTS, NO EXTRA WORDS)
Example format:
{"reasoning": "Let me analyze each option...", "answer": "B"}

"""

PROMPTS = {
    "azerbaijani": """Sual: {question}\n{choices}\n\nJSON formatında cavab verin: """,
    "crimean-tatar": """Sual: {question}\n{choices}\n\nJSON formatında cevap verin: """,
    "crimean-tatar-cyrillic": """Суал: {question}\n{choices}\n\nJSON форматында джевап берин: """,
    "en": """Question: {question}\n{choices}\n\nProvide your answer in JSON format: """,
    "karakalpak": """Soraw: {question}\n{choices}\n\nJSON formatinda juwap beriñ: """,
    "kazakh": """Сұрақ: {question}\n{choices}\n\nJSON форматында жауап беріңіз: """,
    "kazakh-latin": """Suraq: {question}\n{choices}\n\nJSON formatynda jawap beriñiz: """,
    "ru": """Вопрос: {question}\n{choices}\n\nОтветьте в формате JSON: """,
    "tatar": """Сорау: {question}\n{choices}\n\nJSON форматында җавап бирегез: """,
    "turkish": """Soru: {question}\n{choices}\n\nJSON formatında cevap verin: """,
    "uyghur": """سوئال: {question}\n{choices}\n\nJSON فورماتىدا جاۋاب بەرىڭ: """,
    "uyghur-latin": """Soal: {question}\n{choices}\n\nJSON formatida jawab bering: """,
    "uzbek": """Savol: {question}\n{choices}\n\nJSON formatida javob bering: """,
    "uzbek-cyrillic": """Савол: {question}\n{choices}\n\nJSON форматида жавоб беринг: """,
}


# -----------------------------
# Logging helpers
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


# -----------------------------
# Tokenization / chat formatting
# -----------------------------
def apply_chat_if_available(tokenizer: AutoTokenizer, user_text: str) -> str:
    try:
        if hasattr(tokenizer, "apply_chat_template") and getattr(
            tokenizer, "chat_template", None
        ):
            messages = [
                {
                    "role": "system",
                    "content": JSON_INSTRUCTIONS,
                },
                {"role": "user", "content": user_text},
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass
    return JSON_INSTRUCTIONS + "\n" + user_text


# -----------------------------
# Shuffle
# -----------------------------
def shuffle_choices(
    options: List[str], answer: str, rng: random.Random
) -> Tuple[List[str], str]:
    shuffled_options = options[:]
    rng.shuffle(shuffled_options)

    answer = str(answer).strip().upper()
    original_index = ord(answer) - ord("A")
    if original_index < 0 or original_index >= len(options):
        return shuffled_options, answer

    try:
        correct_text = options[original_index]
        shuffled_index = shuffled_options.index(correct_text)
        updated_answer = chr(shuffled_index + ord("A"))
        return shuffled_options, updated_answer.upper()
    except Exception:
        return shuffled_options, answer


# -----------------------------
# Parse output -> A/B/C/D
# -----------------------------
def answer_keyword_from_prompt(prompt_template: str) -> str:
    s = prompt_template.strip()
    last_line = s.splitlines()[-1]
    if ":" in last_line:
        return last_line.split(":", 1)[0].strip()
    return last_line.strip()


ANSWER_KEYWORDS = {
    lang: answer_keyword_from_prompt(PROMPTS[lang]) for lang in PROMPTS.keys()
}


def parse_json_response(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse JSON response from model.
    Returns (reasoning, answer) tuple.
    """
    text = text.strip().replace("\\", "")

    # Try to extract JSON from markdown code blocks
    markdown_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if markdown_match:
        text = markdown_match.group(1)

    # Try to find JSON in the response
    start = text.find("{")
    if start != -1:
        brace_count = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        if end > start:
            json_str = text[start:end]
            try:
                data = json.loads(json_str)
                reasoning = data.get("reasoning", "")
                answer = data.get("answer", "")
                if answer:
                    answer = str(answer).strip().upper()
                    if answer in ["A", "B", "C", "D"]:
                        return reasoning, answer
                    letter_match = re.search(r"\b([ABCD])\b", answer)
                    if letter_match:
                        return reasoning, letter_match.group(1)
            except json.JSONDecodeError:
                pass

    return None, None


def find_matching_pattern(text: str, language: str) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(text))
    text = text.replace("*", "")
    text = text.replace("А", "A")
    text = text.replace("В", "B")
    text = text.replace("Б", "B")
    text = text.replace("С", "C")
    text = text.replace("Д", "D")

    word = ANSWER_KEYWORDS[language]

    patterns = {
        "A": [
            rf"{word}: A",
            rf"{word.lower()}: A",
            rf"{word} A\)",
            rf"{word.lower()} A\)",
            rf"{word} A ",
            rf"{word.lower()} A ",
            r"A\)",
        ],
        "B": [
            rf"{word}: B",
            rf"{word.lower()}: B",
            rf"{word} B\)",
            rf"{word.lower()} B\)",
            rf"{word} B ",
            rf"{word.lower()} B ",
            r"B\)",
        ],
        "C": [
            rf"{word}: C",
            rf"{word.lower()}: C",
            rf"{word} C\)",
            rf"{word.lower()} C\)",
            rf"{word} C ",
            rf"{word.lower()} C ",
            r"C\)",
        ],
        "D": [
            rf"{word}: D",
            rf"{word.lower()}: D",
            rf"{word} D\)",
            rf"{word.lower()} D\)",
            rf"{word} D ",
            rf"{word.lower()} D ",
            r"D\)",
        ],
    }

    for i in range(len(patterns["A"])):
        if re.search(patterns["A"][i], text):
            return "A"
        if re.search(patterns["B"][i], text):
            return "B"
        if re.search(patterns["C"][i], text):
            return "C"
        if re.search(patterns["D"][i], text):
            return "D"

    m = re.fullmatch(r"\s*([ABCD])\s*", str(text).strip().upper())
    if m:
        return m.group(1)
    m = re.fullmatch(r"\s*([ABCD])[\.\)]\s*", str(text).strip().upper())
    if m:
        return m.group(1)

    return None


# -----------------------------
# Data loading
# -----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_existing_results(path: str) -> Dict[int, Dict[str, Any]]:
    """
    Load existing results from output file.
    Returns dict mapping index to result record.
    """
    results = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    idx = rec.get("index")
                    if idx is not None:
                        results[idx] = rec
        except Exception as e:
            log(f"Warning: Could not load existing results from {path}: {e}")
    return results


# -----------------------------
# Prompt building
# -----------------------------
def make_user_prompt(lang: str, question: str, choices: List[str]) -> str:
    formatted_choices = format_choices(choices)
    return PROMPTS[lang].format(question=question, choices=formatted_choices)


def stable_int(s: str) -> int:
    return int(zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)


# -----------------------------
# Greedy generation (single)
# -----------------------------
@torch.inference_mode()
def generate_greedy_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_input_text: str,
    max_new_tokens: int,
) -> str:
    enc = tokenizer(
        model_input_text,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(
        model.device
    )

    prompt_len = int(attention_mask.sum().item())
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )[0]

    gen_tokens = out[prompt_len:]

    if (
        tokenizer.eos_token_id is not None
        and gen_tokens.numel() > 0
        and int(gen_tokens[-1].item()) == int(tokenizer.eos_token_id)
    ):
        gen_tokens = gen_tokens[:-1]

    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    model_safe = safe_model_id(MODEL_ID)
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, model_safe, "evaluate")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("=" * 80)
    log(f"Processing model: {MODEL_ID}")
    log(f"Local model path: {MODEL_PATH}")
    log("=" * 80)
    log(f"Data root: {DATA_ROOT}")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"Languages to process: {len(LANGS)}")
    log(f"RNG_SEED = {RNG_SEED}")
    log("Response format: JSON with reasoning and answer")
    log("")

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        local_files_only=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.eos_token_id is None and tokenizer.eos_token is not None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    log(f"Vocab size: {len(tokenizer)}")
    log(
        f"EOS token id: {tokenizer.eos_token_id}, PAD token id: {tokenizer.pad_token_id}"
    )
    log(f"tokenizer.padding_side = {tokenizer.padding_side}")
    log("")

    log("Loading model (device_map='auto')...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    log(f"Model loaded. torch.cuda.device_count() = {torch.cuda.device_count()}")
    log("")

    global_total = 0
    global_correct = 0
    lang_stats: Dict[str, Dict[str, Any]] = {}

    for lang in LANGS:
        path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")
        if not os.path.exists(path):
            log(f"[{lang}] SKIP: file not found: {path}")
            continue

        log("=" * 80)
        log(f"[{lang}] loading {path}")
        data = load_jsonl(path)
        if not data:
            log(f"[{lang}] SKIP: empty dataset")
            continue

        # Prepare eval items (no few-shot, direct evaluation)
        eval_items: List[Tuple[str, Dict[str, Any]]] = []
        for obj in data:
            subj = obj.get("subject", obj.get("category", ""))
            subj = str(subj) if subj is not None else ""
            eval_items.append((subj, obj))

        n = len(eval_items)
        if n == 0:
            log(f"[{lang}] SKIP: no eval items")
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{lang}_eval.jsonl")

        # Load existing results to resume from where we left off
        existing_results = load_existing_results(out_path)

        log(f"[{lang}] eval_items={n}")
        log(f"[{lang}] writing per-question results to: {out_path}")
        log(f"[{lang}] Found {len(existing_results)} existing results, will skip those")
        log("")

        correct = 0
        total = 0

        with open(out_path, "a", encoding="utf-8") as out_f:
            for i, (subject, obj) in enumerate(eval_items, start=1):
                if i in existing_results:
                    result = existing_results[i]
                    total += 1
                    if result.get("is_correct", False):
                        correct += 1
                    global_total += 1
                    if result.get("is_correct", False):
                        global_correct += 1
                    continue

                question = str(obj.get("question", "")).strip()
                choices = obj.get("choices", [])
                gold = str(obj.get("answer", "")).strip().upper()

                if not isinstance(choices, list):
                    choices = list(choices) if choices is not None else []

                if len(choices) > 4:
                    raise AssertionError(
                        "The number of choices should not be more than 4!"
                    )

                rng = random.Random(RNG_SEED + i * 10007 + stable_int(subject))
                shuffled_choices, shuffled_gold = shuffle_choices(choices, gold, rng)

                user_prompt = make_user_prompt(lang, question, shuffled_choices)
                model_input = apply_chat_if_available(tokenizer, user_prompt)

                log(f"[{lang}] [{i}/{n}] category={subject}")
                log(f"[{lang}] Q: {question}")
                log(f"[{lang}] CHOICES (shuffled):")
                for k, c in enumerate(shuffled_choices[:4]):
                    log(f"[{lang}]   {chr(ord('A') + k)}. {c}")
                log(f"[{lang}] GOLD: {shuffled_gold}")

                output = generate_greedy_single(
                    model=model,
                    tokenizer=tokenizer,
                    model_input_text=model_input,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                reasoning, pred = parse_json_response(output)

                if pred is None:
                    pred = find_matching_pattern(output, lang)

                is_correct = bool(pred == shuffled_gold)

                total += 1
                if is_correct:
                    correct += 1

                global_total += 1
                if is_correct:
                    global_correct += 1

                log(f"[{lang}] MODEL_OUTPUT: {output}")
                log(f"[{lang}] REASONING: {reasoning if reasoning else '[NONE]'}")
                log(f"[{lang}] PRED: {pred if pred is not None else '[INVALID]'}")
                log(f"[{lang}] CORRECT: {is_correct}")
                log("")

                rec = {
                    "index": i,
                    "question": question,
                    "choices": shuffled_choices,
                    "answer": shuffled_gold,
                    "category": subject,
                    "language": lang,
                    "model_output": output,
                    "reasoning": reasoning if reasoning else "",
                    "predicted_answer": pred if pred else "",
                    "is_correct": is_correct,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()

                # Periodic memory cleanup every 10 questions
                if i % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        acc = 100.0 * (correct / total) if total else float("nan")
        log(f"[{lang}] DONE: correct={correct}/{total} -> accuracy={acc:.2f}%")
        log("")

        lang_stats[lang] = {
            "n_examples": total,
            "correct": correct,
            "accuracy": acc,
        }

        # Clean up memory after each language
        gc.collect()
        torch.cuda.empty_cache()

    # ---------- Save summary TSV
    summary_path = os.path.join(OUTPUT_DIR, "evaluate_summary.tsv")
    log(f"Saving summary to {summary_path}")

    with open(summary_path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(["language", "n_examples", "correct", "accuracy"])
        for lang in LANGS:
            if lang not in lang_stats:
                continue
            st = lang_stats[lang]
            writer.writerow(
                [
                    lang,
                    st["n_examples"],
                    st["correct"],
                    f"{st['accuracy']:.2f}",
                ]
            )

    if global_total > 0:
        global_acc = 100.0 * (global_correct / global_total)
        log("=" * 80)
        log(
            f"GLOBAL: correct={global_correct}/{global_total} -> accuracy={global_acc:.2f}%"
        )
        log("Done.")
    else:
        log("No data processed. Done.")


if __name__ == "__main__":
    main()
