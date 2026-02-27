import warnings

warnings.filterwarnings("ignore")

import os
import json
import csv
import math
import gc
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
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

MAX_NEW_TOKENS = 64
LOG_GREEDY_ANSWER = True

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
def make_user_prompt(lang: str, obj: Dict[str, Any]) -> str:
    formatted_choices = format_choices(obj["choices"])
    prompt_template = PROMPTS[lang]
    return prompt_template.format(question=obj["question"], choices=formatted_choices)


def apply_chat_if_available(tokenizer: AutoTokenizer, user_text: str) -> str:
    """
    If tokenizer has a chat template, wrap prompt into chat format.
    """
    try:
        if hasattr(tokenizer, "apply_chat_template") and getattr(
            tokenizer, "chat_template", None
        ):
            messages = [
                {"role": "user", "content": user_text},
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass
    return user_text


# -----------------------------
# Aggregation helpers
# -----------------------------
def load_existing_results(path: str) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
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


def mean_and_std(values: List[float]) -> Tuple[float, float]:
    vals = [
        v
        for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not vals:
        return float("nan"), float("nan")
    arr = np.array(vals, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    return mean, std


# -----------------------------
# Greedy generation (single)
# -----------------------------
@torch.inference_mode()
def generate_greedy_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_input_text: str,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, int, int]:
    """
    Returns:
      full_seq: (L,) token ids = prompt + generated (cut at first EOS, includes EOS if present)
      prompt_len: int
      gen_len: int (includes EOS if it was generated)
    """
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

    gen_part = out[prompt_len:]
    gen_len = int(gen_part.shape[0])

    if tokenizer.eos_token_id is not None and gen_len > 0:
        eos_pos = (gen_part == tokenizer.eos_token_id).nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            gen_len = int(eos_pos[0].item()) + 1  # include EOS

    full_seq = out[: prompt_len + gen_len]
    return full_seq, prompt_len, gen_len


# -----------------------------
# Whitebox metrics (single)
# -----------------------------
@torch.inference_mode()
def compute_metrics_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    full_seq: torch.Tensor,  # (L,)
    prompt_len: int,
    gen_len: int,
) -> Dict[str, float]:
    """
    Metrics:
      - MeanTokenNLL: -mean(log_likelihoods)        — mean negative log-likelihood per token
      - SequenceNLL: -sum(log_likelihoods)           — total negative log-likelihood of the sequence
      - MeanTokenEntropy: mean(entropy)              — mean entropy of predicted distributions
    where log_likelihoods are log p(y_i | y_<i, x) for generated tokens.
    """
    if gen_len <= 0 or prompt_len <= 0:
        return {
            "MeanTokenNLL": float("nan"),
            "SequenceNLL": float("nan"),
            "MeanTokenEntropy": float("nan"),
        }

    input_ids = full_seq.unsqueeze(0)  # (1, L)
    attention_mask = torch.ones_like(
        input_ids, dtype=torch.long, device=input_ids.device
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = outputs.logits.to(dtype=torch.float32)  # (1, L, V)

    # Predictions for generated tokens:
    # target positions: [prompt_len .. prompt_len+gen_len-1]
    # logits positions:  [prompt_len-1 .. prompt_len+gen_len-2]
    gen_start = prompt_len
    gen_end = prompt_len + gen_len
    pred_start = gen_start - 1
    pred_end = gen_end - 1

    if pred_start < 0 or pred_end <= pred_start:
        return {
            "MeanTokenNLL": float("nan"),
            "SequenceNLL": float("nan"),
            "MeanTokenEntropy": float("nan"),
        }

    logits_pred = logits[0, pred_start:pred_end, :]  # (gen_len, V)
    target = input_ids[0, gen_start:gen_end].to(dtype=torch.long)  # (gen_len,)

    logp_all = torch.log_softmax(logits_pred, dim=-1)  # (gen_len, V)
    logp_t = logp_all.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(
        -1
    )  # (gen_len,)

    logp_np = logp_t.detach().cpu().numpy().astype(np.float64)

    # MeanTokenNLL = -mean(log_likelihoods)
    mean_nll_val = float(-np.mean(logp_np))
    # SequenceNLL = -sum(log_likelihoods)
    seq_nll_val = float(-np.sum(logp_np))

    # Entropy per token: -sum p log p
    p_all = torch.exp(logp_all)
    entropy_t = -(p_all * logp_all).sum(dim=-1)  # (gen_len,)
    entropy_np = entropy_t.detach().cpu().numpy().astype(np.float64)
    mean_entropy_val = float(np.mean(entropy_np))

    return {
        "MeanTokenNLL": mean_nll_val,
        "SequenceNLL": seq_nll_val,
        "MeanTokenEntropy": mean_entropy_val,
    }


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    model_id_safe = safe_model_id(MODEL_ID)
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, model_id_safe, "nll_entropy")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("=" * 80)
    log(f"Processing model: {MODEL_ID}")
    log(f"Local model path: {MODEL_PATH}")
    log("=" * 80)
    log(f"Data root: {DATA_ROOT}")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"Languages to process: {len(LANGS)}")
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

    # ---------- First pass: total prompt tokens per language (full corpora)
    log("First pass: counting total prompt tokens per language (full corpora)...")
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
                user_prompt = make_user_prompt(lang, obj)
                model_input = apply_chat_if_available(tokenizer, user_prompt)
                token_ids = tokenizer.encode(model_input, add_special_tokens=False)
                total_tokens += len(token_ids)

        total_tokens_raw[lang] = total_tokens
        log(f"  total_tokens_raw = {total_tokens}")

    common_tokens = min(total_tokens_raw.values()) if total_tokens_raw else 0
    log(f"\nCommon token budget per language (min over langs): {common_tokens}")
    log("")

    # ---------- Second pass: compute metrics up to common_tokens per language
    log("Second pass: estimating metrics (up to common_tokens per language)...")

    stats_per_lang: Dict[str, Dict[str, Any]] = {
        lang: {
            "n_examples": 0,
            "used_tokens": 0,
            "MeanTokenNLL_values": [],
            "SequenceNLL_values": [],
            "MeanTokenEntropy_values": [],
        }
        for lang in LANGS
    }

    for lang in LANGS:
        path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")
        log(f"[{lang}] processing {path}")
        log(f"[{lang}] target token budget: {common_tokens}")

        checkpoint_path = os.path.join(OUTPUT_DIR, f"{lang}_nll_entropy.jsonl")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        existing_results = load_existing_results(checkpoint_path)
        log(f"[{lang}] Found {len(existing_results)} existing results, will skip those")

        used_tokens = 0
        question_idx = 0

        with open(checkpoint_path, "a", encoding="utf-8") as out_f:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if used_tokens >= common_tokens:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    question_idx += 1

                    obj = json.loads(line)
                    user_prompt = make_user_prompt(lang, obj)
                    model_input = apply_chat_if_available(tokenizer, user_prompt)

                    token_ids = tokenizer.encode(model_input, add_special_tokens=False)
                    text_tokens = int(len(token_ids))

                    if used_tokens + text_tokens > common_tokens:
                        break

                    # Resume
                    if question_idx in existing_results:
                        rec = existing_results[question_idx]
                        st = stats_per_lang[lang]
                        st["n_examples"] += 1
                        st["MeanTokenNLL_values"].append(float(rec["MeanTokenNLL"]))
                        st["SequenceNLL_values"].append(float(rec["SequenceNLL"]))
                        st["MeanTokenEntropy_values"].append(
                            float(rec["MeanTokenEntropy"])
                        )
                        used_tokens += text_tokens
                        st["used_tokens"] = used_tokens
                        continue

                    log(
                        f"[{lang}] [{question_idx}] example: {text_tokens} prompt tokens, cumulative: {used_tokens}/{common_tokens}"
                    )

                    if LOG_GREEDY_ANSWER:
                        log(f"[{lang}] PROMPT:\n{user_prompt}")

                    # one greedy generation per request
                    full_seq, prompt_len, gen_len = generate_greedy_single(
                        model=model,
                        tokenizer=tokenizer,
                        model_input_text=model_input,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )

                    if LOG_GREEDY_ANSWER:
                        gen_tokens = full_seq[prompt_len:]
                        if (
                            tokenizer.eos_token_id is not None
                            and gen_tokens.numel() > 0
                            and int(gen_tokens[-1].item())
                            == int(tokenizer.eos_token_id)
                        ):
                            gen_tokens = gen_tokens[:-1]
                        ans = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                        log(f"[{lang}] ANSWER:\n{ans}")

                    m = compute_metrics_single(
                        model=model,
                        tokenizer=tokenizer,
                        full_seq=full_seq,
                        prompt_len=prompt_len,
                        gen_len=gen_len,
                    )

                    st = stats_per_lang[lang]
                    st["n_examples"] += 1
                    st["MeanTokenNLL_values"].append(float(m["MeanTokenNLL"]))
                    st["SequenceNLL_values"].append(float(m["SequenceNLL"]))
                    st["MeanTokenEntropy_values"].append(float(m["MeanTokenEntropy"]))

                    used_tokens += text_tokens
                    st["used_tokens"] = used_tokens

                    # Save checkpoint
                    rec = {
                        "index": question_idx,
                        "language": lang,
                        "text_tokens": text_tokens,
                        "MeanTokenNLL": float(m["MeanTokenNLL"]),
                        "SequenceNLL": float(m["SequenceNLL"]),
                        "MeanTokenEntropy": float(m["MeanTokenEntropy"]),
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()

                    log(
                        f"[{lang}] METRICS: "
                        f"MeanTokenNLL={m['MeanTokenNLL']:.6f}, "
                        f"SequenceNLL={m['SequenceNLL']:.6f}, "
                        f"MeanTokenEntropy={m['MeanTokenEntropy']:.6f}"
                    )
                    log("")

                    # Periodic memory cleanup every 10 questions
                    if question_idx % 10 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

        st = stats_per_lang[lang]
        log(
            f"[{lang}] done: n_examples={st['n_examples']}, used_tokens={st['used_tokens']}"
        )
        log("")

        # Clean up memory after each language
        gc.collect()
        torch.cuda.empty_cache()

    # ---------- Save summary
    output_path = os.path.join(OUTPUT_DIR, "nll_entropy_summary.tsv")
    log(f"Saving aggregated metrics to {output_path}")

    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(
            [
                "language",
                "n_examples",
                "used_tokens",
                "MeanTokenNLL_mean",
                "MeanTokenNLL_std",
                "SequenceNLL_mean",
                "SequenceNLL_std",
                "MeanTokenEntropy_mean",
                "MeanTokenEntropy_std",
            ]
        )

        for lang in LANGS:
            st = stats_per_lang[lang]
            nll_m, nll_s = mean_and_std(st["MeanTokenNLL_values"])
            seq_m, seq_s = mean_and_std(st["SequenceNLL_values"])
            ent_m, ent_s = mean_and_std(st["MeanTokenEntropy_values"])

            def fmt(x: float) -> str:
                return f"{x:.6f}" if not math.isnan(x) else "nan"

            writer.writerow(
                [
                    lang,
                    st["n_examples"],
                    st["used_tokens"],
                    fmt(nll_m),
                    fmt(nll_s),
                    fmt(seq_m),
                    fmt(seq_s),
                    fmt(ent_m),
                    fmt(ent_s),
                ]
            )

    stats_json_path = os.path.join(OUTPUT_DIR, "nll_entropy_stats_per_lang.json")
    log(f"Saving raw stats_per_lang to: {stats_json_path}")
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_per_lang, f, indent=2, ensure_ascii=False)

    log(f"Saved stats for {len(stats_per_lang)} languages")
    log("Done.")


if __name__ == "__main__":
    main()
