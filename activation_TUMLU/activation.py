import os
import re
import gc
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LANGUAGE = "crimean-tatar-cyrillic"

TEST_PROMPTS = {
    "crimean-tatar": """Sual: {question}\n\n1) {choice1}\n2) {choice2}\n3) {choice3}\n4) {choice4}\n\nDoğru cevap: """,
    "crimean-tatar-cyrillic": """Суаль: {question}\n\n1) {choice1}\n2) {choice2}\n3) {choice3}\n4) {choice4}\n\nДогъру джевап: """,
    "tatar": """Сорау: {question}\n\n1) {choice1}\n2) {choice2}\n3) {choice3}\n4) {choice4}\n\nДөрес җавап: """,
}
BENCHMARK_PATH = f"data/TUMLU/{LANGUAGE}/all.jsonl" 
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_NAME = "Tweeties/tweety-tatar-base-7b-2024-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE_COUNT = 4        # батч для подсчёта активаций на входных текстах
MAX_LENGTH_COUNT = 512      # триммим только ПРИ СЧЁТЕ АКТИВАЦИЙ (не при генерации)
GEN_BATCH_SIZE = 4
GEN_MAX_NEW_TOKENS = 8     # генерация не ограничена выбором; обычный decode
GEN_TEMPERATURE = 0.0       # детерминированный гриди по умолчанию
GEN_TOP_P = 1.0
SEED = 42

# LAPE-подобные параметры (одиночный язык)
FILTER_RATE = 0.95          # перцентиль по p(активации) для «существенных» нейронов
TOP_RATE = 0.01             # доля нейронов с минимальной энтропией Бернулли (среди отфильтрованных)
ACTIVATION_BAR_RATIO = 0.95 # порог по перцентилю p(активации) для включения в маску

# Куда сохранять
MODEL_CLEAN = MODEL_NAME.replace("/", "_")
BMK_NAME = os.path.splitext(os.path.basename(BENCHMARK_PATH))[0]
SAVE_DIR_JSON = f"activation_TUMLU/results/json"
SAVE_DIR_PTH  = f"activation_TUMLU/results/pth"
os.makedirs(SAVE_DIR_JSON, exist_ok=True)
os.makedirs(SAVE_DIR_PTH, exist_ok=True)
SAVE_JSON = os.path.join(SAVE_DIR_JSON, f"{BMK_NAME}_{MODEL_CLEAN}_{LANGUAGE}.json")
SAVE_MASK_PTH = os.path.join(SAVE_DIR_PTH, f"{BMK_NAME}_{MODEL_CLEAN}.pth")


def set_seed(seed: int):
    try:
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class QAExample:
    question: str
    choices: List[str]
    answer: str   # "A" | "B" | "C" | "D"
    subject: str = ""

LETTER2IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
IDX2LETTER = {v: k for k, v in LETTER2IDX.items()}

def load_benchmark(path: str) -> List[QAExample]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Не найден бенчмарк: {path}")
    data: List[QAExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = str(obj["question"]).strip()
            choices = [str(c).strip() for c in obj["choices"]]
            ans = str(obj["answer"]).strip().upper()
            subj = str(obj.get("subject", "")).strip()
            # валидация
            if ans not in LETTER2IDX or len(choices) != 4:
                continue
            data.append(QAExample(question=q, choices=choices, answer=ans, subject=subj))
    if not data:
        raise RuntimeError(f"Файл {path} не содержит валидных записей.")
    return data


def build_prompt(ex: QAExample) -> str:
    lines = []
    prompt = TEST_PROMPTS[LANGUAGE]
    lines.append(prompt.format(question=ex.question.strip(), choice1=ex.choices[0], choice2=ex.choices[1], choice3=ex.choices[2], choice4=ex.choices[3]))
    return "\n".join(lines)


def parse_pred_letter(text: str, choices: List[str] = None) -> str:
    """
    Извлекает букву ответа (A-D) из сгенерированного текста.
    Пробует несколько стратегий:
    1. Найти цифру 1-4 после ключевых слов "ответ:" на разных языках
    2. Если не нашли цифру, пробуем сопоставить текст с вариантами ответа
    """
    # Стратегия 1: найти цифру 1-4 после ключевых слов ответа
    # Поддерживаем все три языка:
    # - "Дөрес җавап:" (tatar)
    # - "Doğru cevap:" (crimean-tatar latin)
    # - "Догъру джевап:" (crimean-tatar cyrillic)
    answer_patterns = [
        r"Дөрес җавап:\s*(\d)",      # Tatar
        r"Doğru cevap:\s*(\d)",        # Crimean Tatar (Latin)
        r"Догъру джевап:\s*(\d)",     # Crimean Tatar (Cyrillic)
    ]
    
    for pattern in answer_patterns:
        find_digit = re.search(pattern, text, re.IGNORECASE)
        if find_digit:
            digit = int(find_digit.group(1))
            # Преобразуем 1-4 в A-D (0-3 индексы)
            if 1 <= digit <= 4:
                return IDX2LETTER.get(digit - 1, None)
    
    # Стратегия 2: если есть варианты ответов, попробуем найти совпадение текста
    if choices:
        # Извлекаем текст после ключевых слов ответа
        text_patterns = [
            r"Дөрес җавап:\s*(.+)",
            r"Doğru cevap:\s*(.+)",
            r"Догъру джевап:\s*(.+)",
        ]
        
        answer_text = None
        for pattern in text_patterns:
            match_after = re.search(pattern, text, re.IGNORECASE)
            if match_after:
                answer_text = match_after.group(1).strip()
                break
        
        if answer_text:
            # Удаляем возможные префиксы типа "1) ", "2) " и т.д.
            answer_text = re.sub(r"^\d+\)\s*", "", answer_text)
            # Берем только первую строку (до \n)
            answer_text = answer_text.split('\n')[0].strip()
            
            # Пробуем найти совпадение с вариантами
            for idx, choice in enumerate(choices):
                choice_clean = choice.strip()
                # Проверяем точное совпадение или начало
                if answer_text == choice_clean:
                    return IDX2LETTER.get(idx, None)
                if len(answer_text) > 10 and (answer_text.startswith(choice_clean) or choice_clean.startswith(answer_text[:20])):
                    return IDX2LETTER.get(idx, None)
    
    return None


def detect_mlp_kind(mlp) -> str:
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
        return "gated"  # Mistral/LLaMA
    if hasattr(mlp, "dense_h_to_4h") and hasattr(mlp, "dense_4h_to_h"):
        return "bloom"
    raise RuntimeError("Неизвестный тип MLP-модуля (ожидался gated или bloom).")


def get_layers_and_intermediate(model) -> Tuple[List[Any], int]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = list(model.model.layers)
        first_mlp = layers[0].mlp
        kind = detect_mlp_kind(first_mlp)
        inter = first_mlp.gate_proj.out_features if kind == "gated" else first_mlp.dense_h_to_4h.out_features
        return layers, inter
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = list(model.transformer.h)
        first_mlp = layers[0].mlp
        kind = detect_mlp_kind(first_mlp)
        inter = first_mlp.gate_proj.out_features if kind == "gated" else first_mlp.dense_h_to_4h.out_features
        return layers, inter
    else:
        raise RuntimeError("Не удалось найти список слоёв модели (.model.layers или .transformer.h).")


def make_patchers_single_lang(layers: List[Any], over_zero_ref: Dict[str, torch.Tensor]) -> List[Any]:
    patched = []
    for li, layer in enumerate(layers):
        mlp = layer.mlp
        kind = detect_mlp_kind(mlp)

        if kind == "gated":
            gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
            act = mlp.act_fn  # SiLU

            def fwd(x, gate=gate, up=up, down=down, act=act, li=li):
                g = gate(x)      # (B, L, I)
                a = act(g)       # (B, L, I)
                ctr = over_zero_ref["tensor"]
                if ctr is not None:
                    pos = (a > 0).sum(dim=(0, 1))  # (I,)
                    ctr[li].add_(pos.to(ctr.dtype))
                u = up(x)        # (B, L, I)
                y = down(a * u)  # (B, L, H)
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
    for mlp, orig, fwd in patched:
        mlp.forward = fwd


def restore_patches(patched: List[Any]) -> None:
    for mlp, orig, fwd in patched:
        mlp.forward = orig


def layerwise_indices_from_mask(mask_tensor: torch.Tensor) -> List[List[int]]:
    out: List[List[int]] = []
    for li in range(mask_tensor.size(0)):
        idx = torch.nonzero(mask_tensor[li], as_tuple=False).squeeze(-1)
        out.append([int(i.item()) for i in idx])
    return out


# ===========================
# PASSES
# ===========================
@torch.no_grad()
def count_activations_over_prompts(
    prompts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    over_zero_ref: Dict[str, torch.Tensor],
) -> int:
    """
    Прогоняем ТОЛЬКО входные промпты (вопрос+варианты) без генерации.
    В патченных MLP считаем кол-во токенов с a>0. Возвращаем всего токенов.
    """
    model.eval()
    total_tokens = 0
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        total_tokens += int(attention_mask.sum().item())
        if (i // batch_size + 1) % 10 == 0 or (i + batch_size) >= len(prompts):
            print(f"  Подсчитано батчей: {i // batch_size + 1}/{num_batches} ({total_tokens} токенов)")
    return total_tokens


@torch.no_grad()
def generate_answers(
    prompts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    """
    Генерируем ответы модели по батчам. Не ограничиваемся выбором (никаких логитов по буквам).
    Возвращаем сырые тексты генераций.
    """
    model.eval()
    all_out: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        print(f"i: {i} / {len(prompts)} - Generated texts: {texts}")
        all_out.extend(texts)
    return all_out


# ===========================
# MAIN
# ===========================
def main():
    set_seed(SEED)
    device = torch.device(DEVICE)

    # 1) Данные
    print(f"[1/7] Загрузка бенчмарка из {BENCHMARK_PATH} ...")
    examples = load_benchmark(BENCHMARK_PATH)
    print(f"  Загружено {len(examples)} заданий")

    prompts = [build_prompt(ex) for ex in examples]

    # 2) Модель и токенайзер
    print(f"[2/7] Загрузка токенайзера {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[3/7] Загрузка модели {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32),
        low_cpu_mem_usage=True,
    ).to(device)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # 3) Слои/интермедиат
    layers, intermediate_size = get_layers_and_intermediate(model)
    num_layers = len(layers)
    total_neurons = num_layers * intermediate_size
    print(f"  Модель: {num_layers} слоёв, {intermediate_size} нейронов/слой, всего {total_neurons}")

    # 4) Подсчёт активаций на ВХОДНЫХ промптах (один язык)
    print(f"[4/7] Подсчёт активаций на вопросах бенчмарка ...")
    over_zero = torch.zeros((num_layers, intermediate_size), dtype=torch.int64, device=device)
    over_zero_ref = {"tensor": over_zero}
    patched = make_patchers_single_lang(layers, over_zero_ref)
    apply_patches(patched)

    tokens_prompt = count_activations_over_prompts(
        prompts,
        tokenizer,
        model,
        device,
        batch_size=BATCH_SIZE_COUNT,
        max_length=MAX_LENGTH_COUNT,
        over_zero_ref=over_zero_ref,
    )
    print(f"  Готово: обработано {tokens_prompt} входных токенов.")

    # ВАЖНО: убираем патчи перед генерацией, чтобы счётчики не «загрязнялись» токенами ответа
    restore_patches(patched)
    over_zero_ref["tensor"] = None
    del patched

    # 5) Генерация ответов и вычисление accuracy
    print(f"[5/7] Генерация ответов модели ...")
    generations = generate_answers(
        prompts,
        tokenizer,
        model,
        device,
        batch_size=GEN_BATCH_SIZE,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
    )

    preds_letter: List[str] = [parse_pred_letter(gen, ex.choices) for gen, ex in zip(generations, examples)]
    gold_letter: List[str] = [ex.answer for ex in examples]
    
    # Подсчет правильных ответов (игнорируем случаи, где pred=None)
    correct = sum(1 for p, g in zip(preds_letter, gold_letter) if p is not None and p == g)
    total = len(examples)
    total_valid_preds = sum(1 for p in preds_letter if p is not None)
    accuracy = correct / total if total else 0.0
    
    print(f"  Точность (accuracy): {accuracy:.4f} ({correct}/{total})")
    print(f"  Валидных предсказаний: {total_valid_preds}/{total}")

    # 6) Единоязычные «языковые нейроны» (LAPE-идея адаптирована на один язык)
    print(f"[6/7] Оценка p(активации) и отбор масок под один язык ...")
    eps = 1e-12
    p = over_zero.float() / max(1, tokens_prompt)                # (L, I), вероятность a>0 на входных токенах
    # Бернулли-энтропия (активация/не активация) на одном языке
    entropy = -(p.clamp_min(eps) * p.clamp_min(eps).log() + (1 - p).clamp_min(eps) * (1 - p).clamp_min(eps).log())

    # Фильтр «существенных» по p
    flat_p = p.view(-1)
    k_filter = max(1, int(round(len(flat_p) * FILTER_RATE)))
    top_prob_value = torch.kthvalue(flat_p, k_filter).values.item()
    mask_keep = p > top_prob_value
    entropy_filtered = entropy.clone()
    entropy_filtered[~mask_keep] = float("inf")

    # Отбор нижних TOP_RATE по энтропии среди существенных
    flat_entropy = entropy_filtered.view(-1)
    k_top = max(1, int(round(len(flat_entropy) * TOP_RATE)))
    top_vals, top_idx = torch.topk(-flat_entropy, k=k_top, largest=True)  # мин энтропия
    row_index = top_idx // entropy.size(1)
    col_index = top_idx % entropy.size(1)

    # Порог активации для включения в «языковую» маску
    k_bar = max(1, int(round(len(flat_p) * ACTIVATION_BAR_RATIO)))
    activation_bar = torch.kthvalue(flat_p, k_bar).values.item()

    # Итоговая маска: отобранные нейроны, где p > activation_bar
    selected_mask = torch.zeros_like(p, dtype=torch.bool)
    selected_mask[row_index, col_index] = True
    lang_mask = selected_mask & (p > activation_bar)

    # Списки индексов по слоям
    language_specific_indices = layerwise_indices_from_mask(lang_mask)
    active_any_indices = layerwise_indices_from_mask(over_zero > 0)

    # Топ по p (для справки)
    top_list = []
    diff_rate = p.view(-1)  # «важность» = просто p
    top_vals2, top_idx2 = torch.topk(diff_rate, k=int(diff_rate.numel()), largest=True, sorted=True)
    for val, idx in zip(top_vals2.tolist(), top_idx2.tolist()):
        layer = idx // intermediate_size
        neuron = idx % intermediate_size
        top_list.append(
            {
                "layer": int(layer),
                "neuron": int(neuron),
                "activation_rate": float(val),
            }
        )

    # 7) Сохранение результатов
    print(f"[7/7] Сохранение результатов ...")

    # Предсказания по каждому примеру
    per_item = []
    for i, (ex, gen_text, pred, gold) in enumerate(zip(examples, generations, preds_letter, gold_letter)):
        per_item.append(
            {
                "idx": i,
                "subject": ex.subject,
                "question": ex.question,
                "choices": ex.choices,
                "gold": gold,
                "pred": pred if pred is not None else "NONE",
                "correct": bool(pred is not None and pred == gold),
                "generation": gen_text,
            }
        )

    result: Dict[str, Any] = {
        "model": MODEL_NAME,
        "benchmark_path": BENCHMARK_PATH,
        "device": str(device),
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "tokens_processed_prompt": int(tokens_prompt),
        "accuracy": {
            "correct": int(correct),
            "total": int(total),
            "valid_predictions": int(total_valid_preds),
            "accuracy": float(accuracy),
        },
        "thresholds": {
            "filter_rate": FILTER_RATE,
            "top_rate": TOP_RATE,
            "activation_bar_ratio": ACTIVATION_BAR_RATIO,
            "activation_bar_value": float(activation_bar),
            "top_prob_value": float(top_prob_value),
        },
        "neuron_totals": {
            "total_neurons": int(total_neurons),
            "active_any": int((over_zero > 0).sum().item()),
            "selected_language_specific": int(sum(len(v) for v in language_specific_indices)),
        },
        "layerwise_indices": {
            "active_any": active_any_indices,
            "language_specific_mask": language_specific_indices,  # единый язык
        },
        "top_by_activation_rate": top_list,  # отсортировано по p убыв.
        "items": per_item,  # подробные предсказания по каждому вопросу
    }

    with open(SAVE_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # PTH: единый язык -> List[ per-layer ] of LongTensor
    by_lang_single = []
    for li in range(num_layers):
        idxs = torch.tensor(language_specific_indices[li], dtype=torch.long)
        by_lang_single.append(idxs)
    torch.save(by_lang_single, SAVE_MASK_PTH)

    print("\n=== DONE ===")
    print(f"Benchmark: {BENCHMARK_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"JSON: {SAVE_JSON}")
    print(f"Masks .pth: {SAVE_MASK_PTH}")

    # Очистка памяти
    print("\n=== Очистка памяти ===")
    if torch.cuda.is_available():
        print(f"Память до очистки: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, "
              f"{torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

    # перенос модели на CPU и удаление
    model.cpu()
    del model, tokenizer, layers, over_zero, p, entropy, entropy_filtered
    del flat_p, flat_entropy, selected_mask, lang_mask
    del top_vals, top_idx, row_index, col_index
    del top_vals2, top_idx2, diff_rate
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        time.sleep(2)
        print(f"Память после очистки: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, "
              f"{torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
    print("Память очищена.\n")


if __name__ == "__main__":
    main()
