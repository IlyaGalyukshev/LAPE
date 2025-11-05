#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import List, Dict, Any

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser(description="Подсчёт языко-специфичных MLP-нейронов для Mistral-7B-Instruct-v0.2.")
    p.add_argument("--csv", type=str, required=True,
                   help="Путь к параллельному корпусу (CSV) с двумя языковыми колонками.")
    p.add_argument("--lang1_col", type=str, required=True,
                   help="Имя колонки с первым языком.")
    p.add_argument("--lang2_col", type=str, required=True,
                   help="Имя колонки со вторым языком.")
    p.add_argument("--lang1_name", type=str, default="lang1",
                   help="Название первого языка для отчётов (по умолчанию 'lang1').")
    p.add_argument("--lang2_name", type=str, default="lang2",
                   help="Название второго языка для отчётов (по умолчанию 'lang2').")
    p.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                   help="HF модель (MistralForCausalLM) или путь к локальной модели.")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Размер батча по числу предложений.")
    p.add_argument("--max_length", type=int, default=512,
                   help="Максимальная длина токенов на предложение (truncation).")
    p.add_argument("--limit", type=int, default=0,
                   help="Ограничить число строк корпуса (0 = без ограничения).")
    p.add_argument("--save_json", type=str, default="neuron_stats.json",
                   help="Куда сохранить подробную статистику.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cuda или cpu.")
    return p.parse_args()


@torch.no_grad()
def run_language_pass(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    batch_size: int,
    max_length: int,
    current_counter: torch.Tensor,
) -> int:
    """
    Прогоняет список предложений через модель; счётчик активаций (current_counter)
    инкрементируется внутри патченного forward MLP. Возвращает число реально
    обработанных токенов (сумма attention_mask).
    """
    total_tokens = 0
    model.eval()
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, len(texts), batch_size), 1):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Хитрость: считаем токены промпта; логиты не используются
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        total_tokens += int(attention_mask.sum().item())
        
        # Прогресс каждые 10 батчей или на последнем батче
        if batch_idx % 10 == 0 or batch_idx == num_batches:
            print(f"      Прогресс: {batch_idx}/{num_batches} батчей ({total_tokens} токенов)")
    return total_tokens


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    print("="*60)
    print("НАЧАЛО АНАЛИЗА ЯЗЫКО-СПЕЦИФИЧНЫХ НЕЙРОНОВ")
    print(f"Модель: {args.model}")
    print("="*60)
    print(f"Устройство: {device}")
    print(f"Языки: {args.lang1_name} vs {args.lang2_name}")
    print()

    # === Загрузка корпуса ===
    print(f"[1/6] Загрузка корпуса из {args.csv}...")
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Файл не найден: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"      Загружено {len(df)} строк из CSV")
    if args.lang1_col not in df.columns or args.lang2_col not in df.columns:
        raise ValueError(f"В CSV должны присутствовать колонки '{args.lang1_col}' и '{args.lang2_col}'. "
                         f"Найдены: {list(df.columns)}")

    # Очистка и опциональное ограничение
    print(f"[2/6] Очистка данных...")
    lang1_texts = [str(x).strip() for x in df[args.lang1_col].tolist() if isinstance(x, str) and str(x).strip()]
    lang2_texts = [str(x).strip() for x in df[args.lang2_col].tolist() if isinstance(x, str) and str(x).strip()]
    n = min(len(lang1_texts), len(lang2_texts))
    lang1_texts, lang2_texts = lang1_texts[:n], lang2_texts[:n]
    if args.limit and args.limit > 0:
        lang1_texts, lang2_texts = lang1_texts[:args.limit], lang2_texts[:args.limit]
    print(f"      {args.lang1_name} предложений: {len(lang1_texts)}")
    print(f"      {args.lang2_name} предложений: {len(lang2_texts)}")
    print()

    # === Модель и токенайзер ===
    # Используем кэш HuggingFace (по умолчанию ~/.cache/huggingface/)
    print(f"[3/6] Загрузка модели и токенизатора...")
    print(f"      Модель: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, 
        use_fast=True,
        local_files_only=False  # При первом запуске скачает, потом будет использовать кэш
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"      ✓ Токенизатор загружен")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32),
        local_files_only=False  # При первом запуске скачает, потом будет использовать кэш
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"      ✓ Модель загружена на {device}")
    print()

    # === Готовим глобальные счётчики активаций ===
    # Mistral: mlp.gate_proj(out_features) == intermediate_size
    # Число слоёв — len(model.model.layers)
    print(f"[4/6] Подготовка счётчиков активаций...")
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("Неожиданная структура модели. Ожидался MistralForCausalLM с model.layers.")
    num_layers = len(model.model.layers)
    example_mlp = model.model.layers[0].mlp
    intermediate_size = example_mlp.gate_proj.out_features
    print(f"      Слоёв: {num_layers}")
    print(f"      Нейронов на слой: {intermediate_size}")
    print(f"      Всего нейронов: {num_layers * intermediate_size}")

    # Счётчики: [num_layers, intermediate_size], int64 на устройстве
    over_zero_lang1 = torch.zeros((num_layers, intermediate_size), dtype=torch.int64, device=device)
    over_zero_lang2 = torch.zeros((num_layers, intermediate_size), dtype=torch.int64, device=device)

    # === Патчинг forward для всех MLP ===
    # Мы полностью воспроизводим MistralMLP: down_proj( SiLU(gate_proj(x)) * up_proj(x) )
    # и между SiLU и перемножением считаем (>0) по осям (batch, seq_len).
    COUNTER_REF = {"tensor": None}  # небольшой контейнер, чтобы замыкание видело актуальный счётчик

    def make_mlp_forward(layer_idx: int, mlp_module):
        gate = mlp_module.gate_proj
        up = mlp_module.up_proj
        down = mlp_module.down_proj
        act = mlp_module.act_fn  # SiLU

        def patched_forward(x: torch.Tensor) -> torch.Tensor:
            g = gate(x)             # (B, L, I)
            a = act(g)              # (B, L, I)
            # считаем положительные активации
            ctr = COUNTER_REF["tensor"]
            if ctr is not None:
                # сумма по batch и seq -> вектор длины I
                pos = (a > 0).sum(dim=(0, 1))
                # in-place инкремент для конкретного слоя
                ctr[layer_idx].add_(pos.to(ctr.dtype))
            u = up(x)               # (B, L, I)
            y = down(a * u)         # (B, L, H)
            return y

        return patched_forward

    # Сохраним исходные forwards (на всякий случай) и пропатчим
    original_forwards = []
    for li in range(num_layers):
        mlp = model.model.layers[li].mlp
        original_forwards.append(mlp.forward)
        mlp.forward = make_mlp_forward(li, mlp)
    print(f"      ✓ Патчинг {num_layers} MLP слоёв завершён")
    print()

    # === Прогон: Язык 1 ===
    print(f"[5/6] Прогон {args.lang1_name} текстов через модель...")
    print(f"      Предложений: {len(lang1_texts)}, batch_size: {args.batch_size}")
    COUNTER_REF["tensor"] = over_zero_lang1
    lang1_tokens = run_language_pass(
        texts=lang1_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        current_counter=over_zero_lang1,
    )
    print(f"      ✓ {args.lang1_name}: обработано {lang1_tokens} токенов")
    print()

    # === Прогон: Язык 2 ===
    print(f"[6/6] Прогон {args.lang2_name} текстов через модель...")
    print(f"      Предложений: {len(lang2_texts)}, batch_size: {args.batch_size}")
    COUNTER_REF["tensor"] = over_zero_lang2
    lang2_tokens = run_language_pass(
        texts=lang2_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        current_counter=over_zero_lang2,
    )
    print(f"      ✓ {args.lang2_name}: обработано {lang2_tokens} токенов")
    print()

    # (Опционально можно восстановить forward-ы, но в рамках скрипта не требуется)
    # for li in range(num_layers):
    #     model.model.layers[li].mlp.forward = original_forwards[li]

    # === Статистика и агрегаты ===
    print("Вычисление статистики...")
    lang1_active = (over_zero_lang1 > 0)
    lang2_active = (over_zero_lang2 > 0)

    only_lang1_mask = lang1_active & (~lang2_active)
    only_lang2_mask = lang2_active & (~lang1_active)
    both_mask = lang1_active & lang2_active

    total_neurons = num_layers * intermediate_size
    only_lang1_count = int(only_lang1_mask.sum().item())
    only_lang2_count = int(only_lang2_mask.sum().item())
    both_count = int(both_mask.sum().item())
    lang1_any = int(lang1_active.sum().item())
    lang2_any = int(lang2_active.sum().item())

    # Подготовим «топ-lang1» нейроны: сортируем по diff = lang1_cnt - lang2_cnt (по убыванию)
    with torch.no_grad():
        diff = (over_zero_lang1 - over_zero_lang2).view(-1)
        lang1_flat = over_zero_lang1.view(-1)
        lang2_flat = over_zero_lang2.view(-1)
        k = min(200, diff.numel())  # ограничим до 200 строк для JSON
        topk_vals, topk_idx = torch.topk(diff, k=k, largest=True, sorted=True)
        top_lang1_specific = []
        for val, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
            layer = idx // intermediate_size
            neuron = idx % intermediate_size
            top_lang1_specific.append({
                "layer": int(layer),
                "neuron": int(neuron),
                f"{args.lang1_name}_count": int(lang1_flat[idx].item()),
                f"{args.lang2_name}_count": int(lang2_flat[idx].item()),
                "diff": int(val)
            })

    # Список индексов по слоям
    def layerwise_indices(mask_tensor: torch.Tensor) -> List[List[int]]:
        out: List[List[int]] = []
        for li in range(num_layers):
            idx = torch.nonzero(mask_tensor[li], as_tuple=False).squeeze(-1)
            out.append([int(i.item()) for i in idx])
        return out

    result: Dict[str, Any] = {
        "model": args.model,
        "csv": args.csv,
        "columns": {
            f"{args.lang1_name}_col": args.lang1_col, 
            f"{args.lang2_name}_col": args.lang2_col
        },
        "languages": {"lang1": args.lang1_name, "lang2": args.lang2_name},
        "device": str(device),
        "num_layers": num_layers,
        "intermediate_size": intermediate_size,
        "tokens_processed": {args.lang1_name: lang1_tokens, args.lang2_name: lang2_tokens},
        "neuron_totals": {
            "total_neurons": total_neurons,
            f"{args.lang1_name}_active_any": lang1_any,
            f"{args.lang2_name}_active_any": lang2_any,
            "both_active": both_count,
            f"only_{args.lang1_name}": only_lang1_count,
            f"only_{args.lang2_name}": only_lang2_count
        },
        "layerwise_indices": {
            f"only_{args.lang1_name}": layerwise_indices(only_lang1_mask),
            f"only_{args.lang2_name}": layerwise_indices(only_lang2_mask),
            "both": layerwise_indices(both_mask)
        },
        f"top_{args.lang1_name}_specific_by_diff": top_lang1_specific
    }
    print(f"      ✓ Статистика вычислена")
    print()

    # Сохраним JSON
    print(f"Сохранение результатов в {args.save_json}...")
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"      ✓ Результаты сохранены")
    print()

    # Печать короткой сводки
    print("\n=== Сводка по активациям (SiLU(gate_proj) > 0) ===")
    print(f"Модель: {args.model}")
    print(f"Слои: {num_layers} | intermediate_size: {intermediate_size} | всего нейронов: {total_neurons}")
    print(f"Токенов обработано: {args.lang1_name}={lang1_tokens}, {args.lang2_name}={lang2_tokens}")
    print(f"Активные (≥1 раз): {args.lang1_name}={lang1_any}, {args.lang2_name}={lang2_any}")
    print(f"Только {args.lang1_name}: {only_lang1_count}")
    print(f"Только {args.lang2_name}:   {only_lang2_count}")
    print(f"Оба языка:        {both_count}")
    print(f"\nДетальный JSON сохранён в: {args.save_json}")
    print("="*60)
    print("АНАЛИЗ ЗАВЕРШЁН УСПЕШНО!")
    print("="*60)


if __name__ == "__main__":
    main()

