import json
from typing import Dict, Any, Tuple, List
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

LANGUAGE = "crimean-tatar-cyrillic"
MODEL1_NAME = "Tweeties/tweety-tatar-base-7b-2024-v1"
MODEL2_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

JSON1_PATH = f"activation_TUMLU/results/json/all_{MODEL1_NAME.replace('/', '_')}_{LANGUAGE}.json"
JSON2_PATH = f"activation_TUMLU/results/json/all_{MODEL2_NAME.replace('/', '_')}_{LANGUAGE}.json"

OUTPUT_DIR = "activation_TUMLU/results/comparison"
OUTPUT_PREFIX = f"{OUTPUT_DIR}/{LANGUAGE}_{MODEL1_NAME.replace('/', '_')}_vs_{MODEL2_NAME.replace('/', '_')}"


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    print(f"Загрузка {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_by_subject(items: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """Analyze accuracy by subject."""
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0, "valid": 0})
    
    for item in items:
        subject = item.get("subject", "Unknown")
        by_subject[subject]["total"] += 1
        if item.get("pred") != "NONE":
            by_subject[subject]["valid"] += 1
        if item.get("correct", False):
            by_subject[subject]["correct"] += 1
    
    # Calculate accuracy for each subject
    result = {}
    for subject, stats in by_subject.items():
        result[subject] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "valid": stats["valid"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
        }
    
    return result


def compare_basic_stats(data1: Dict, data2: Dict, name1: str, name2: str) -> str:
    """Generate basic statistics comparison."""
    report = []
    report.append("=" * 80)
    report.append("БАЗОВАЯ СТАТИСТИКА МОДЕЛЕЙ")
    report.append("=" * 80)
    
    # Model info
    report.append(f"\n{'Параметр':<40} {name1:<20} {name2:<20}")
    report.append("-" * 80)
    report.append(f"{'Модель':<40} {data1['model'][:18]:<20} {data2['model'][:18]:<20}")
    report.append(f"{'Benchmark':<40} {data1['benchmark_path'].split('/')[-1][:18]:<20} {data2['benchmark_path'].split('/')[-1][:18]:<20}")
    report.append(f"{'Количество слоёв':<40} {data1['num_layers']:<20} {data2['num_layers']:<20}")
    report.append(f"{'Размер intermediate':<40} {data1['intermediate_size']:<20} {data2['intermediate_size']:<20}")
    report.append(f"{'Всего нейронов':<40} {data1['neuron_totals']['total_neurons']:<20,} {data2['neuron_totals']['total_neurons']:<20,}")
    
    # Token processing
    report.append(f"\n{'ТОКЕНЫ ОБРАБОТАНО':<40}")
    report.append("-" * 80)
    t1 = data1['tokens_processed_prompt']
    t2 = data2['tokens_processed_prompt']
    report.append(f"{'Токенов в промптах':<40} {t1:<20,} {t2:<20,}")
    
    # Accuracy
    report.append(f"\n{'ТОЧНОСТЬ (ACCURACY)':<40}")
    report.append("-" * 80)
    acc1 = data1['accuracy']
    acc2 = data2['accuracy']
    
    report.append(f"{'Правильных ответов':<40} {acc1['correct']:<20,} {acc2['correct']:<20,}")
    report.append(f"{'Всего вопросов':<40} {acc1['total']:<20,} {acc2['total']:<20,}")
    report.append(f"{'Валидных предсказаний':<40} {acc1['valid_predictions']:<20,} {acc2['valid_predictions']:<20,}")
    report.append(f"{'Accuracy':<40} {acc1['accuracy']:<20.4f} {acc2['accuracy']:<20.4f}")
    report.append(f"{'Accuracy %':<40} {acc1['accuracy']*100:<20.2f}% {acc2['accuracy']*100:<20.2f}%")
    
    diff = acc1['accuracy'] - acc2['accuracy']
    winner = name1 if diff > 0 else (name2 if diff < 0 else "Равно")
    report.append(f"\n{'Разница (M1 - M2):':<40} {diff:+.4f} ({diff*100:+.2f}%)")
    report.append(f"{'Лучшая модель:':<40} {winner}")
    
    # Neuron activation
    report.append(f"\n{'АКТИВАЦИЯ НЕЙРОНОВ':<40}")
    report.append("-" * 80)
    n1 = data1['neuron_totals']
    n2 = data2['neuron_totals']
    
    report.append(f"{'Активных нейронов (хотя бы раз)':<40} {n1['active_any']:<20,} {n2['active_any']:<20,}")
    report.append(f"{'Отобрано языко-специфичных':<40} {n1['selected_language_specific']:<20,} {n2['selected_language_specific']:<20,}")
    
    pct1_active = 100 * n1['active_any'] / n1['total_neurons']
    pct2_active = 100 * n2['active_any'] / n2['total_neurons']
    pct1_specific = 100 * n1['selected_language_specific'] / n1['total_neurons']
    pct2_specific = 100 * n2['selected_language_specific'] / n2['total_neurons']
    
    report.append(f"{'% активных':<40} {pct1_active:<20.2f}% {pct2_active:<20.2f}%")
    report.append(f"{'% языко-специфичных':<40} {pct1_specific:<20.3f}% {pct2_specific:<20.3f}%")
    
    return "\n".join(report)


def compare_accuracy_by_subject(data1: Dict, data2: Dict, name1: str, name2: str) -> Tuple[str, Dict, Dict]:
    """Compare accuracy by subject."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("ТОЧНОСТЬ ПО КАТЕГОРИЯМ (SUBJECTS)")
    report.append("=" * 80)
    
    subjects1 = analyze_by_subject(data1['items'])
    subjects2 = analyze_by_subject(data2['items'])
    
    all_subjects = sorted(set(subjects1.keys()) | set(subjects2.keys()))
    
    report.append(f"\n{'Категория':<30} {'Model 1 Acc':<15} {'Model 2 Acc':<15} {'Разница':<15}")
    report.append("-" * 80)
    
    for subject in all_subjects:
        acc1 = subjects1.get(subject, {}).get('accuracy', 0.0)
        acc2 = subjects2.get(subject, {}).get('accuracy', 0.0)
        diff = acc1 - acc2
        
        total1 = subjects1.get(subject, {}).get('total', 0)
        total2 = subjects2.get(subject, {}).get('total', 0)
        
        report.append(f"{subject:<30} {acc1*100:<7.2f}% ({total1:<3}) {acc2*100:<7.2f}% ({total2:<3}) {diff*100:+7.2f}%")
    
    # Summary statistics
    report.append(f"\n{'СТАТИСТИКА ПО КАТЕГОРИЯМ':<40}")
    report.append("-" * 80)
    
    accs1 = [s['accuracy'] for s in subjects1.values()]
    accs2 = [s['accuracy'] for s in subjects2.values()]
    
    if accs1:
        report.append(f"{name1}:")
        report.append(f"  Среднее: {np.mean(accs1):.4f}, Медиана: {np.median(accs1):.4f}")
        report.append(f"  Мин: {np.min(accs1):.4f}, Макс: {np.max(accs1):.4f}, Стд: {np.std(accs1):.4f}")
    
    if accs2:
        report.append(f"\n{name2}:")
        report.append(f"  Среднее: {np.mean(accs2):.4f}, Медиана: {np.median(accs2):.4f}")
        report.append(f"  Мин: {np.min(accs2):.4f}, Макс: {np.max(accs2):.4f}, Стд: {np.std(accs2):.4f}")
    
    # Best and worst categories for each model
    report.append(f"\nЛУЧШИЕ КАТЕГОРИИ:")
    if subjects1:
        sorted1 = sorted(subjects1.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:5]
        report.append(f"\n{name1}:")
        for subj, stats in sorted1:
            report.append(f"  {subj}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")
    
    if subjects2:
        sorted2 = sorted(subjects2.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:5]
        report.append(f"\n{name2}:")
        for subj, stats in sorted2:
            report.append(f"  {subj}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")
    
    report.append(f"\nХУДШИЕ КАТЕГОРИИ:")
    if subjects1:
        sorted1 = sorted(subjects1.items(), key=lambda x: x[1]['accuracy'])[:5]
        report.append(f"\n{name1}:")
        for subj, stats in sorted1:
            report.append(f"  {subj}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")
    
    if subjects2:
        sorted2 = sorted(subjects2.items(), key=lambda x: x[1]['accuracy'])[:5]
        report.append(f"\n{name2}:")
        for subj, stats in sorted2:
            report.append(f"  {subj}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")
    
    return "\n".join(report), subjects1, subjects2


def analyze_layer_distribution(data1: Dict, data2: Dict, name1: str, name2: str) -> Tuple[str, np.ndarray, np.ndarray]:
    """Analyze distribution of language-specific neurons across layers."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("РАСПРЕДЕЛЕНИЕ ЯЗЫКО-СПЕЦИФИЧНЫХ НЕЙРОНОВ ПО СЛОЯМ")
    report.append("=" * 80)
    
    masks1 = data1['layerwise_indices']['language_specific_mask']
    masks2 = data2['layerwise_indices']['language_specific_mask']
    
    num_layers1 = data1['num_layers']
    num_layers2 = data2['num_layers']
    
    # Count per layer
    layer_counts1 = [len(masks1[i]) for i in range(num_layers1)]
    layer_counts2 = [len(masks2[i]) for i in range(num_layers2)]
    
    # Summary statistics
    report.append(f"\n{name1}:")
    report.append(f"  Мин: {min(layer_counts1)}, Макс: {max(layer_counts1)}, "
                 f"Среднее: {np.mean(layer_counts1):.1f}, Медиана: {np.median(layer_counts1):.1f}")
    
    report.append(f"\n{name2}:")
    report.append(f"  Мин: {min(layer_counts2)}, Макс: {max(layer_counts2)}, "
                 f"Среднее: {np.mean(layer_counts2):.1f}, Медиана: {np.median(layer_counts2):.1f}")
    
    # Find layers with most differences
    min_layers = min(num_layers1, num_layers2)
    diff = np.array(layer_counts1[:min_layers]) - np.array(layer_counts2[:min_layers])
    top_diff_idx = np.argsort(np.abs(diff))[-5:][::-1]
    
    report.append(f"\nСлои с наибольшими различиями:")
    for idx in top_diff_idx:
        report.append(f"  Слой {idx}: {name1}={layer_counts1[idx]}, {name2}={layer_counts2[idx]}, разница={diff[idx]:+d}")
    
    # Prepare data for plotting
    data_array1 = np.array(layer_counts1)
    data_array2 = np.array(layer_counts2)
    
    return "\n".join(report), data_array1, data_array2


def calculate_overlap(data1: Dict, data2: Dict, name1: str, name2: str) -> Tuple[Dict, str]:
    """Calculate overlap of language-specific neurons between models."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("ПЕРЕСЕЧЕНИЕ ЯЗЫКО-СПЕЦИФИЧНЫХ НЕЙРОНОВ")
    report.append("=" * 80)
    
    masks1 = data1['layerwise_indices']['language_specific_mask']
    masks2 = data2['layerwise_indices']['language_specific_mask']
    
    num_layers = min(data1['num_layers'], data2['num_layers'])
    
    # Build sets of (layer, neuron) tuples
    set1 = set()
    set2 = set()
    
    for layer in range(num_layers):
        for neuron in masks1[layer]:
            set1.add((layer, neuron))
        for neuron in masks2[layer]:
            set2.add((layer, neuron))
    
    # Calculate overlaps
    overlap = set1 & set2
    only_model1 = set1 - set2
    only_model2 = set2 - set1
    
    # Jaccard similarity
    union = set1 | set2
    jaccard = len(overlap) / len(union) if len(union) > 0 else 0.0
    
    report.append(f"\n{name1}: {len(set1):,} языко-специфичных нейронов")
    report.append(f"{name2}: {len(set2):,} языко-специфичных нейронов")
    report.append(f"Пересечение: {len(overlap):,} нейронов ({(100*len(overlap)/len(set1)) if len(set1) else 0:.2f}% от {name1})")
    report.append(f"Только в {name1}: {len(only_model1):,} нейронов")
    report.append(f"Только в {name2}: {len(only_model2):,} нейронов")
    report.append(f"Jaccard similarity: {jaccard:.4f}")
    
    overlap_data = {
        'overlap': len(overlap),
        'only_model1': len(only_model1),
        'only_model2': len(only_model2),
        'jaccard': jaccard
    }
    
    return overlap_data, "\n".join(report)


def create_visualizations(data1: Dict, data2: Dict, name1: str, name2: str,
                          layer_dist1: np.ndarray, layer_dist2: np.ndarray,
                          subjects1: Dict, subjects2: Dict,
                          overlap_data: Dict, output_prefix: str):
    """Create comprehensive visualizations."""
    # ------- Большое полотно: 3x3 -------
    fig = plt.figure(figsize=(20, 14))
    
    # 1. Overall accuracy comparison
    ax1 = plt.subplot(3, 3, 1)
    models = [name1[:15], name2[:15]]
    accuracies = [data1['accuracy']['accuracy'], data2['accuracy']['accuracy']]
    colors = ['#2E86AB', '#F18F01']
    bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Общая точность моделей')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 2. Accuracy by subject comparison
    ax2 = plt.subplot(3, 3, 2)
    all_subjects = sorted(set(subjects1.keys()) | set(subjects2.keys()))
    x = np.arange(len(all_subjects))
    width = 0.35
    
    accs1 = [subjects1.get(s, {}).get('accuracy', 0) for s in all_subjects]
    accs2 = [subjects2.get(s, {}).get('accuracy', 0) for s in all_subjects]
    
    ax2.bar(x - width/2, accs1, width, label=name1[:15], alpha=0.8, color='#2E86AB')
    ax2.bar(x + width/2, accs2, width, label=name2[:15], alpha=0.8, color='#F18F01')
    ax2.set_xlabel('Категория')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Точность по категориям')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_subjects, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Accuracy difference by subject
    ax3 = plt.subplot(3, 3, 3)
    diffs = [accs1[i] - accs2[i] for i in range(len(all_subjects))]
    colors_diff = ['#06A77D' if d > 0 else '#D62828' for d in diffs]
    ax3.bar(all_subjects, diffs, color=colors_diff, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_xlabel('Категория')
    ax3.set_ylabel(f'Разница (M1 - M2)')
    ax3.set_title('Разница в точности по категориям')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Layer distribution - Model 1
    ax4 = plt.subplot(3, 3, 4)
    layers1 = np.arange(data1['num_layers'])
    ax4.bar(layers1, layer_dist1, alpha=0.8, color='#2E86AB', edgecolor='black')
    ax4.set_xlabel('Слой')
    ax4.set_ylabel('Количество нейронов')
    ax4.set_title(f'{name1[:25]}: Распределение по слоям')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Layer distribution - Model 2
    ax5 = plt.subplot(3, 3, 5)
    layers2 = np.arange(data2['num_layers'])
    ax5.bar(layers2, layer_dist2, alpha=0.8, color='#F18F01', edgecolor='black')
    ax5.set_xlabel('Слой')
    ax5.set_ylabel('Количество нейронов')
    ax5.set_title(f'{name2[:25]}: Распределение по слоям')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Direct comparison of layer distributions
    ax6 = plt.subplot(3, 3, 6)
    min_layers = min(len(layer_dist1), len(layer_dist2))
    x = np.arange(min_layers)
    ax6.plot(x, layer_dist1[:min_layers], 'o-', label=name1[:15], linewidth=2, markersize=4, color='#2E86AB')
    ax6.plot(x, layer_dist2[:min_layers], 's-', label=name2[:15], linewidth=2, markersize=4, color='#F18F01')
    ax6.set_xlabel('Слой')
    ax6.set_ylabel('Количество нейронов')
    ax6.set_title('Сравнение моделей по слоям')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 7. Neuron counts comparison
    ax7 = plt.subplot(3, 3, 7)
    categories = ['Активных\nнейронов', 'Языко-\nспецифичных']
    model1_vals = [
        data1['neuron_totals']['active_any'],
        data1['neuron_totals']['selected_language_specific']
    ]
    model2_vals = [
        data2['neuron_totals']['active_any'],
        data2['neuron_totals']['selected_language_specific']
    ]
    x = np.arange(len(categories))
    width = 0.35
    ax7.bar(x - width/2, model1_vals, width, label=name1[:15], alpha=0.8, color='#2E86AB')
    ax7.bar(x + width/2, model2_vals, width, label=name2[:15], alpha=0.8, color='#F18F01')
    ax7.set_ylabel('Количество нейронов')
    ax7.set_title('Сравнение нейронов')
    ax7.set_xticks(x)
    ax7.set_xticklabels(categories)
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)
    
    # 8. Overlap visualization
    ax8 = plt.subplot(3, 3, 8)
    overlap_cats = ['Пересечение', f'Только\n{name1[:10]}', f'Только\n{name2[:10]}']
    overlap_vals = [
        overlap_data['overlap'],
        overlap_data['only_model1'],
        overlap_data['only_model2']
    ]
    colors_overlap = ['#06A77D', '#2E86AB', '#F18F01']
    bars = ax8.bar(overlap_cats, overlap_vals, color=colors_overlap, alpha=0.8, edgecolor='black')
    ax8.set_ylabel('Количество нейронов')
    ax8.set_title(f'Пересечение нейронов\n(Jaccard: {overlap_data["jaccard"]:.3f})')
    ax8.grid(axis='y', alpha=0.3)
    for i, v in enumerate(overlap_vals):
        ax8.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    
    # 9. Correct/Total comparison
    ax9 = plt.subplot(3, 3, 9)
    categories = ['Правильных', 'Всего вопросов', 'Валидных\nпредсказаний']
    model1_vals = [
        data1['accuracy']['correct'],
        data1['accuracy']['total'],
        data1['accuracy']['valid_predictions']
    ]
    model2_vals = [
        data2['accuracy']['correct'],
        data2['accuracy']['total'],
        data2['accuracy']['valid_predictions']
    ]
    x = np.arange(len(categories))
    ax9.bar(x - width/2, model1_vals, width, label=name1[:15], alpha=0.8, color='#2E86AB')
    ax9.bar(x + width/2, model2_vals, width, label=name2[:15], alpha=0.8, color='#F18F01')
    ax9.set_ylabel('Количество')
    ax9.set_title('Сравнение ответов')
    ax9.set_xticks(x)
    ax9.set_xticklabels(categories)
    ax9.legend()
    ax9.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Сохранено: {output_prefix}_comparison.png")
    plt.close()
    
    # ------- Детальное полотно 2x2 -------
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Difference plot by layers
    num_layers = min(data1['num_layers'], data2['num_layers'])
    diff_layers = layer_dist1[:num_layers] - layer_dist2[:num_layers]
    
    axes[0, 0].bar(range(num_layers), diff_layers, 
                   color=['#06A77D' if x >= 0 else '#D62828' for x in diff_layers], alpha=0.7)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0, 0].set_xlabel('Слой')
    axes[0, 0].set_ylabel(f'Разница ({name1[:10]} - {name2[:10]})')
    axes[0, 0].set_title('Разница в языко-специфичных нейронах по слоям')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Cumulative distribution
    axes[0, 1].plot(range(num_layers), np.cumsum(layer_dist1[:num_layers]), 
                    'o-', label=name1[:15], linewidth=2, color='#2E86AB')
    axes[0, 1].plot(range(num_layers), np.cumsum(layer_dist2[:num_layers]), 
                    's-', label=name2[:15], linewidth=2, color='#F18F01')
    axes[0, 1].set_xlabel('Слой')
    axes[0, 1].set_ylabel('Кумулятивное количество')
    axes[0, 1].set_title('Кумулятивное распределение по слоям')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Percentage by layer
    pct_model1 = 100 * layer_dist1[:num_layers] / data1['intermediate_size']
    pct_model2 = 100 * layer_dist2[:num_layers] / data2['intermediate_size']
    
    axes[1, 0].plot(range(num_layers), pct_model1, 'o-', label=name1[:15], 
                    linewidth=2, markersize=4, color='#2E86AB')
    axes[1, 0].plot(range(num_layers), pct_model2, 's-', label=name2[:15], 
                    linewidth=2, markersize=4, color='#F18F01')
    axes[1, 0].set_xlabel('Слой')
    axes[1, 0].set_ylabel('% от intermediate_size')
    axes[1, 0].set_title('Процент языко-специфичных нейронов по слоям')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Heatmap comparison
    axes[1, 1].bar(['Model 1', 'Model 2'], 
                   [data1['accuracy']['accuracy'], data2['accuracy']['accuracy']], 
                   color=['#2E86AB', '#F18F01'], alpha=0.8, edgecolor='black')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Общая точность (сводка)')
    axes[1, 1].set_ylim(0, 1.0)
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate([data1['accuracy']['accuracy'], data2['accuracy']['accuracy']]):
        axes[1, 1].text(i, v, f'{v*100:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_detailed.png', dpi=150, bbox_inches='tight')
    print(f"Сохранено: {output_prefix}_detailed.png")
    plt.close()


def main():
    print("=" * 80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ НА TUMLU BENCHMARK")
    print("=" * 80)
    
    # Load data
    data1 = load_json(JSON1_PATH)
    data2 = load_json(JSON2_PATH)
    
    print(f"Модель 1: {MODEL1_NAME}")
    print(f"Модель 2: {MODEL2_NAME}")
    print(f"Язык: {LANGUAGE}")
    
    # Generate reports
    report_parts = []
    
    # Basic statistics
    report_parts.append(compare_basic_stats(data1, data2, MODEL1_NAME, MODEL2_NAME))
    
    # Accuracy by subject
    subject_report, subjects1, subjects2 = compare_accuracy_by_subject(data1, data2, MODEL1_NAME, MODEL2_NAME)
    report_parts.append(subject_report)
    
    # Layer distribution
    layer_report, layer_dist1, layer_dist2 = analyze_layer_distribution(data1, data2, MODEL1_NAME, MODEL2_NAME)
    report_parts.append(layer_report)
    
    # Overlap analysis
    overlap_data, overlap_report = calculate_overlap(data1, data2, MODEL1_NAME, MODEL2_NAME)
    report_parts.append(overlap_report)
    
    # Combine all reports
    full_report = "\n".join(report_parts)
    
    # Print to console
    print(full_report)
    
    # Save to file
    report_file = f"{OUTPUT_PREFIX}_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(full_report)
    print(f"\n{'-'*80}")
    print(f"Отчёт сохранён: {report_file}")
    
    # Generate visualizations
    print(f"\nГенерация визуализаций...")
    create_visualizations(data1, data2, MODEL1_NAME, MODEL2_NAME,
                          layer_dist1, layer_dist2, subjects1, subjects2,
                          overlap_data, OUTPUT_PREFIX)
    
    # Save summary JSON
    summary = {
        "models": {
            "model1": MODEL1_NAME,
            "model2": MODEL2_NAME,
        },
        "language": LANGUAGE,
        "accuracy": {
            "model1": data1['accuracy']['accuracy'],
            "model2": data2['accuracy']['accuracy'],
            "difference": data1['accuracy']['accuracy'] - data2['accuracy']['accuracy'],
            "winner": MODEL1_NAME if data1['accuracy']['accuracy'] > data2['accuracy']['accuracy'] else MODEL2_NAME,
        },
        "by_subject": {
            "model1": subjects1,
            "model2": subjects2,
        },
        "neurons": {
            "model1": {
                "active": data1['neuron_totals']['active_any'],
                "specific": data1['neuron_totals']['selected_language_specific'],
            },
            "model2": {
                "active": data2['neuron_totals']['active_any'],
                "specific": data2['neuron_totals']['selected_language_specific'],
            },
            "overlap": overlap_data,
        }
    }
    
    summary_file = f"{OUTPUT_PREFIX}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Сводка сохранена: {summary_file}")
    
    print(f"\n{'='*80}")
    print("ГОТОВО!")
    print(f"{'='*80}")
    print("Файлы созданы:")
    print(f"  - {report_file}")
    print(f"  - {OUTPUT_PREFIX}_comparison.png")
    print(f"  - {OUTPUT_PREFIX}_detailed.png")
    print(f"  - {summary_file}")


if __name__ == "__main__":
    main()

