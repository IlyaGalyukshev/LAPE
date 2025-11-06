import argparse
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    print(f"Загрузка {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_basic_stats(data1: Dict, data2: Dict, name1: str, name2: str, lang1_name: str, lang2_name: str) -> str:
    """Generate basic statistics comparison."""
    report = []
    report.append("=" * 80)
    report.append("БАЗОВАЯ СТАТИСТИКА МОДЕЛЕЙ")
    report.append("=" * 80)
    
    # Model info
    report.append(f"\n{'Параметр':<40} {name1:<20} {name2:<20}")
    report.append("-" * 80)
    report.append(f"{'Модель':<40} {data1['model'][:18]:<20} {data2['model'][:18]:<20}")
    report.append(f"{'Количество слоёв':<40} {data1['num_layers']:<20} {data2['num_layers']:<20}")
    report.append(f"{'Размер intermediate':<40} {data1['intermediate_size']:<20} {data2['intermediate_size']:<20}")
    report.append(f"{'Всего нейронов':<40} {data1['neuron_totals']['total_neurons']:<20} {data2['neuron_totals']['total_neurons']:<20}")
    
    # Token processing
    report.append(f"\n{'ТОКЕНЫ ОБРАБОТАНО':<40}")
    report.append("-" * 80)
    t1_lang1 = data1['tokens_processed']['lang1']
    t1_lang2 = data1['tokens_processed']['lang2']
    t2_lang1 = data2['tokens_processed']['lang1']
    t2_lang2 = data2['tokens_processed']['lang2']
    
    report.append(f"{'Lang1 (' + lang1_name + ')':<40} {t1_lang1:<20,} {t2_lang1:<20,}")
    report.append(f"{'Lang2 (' + lang2_name + ')':<40} {t1_lang2:<20,} {t2_lang2:<20,}")
    report.append(f"{'Всего':<40} {t1_lang1+t1_lang2:<20,} {t2_lang1+t2_lang2:<20,}")
    
    # Neuron activation
    report.append(f"\n{'АКТИВАЦИЯ НЕЙРОНОВ':<40}")
    report.append("-" * 80)
    n1 = data1['neuron_totals']
    n2 = data2['neuron_totals']
    
    report.append(f"{lang1_name + ' активных (хотя бы раз)':<40} {n1['lang1_active_any']:<20,} {n2['lang1_active_any']:<20,}")
    report.append(f"{lang2_name + ' активных (хотя бы раз)':<40} {n1['lang2_active_any']:<20,} {n2['lang2_active_any']:<20,}")
    report.append(f"{'Оба языка':<40} {n1['both_active']:<20,} {n2['both_active']:<20,}")
    report.append(f"{'Только ' + lang1_name:<40} {n1['only_lang1']:<20,} {n2['only_lang1']:<20,}")
    report.append(f"{'Только ' + lang2_name:<40} {n1['only_lang2']:<20,} {n2['only_lang2']:<20,}")
    
    # Percentage
    pct1_both = 100 * n1['both_active'] / n1['total_neurons']
    pct2_both = 100 * n2['both_active'] / n2['total_neurons']
    report.append(f"{'% активных в обоих языках':<40} {pct1_both:<20.2f}% {pct2_both:<20.2f}%")
    
    return "\n".join(report)


def compare_language_specific_neurons(data1: Dict, data2: Dict, name1: str, name2: str, lang1_name: str, lang2_name: str) -> str:
    """Compare language-specific neurons identified by LAPE method."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("ЯЗЫКО-СПЕЦИФИЧНЫЕ НЕЙРОНЫ (LAPE метод)")
    report.append("=" * 80)
    
    masks1 = data1['language_specific_masks']
    masks2 = data2['language_specific_masks']
    
    # Count neurons per language
    count1_lang1 = sum(len(layer) for layer in masks1['lang1'])
    count1_lang2 = sum(len(layer) for layer in masks1['lang2'])
    count2_lang1 = sum(len(layer) for layer in masks2['lang1'])
    count2_lang2 = sum(len(layer) for layer in masks2['lang2'])
    
    total1 = count1_lang1 + count1_lang2
    total2 = count2_lang1 + count2_lang2
    
    report.append(f"\n{'Метрика':<40} {name1:<20} {name2:<20}")
    report.append("-" * 80)
    report.append(f"{lang1_name + '-специфичные нейроны':<40} {count1_lang1:<20,} {count2_lang1:<20,}")
    report.append(f"{lang2_name + '-специфичные нейроны':<40} {count1_lang2:<20,} {count2_lang2:<20,}")
    report.append(f"{'Всего языко-специфичных':<40} {total1:<20,} {total2:<20,}")
    
    # Percentage of total neurons
    pct1_lang1 = 100 * count1_lang1 / data1['neuron_totals']['total_neurons']
    pct2_lang1 = 100 * count2_lang1 / data2['neuron_totals']['total_neurons']
    pct1_lang2 = 100 * count1_lang2 / data1['neuron_totals']['total_neurons']
    pct2_lang2 = 100 * count2_lang2 / data2['neuron_totals']['total_neurons']
    
    report.append(f"\n{'% от всех нейронов:':<40}")
    report.append(f"{'  ' + lang1_name + '-специфичные':<40} {pct1_lang1:<20.3f}% {pct2_lang1:<20.3f}%")
    report.append(f"{'  ' + lang2_name + '-специфичные':<40} {pct1_lang2:<20.3f}% {pct2_lang2:<20.3f}%")
    report.append(f"{'  Всего специфичных':<40} {100*total1/data1['neuron_totals']['total_neurons']:<20.3f}% {100*total2/data2['neuron_totals']['total_neurons']:<20.3f}%")
    
    # Thresholds used
    report.append(f"\n{'ПОРОГИ ОТБОРА':<40}")
    report.append("-" * 80)
    t1 = data1['thresholds']
    t2 = data2['thresholds']
    report.append(f"{'Filter rate':<40} {t1['filter_rate']:<20} {t2['filter_rate']:<20}")
    report.append(f"{'Top rate (энтропия)':<40} {t1['top_rate']:<20} {t2['top_rate']:<20}")
    report.append(f"{'Activation bar ratio':<40} {t1['activation_bar_ratio']:<20} {t2['activation_bar_ratio']:<20}")
    report.append(f"{'Activation bar value':<40} {t1['activation_bar_value']:<20.4f} {t2['activation_bar_value']:<20.4f}")
    
    return "\n".join(report)


def analyze_layer_distribution(data1: Dict, data2: Dict, name1: str, name2: str, lang1_name: str, lang2_name: str) -> Tuple[str, np.ndarray, np.ndarray]:
    """Analyze distribution of language-specific neurons across layers."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("РАСПРЕДЕЛЕНИЕ ПО СЛОЯМ")
    report.append("=" * 80)
    
    masks1 = data1['language_specific_masks']
    masks2 = data2['language_specific_masks']
    
    num_layers1 = data1['num_layers']
    num_layers2 = data2['num_layers']
    
    # Count per layer for model 1
    layer_counts1_lang1 = [len(masks1['lang1'][i]) for i in range(num_layers1)]
    layer_counts1_lang2 = [len(masks1['lang2'][i]) for i in range(num_layers1)]
    
    # Count per layer for model 2
    layer_counts2_lang1 = [len(masks2['lang1'][i]) for i in range(num_layers2)]
    layer_counts2_lang2 = [len(masks2['lang2'][i]) for i in range(num_layers2)]
    
    # Summary statistics
    report.append(f"\n{name1}:")
    report.append(f"  {lang1_name} - Мин: {min(layer_counts1_lang1)}, Макс: {max(layer_counts1_lang1)}, "
                 f"Среднее: {np.mean(layer_counts1_lang1):.1f}, Медиана: {np.median(layer_counts1_lang1):.1f}")
    report.append(f"  {lang2_name} - Мин: {min(layer_counts1_lang2)}, Макс: {max(layer_counts1_lang2)}, "
                 f"Среднее: {np.mean(layer_counts1_lang2):.1f}, Медиана: {np.median(layer_counts1_lang2):.1f}")
    
    report.append(f"\n{name2}:")
    report.append(f"  {lang1_name} - Мин: {min(layer_counts2_lang1)}, Макс: {max(layer_counts2_lang1)}, "
                 f"Среднее: {np.mean(layer_counts2_lang1):.1f}, Медиана: {np.median(layer_counts2_lang1):.1f}")
    report.append(f"  {lang2_name} - Мин: {min(layer_counts2_lang2)}, Макс: {max(layer_counts2_lang2)}, "
                 f"Среднее: {np.mean(layer_counts2_lang2):.1f}, Медиана: {np.median(layer_counts2_lang2):.1f}")
    
    # Find layers with most differences
    report.append(f"\nСлои с наибольшими различиями ({lang1_name}-специфичные):")
    diff_lang1 = np.array(layer_counts1_lang1) - np.array(layer_counts2_lang1)
    top_diff_idx = np.argsort(np.abs(diff_lang1))[-5:][::-1]
    for idx in top_diff_idx:
        report.append(f"  Слой {idx}: {name1}={layer_counts1_lang1[idx]}, {name2}={layer_counts2_lang1[idx]}, разница={diff_lang1[idx]}")
    
    # Prepare data for plotting
    data_array1 = np.array([layer_counts1_lang1, layer_counts1_lang2])
    data_array2 = np.array([layer_counts2_lang1, layer_counts2_lang2])
    
    return "\n".join(report), data_array1, data_array2


def analyze_top_neurons(data1: Dict, data2: Dict, name1: str, name2: str, lang1_name: str, lang2_name: str) -> str:
    """Analyze top language-specific neurons by rate difference."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("ТОП НЕЙРОНОВ ПО РАЗНИЦЕ АКТИВАЦИИ")
    report.append("=" * 80)
    
    top1 = data1.get('top_lang1_specific_by_rate_diff', [])[:20]
    top2 = data2.get('top_lang1_specific_by_rate_diff', [])[:20]
    
    report.append(f"\nТоп 10 для {name1}:")
    report.append(f"{'Слой':<8} {'Нейрон':<10} {lang1_name + ' rate':<15} {lang2_name + ' rate':<15} {'Разница':<15}")
    report.append("-" * 70)
    for i, neuron in enumerate(top1[:10]):
        report.append(f"{neuron['layer']:<8} {neuron['neuron']:<10} "
                     f"{neuron['lang1_rate']:<15.4f} {neuron['lang2_rate']:<15.4f} "
                     f"{neuron['rate_diff']:<15.4f}")
    
    report.append(f"\nТоп 10 для {name2}:")
    report.append(f"{'Слой':<8} {'Нейрон':<10} {lang1_name + ' rate':<15} {lang2_name + ' rate':<15} {'Разница':<15}")
    report.append("-" * 70)
    for i, neuron in enumerate(top2[:10]):
        report.append(f"{neuron['layer']:<8} {neuron['neuron']:<10} "
                     f"{neuron['lang1_rate']:<15.4f} {neuron['lang2_rate']:<15.4f} "
                     f"{neuron['rate_diff']:<15.4f}")
    
    # Compare statistics of rate differences
    diffs1 = [n['rate_diff'] for n in top1]
    diffs2 = [n['rate_diff'] for n in top2]
    
    report.append(f"\nСтатистика разниц активаций (топ-20):")
    report.append(f"{'Модель':<20} {'Макс':<15} {'Среднее':<15} {'Медиана':<15}")
    report.append("-" * 70)
    report.append(f"{name1:<20} {max(diffs1):<15.4f} {np.mean(diffs1):<15.4f} {np.median(diffs1):<15.4f}")
    report.append(f"{name2:<20} {max(diffs2):<15.4f} {np.mean(diffs2):<15.4f} {np.median(diffs2):<15.4f}")
    
    return "\n".join(report)


def calculate_overlap(data1: Dict, data2: Dict, name1: str, name2: str, lang1_name: str, lang2_name: str) -> Tuple[Dict, str]:
    """Calculate overlap of language-specific neurons between models."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("ПЕРЕСЕЧЕНИЕ ЯЗЫКО-СПЕЦИФИЧНЫХ НЕЙРОНОВ")
    report.append("=" * 80)
    
    masks1 = data1['language_specific_masks']
    masks2 = data2['language_specific_masks']
    
    num_layers = min(data1['num_layers'], data2['num_layers'])
    
    # Build sets of (layer, neuron) tuples
    set1_lang1 = set()
    set1_lang2 = set()
    set2_lang1 = set()
    set2_lang2 = set()
    
    for layer in range(num_layers):
        for neuron in masks1['lang1'][layer]:
            set1_lang1.add((layer, neuron))
        for neuron in masks1['lang2'][layer]:
            set1_lang2.add((layer, neuron))
        for neuron in masks2['lang1'][layer]:
            set2_lang1.add((layer, neuron))
        for neuron in masks2['lang2'][layer]:
            set2_lang2.add((layer, neuron))
    
    # Calculate overlaps
    overlap_lang1 = set1_lang1 & set2_lang1
    overlap_lang2 = set1_lang2 & set2_lang2
    
    only_model1_lang1 = set1_lang1 - set2_lang1
    only_model2_lang1 = set2_lang1 - set1_lang1
    only_model1_lang2 = set1_lang2 - set2_lang2
    only_model2_lang2 = set2_lang2 - set1_lang2
    
    # Jaccard similarity
    jaccard_lang1 = len(overlap_lang1) / len(set1_lang1 | set2_lang1) if len(set1_lang1 | set2_lang1) > 0 else 0
    jaccard_lang2 = len(overlap_lang2) / len(set1_lang2 | set2_lang2) if len(set1_lang2 | set2_lang2) > 0 else 0
    
    report.append(f"\n{lang1_name}-специфичные нейроны:")
    report.append(f"  {name1}: {len(set1_lang1):,} нейронов")
    report.append(f"  {name2}: {len(set2_lang1):,} нейронов")
    report.append(f"  Пересечение: {len(overlap_lang1):,} нейронов ({100*len(overlap_lang1)/len(set1_lang1):.2f}% от {name1})")
    report.append(f"  Только в {name1}: {len(only_model1_lang1):,} нейронов")
    report.append(f"  Только в {name2}: {len(only_model2_lang1):,} нейронов")
    report.append(f"  Jaccard similarity: {jaccard_lang1:.4f}")
    
    report.append(f"\n{lang2_name}-специфичные нейроны:")
    report.append(f"  {name1}: {len(set1_lang2):,} нейронов")
    report.append(f"  {name2}: {len(set2_lang2):,} нейронов")
    report.append(f"  Пересечение: {len(overlap_lang2):,} нейронов ({100*len(overlap_lang2)/len(set1_lang2):.2f}% от {name1})")
    report.append(f"  Только в {name1}: {len(only_model1_lang2):,} нейронов")
    report.append(f"  Только в {name2}: {len(only_model2_lang2):,} нейронов")
    report.append(f"  Jaccard similarity: {jaccard_lang2:.4f}")
    
    overlap_data = {
        'lang1': {
            'overlap': len(overlap_lang1),
            'only_model1': len(only_model1_lang1),
            'only_model2': len(only_model2_lang1),
            'jaccard': jaccard_lang1
        },
        'lang2': {
            'overlap': len(overlap_lang2),
            'only_model1': len(only_model1_lang2),
            'only_model2': len(only_model2_lang2),
            'jaccard': jaccard_lang2
        }
    }
    
    return overlap_data, "\n".join(report)


def create_visualizations(data1: Dict, data2: Dict, name1: str, name2: str, 
                          layer_dist1: np.ndarray, layer_dist2: np.ndarray,
                          overlap_data: Dict, output_prefix: str, lang1_name: str, lang2_name: str):
    """Create comprehensive visualizations."""
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Layer distribution comparison - Model 1
    ax1 = plt.subplot(3, 3, 1)
    layers1 = np.arange(data1['num_layers'])
    width = 0.35
    ax1.bar(layers1 - width/2, layer_dist1[0], width, label=f'Lang1 ({lang1_name})', alpha=0.8, color='#2E86AB')
    ax1.bar(layers1 + width/2, layer_dist1[1], width, label=f'Lang2 ({lang2_name})', alpha=0.8, color='#A23B72')
    ax1.set_xlabel('Слой')
    ax1.set_ylabel('Количество нейронов')
    ax1.set_title(f'{name1}: Распределение по слоям')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Layer distribution comparison - Model 2
    ax2 = plt.subplot(3, 3, 2)
    layers2 = np.arange(data2['num_layers'])
    ax2.bar(layers2 - width/2, layer_dist2[0], width, label=f'Lang1 ({lang1_name})', alpha=0.8, color='#2E86AB')
    ax2.bar(layers2 + width/2, layer_dist2[1], width, label=f'Lang2 ({lang2_name})', alpha=0.8, color='#A23B72')
    ax2.set_xlabel('Слой')
    ax2.set_ylabel('Количество нейронов')
    ax2.set_title(f'{name2}: Распределение по слоям')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Direct comparison of layer distributions
    ax3 = plt.subplot(3, 3, 3)
    x = np.arange(min(data1['num_layers'], data2['num_layers']))
    width = 0.35
    ax3.plot(x, layer_dist1[0][:len(x)], 'o-', label=f'{name1} - {lang1_name}', linewidth=2, markersize=4, color='#2E86AB')
    ax3.plot(x, layer_dist2[0][:len(x)], 's-', label=f'{name2} - {lang1_name}', linewidth=2, markersize=4, color='#58A4B0')
    ax3.plot(x, layer_dist1[1][:len(x)], 'o-', label=f'{name1} - {lang2_name}', linewidth=2, markersize=4, color='#A23B72')
    ax3.plot(x, layer_dist2[1][:len(x)], 's-', label=f'{name2} - {lang2_name}', linewidth=2, markersize=4, color='#F18F01')
    ax3.set_xlabel('Слой')
    ax3.set_ylabel('Количество нейронов')
    ax3.set_title('Сравнение моделей по слоям')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    
    # 4. Total neurons comparison
    ax4 = plt.subplot(3, 3, 4)
    categories = [f'{lang1_name}-specific', f'{lang2_name}-specific', 'Total specific']
    model1_vals = [
        sum(len(l) for l in data1['language_specific_masks']['lang1']),
        sum(len(l) for l in data1['language_specific_masks']['lang2']),
        sum(len(l) for l in data1['language_specific_masks']['lang1']) + 
        sum(len(l) for l in data1['language_specific_masks']['lang2'])
    ]
    model2_vals = [
        sum(len(l) for l in data2['language_specific_masks']['lang1']),
        sum(len(l) for l in data2['language_specific_masks']['lang2']),
        sum(len(l) for l in data2['language_specific_masks']['lang1']) + 
        sum(len(l) for l in data2['language_specific_masks']['lang2'])
    ]
    x = np.arange(len(categories))
    width = 0.35
    ax4.bar(x - width/2, model1_vals, width, label=name1, alpha=0.8, color='#2E86AB')
    ax4.bar(x + width/2, model2_vals, width, label=name2, alpha=0.8, color='#F18F01')
    ax4.set_ylabel('Количество нейронов')
    ax4.set_title('Сравнение общего количества')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, rotation=15, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Overlap visualization - Lang1
    ax5 = plt.subplot(3, 3, 5)
    overlap_cats = ['Пересечение', f'Только\n{name1}', f'Только\n{name2}']
    overlap_vals_lang1 = [
        overlap_data['lang1']['overlap'],
        overlap_data['lang1']['only_model1'],
        overlap_data['lang1']['only_model2']
    ]
    colors = ['#06A77D', '#2E86AB', '#F18F01']
    ax5.bar(overlap_cats, overlap_vals_lang1, color=colors, alpha=0.8)
    ax5.set_ylabel('Количество нейронов')
    ax5.set_title(f'Пересечение: {lang1_name}-специфичные\n(Jaccard: {overlap_data["lang1"]["jaccard"]:.3f})')
    ax5.grid(axis='y', alpha=0.3)
    for i, v in enumerate(overlap_vals_lang1):
        ax5.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    
    # 6. Overlap visualization - Lang2
    ax6 = plt.subplot(3, 3, 6)
    overlap_vals_lang2 = [
        overlap_data['lang2']['overlap'],
        overlap_data['lang2']['only_model1'],
        overlap_data['lang2']['only_model2']
    ]
    ax6.bar(overlap_cats, overlap_vals_lang2, color=colors, alpha=0.8)
    ax6.set_ylabel('Количество нейронов')
    ax6.set_title(f'Пересечение: {lang2_name}-специфичные\n(Jaccard: {overlap_data["lang2"]["jaccard"]:.3f})')
    ax6.grid(axis='y', alpha=0.3)
    for i, v in enumerate(overlap_vals_lang2):
        ax6.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    
    # 7. Heatmap - Model 1 layer distribution
    ax7 = plt.subplot(3, 3, 7)
    heatmap_data1 = layer_dist1.T  # Transpose for better visualization
    im1 = ax7.imshow(heatmap_data1, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax7.set_xlabel('Язык')
    ax7.set_ylabel('Слой')
    ax7.set_title(f'{name1}: Тепловая карта')
    ax7.set_xticks([0, 1])
    ax7.set_xticklabels([lang1_name, lang2_name])
    plt.colorbar(im1, ax=ax7, label='Количество нейронов')
    
    # 8. Heatmap - Model 2 layer distribution
    ax8 = plt.subplot(3, 3, 8)
    heatmap_data2 = layer_dist2.T
    im2 = ax8.imshow(heatmap_data2, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax8.set_xlabel('Язык')
    ax8.set_ylabel('Слой')
    ax8.set_title(f'{name2}: Тепловая карта')
    ax8.set_xticks([0, 1])
    ax8.set_xticklabels([lang1_name, lang2_name])
    plt.colorbar(im2, ax=ax8, label='Количество нейронов')
    
    # 9. Distribution histogram
    ax9 = plt.subplot(3, 3, 9)
    all_counts1 = np.concatenate([layer_dist1[0], layer_dist1[1]])
    all_counts2 = np.concatenate([layer_dist2[0], layer_dist2[1]])
    bins = np.linspace(0, max(all_counts1.max(), all_counts2.max()), 30)
    ax9.hist(all_counts1, bins=bins, alpha=0.5, label=name1, color='#2E86AB', edgecolor='black')
    ax9.hist(all_counts2, bins=bins, alpha=0.5, label=name2, color='#F18F01', edgecolor='black')
    ax9.set_xlabel('Количество нейронов в слое')
    ax9.set_ylabel('Частота')
    ax9.set_title('Гистограмма распределения')
    ax9.legend()
    ax9.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Сохранено: {output_prefix}_comparison.png")
    plt.close()
    
    # Create second figure for detailed layer-by-layer comparison
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Difference plots
    num_layers = min(data1['num_layers'], data2['num_layers'])
    diff_lang1 = layer_dist1[0][:num_layers] - layer_dist2[0][:num_layers]
    diff_lang2 = layer_dist1[1][:num_layers] - layer_dist2[1][:num_layers]
    
    # Lang1 difference
    axes[0, 0].bar(range(num_layers), diff_lang1, color=['#06A77D' if x >= 0 else '#D62828' for x in diff_lang1], alpha=0.7)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0, 0].set_xlabel('Слой')
    axes[0, 0].set_ylabel(f'Разница ({name1} - {name2})')
    axes[0, 0].set_title(f'Разница в {lang1_name}-специфичных нейронах по слоям')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Lang2 difference
    axes[0, 1].bar(range(num_layers), diff_lang2, color=['#06A77D' if x >= 0 else '#D62828' for x in diff_lang2], alpha=0.7)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0, 1].set_xlabel('Слой')
    axes[0, 1].set_ylabel(f'Разница ({name1} - {name2})')
    axes[0, 1].set_title(f'Разница в {lang2_name}-специфичных нейронах по слоям')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Cumulative distribution
    axes[1, 0].plot(range(num_layers), np.cumsum(layer_dist1[0][:num_layers]), 'o-', label=f'{name1} - {lang1_name}', linewidth=2)
    axes[1, 0].plot(range(num_layers), np.cumsum(layer_dist2[0][:num_layers]), 's-', label=f'{name2} - {lang1_name}', linewidth=2)
    axes[1, 0].plot(range(num_layers), np.cumsum(layer_dist1[1][:num_layers]), 'o-', label=f'{name1} - {lang2_name}', linewidth=2)
    axes[1, 0].plot(range(num_layers), np.cumsum(layer_dist2[1][:num_layers]), 's-', label=f'{name2} - {lang2_name}', linewidth=2)
    axes[1, 0].set_xlabel('Слой')
    axes[1, 0].set_ylabel('Кумулятивное количество')
    axes[1, 0].set_title('Кумулятивное распределение по слоям')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Percentage by layer
    pct_lang1_model1 = 100 * layer_dist1[0][:num_layers] / data1['intermediate_size']
    pct_lang2_model1 = 100 * layer_dist1[1][:num_layers] / data1['intermediate_size']
    pct_lang1_model2 = 100 * layer_dist2[0][:num_layers] / data2['intermediate_size']
    pct_lang2_model2 = 100 * layer_dist2[1][:num_layers] / data2['intermediate_size']
    
    axes[1, 1].plot(range(num_layers), pct_lang1_model1, 'o-', label=f'{name1} - {lang1_name}', linewidth=2, markersize=4)
    axes[1, 1].plot(range(num_layers), pct_lang1_model2, 's-', label=f'{name2} - {lang1_name}', linewidth=2, markersize=4)
    axes[1, 1].plot(range(num_layers), pct_lang2_model1, 'o-', label=f'{name1} - {lang2_name}', linewidth=2, markersize=4)
    axes[1, 1].plot(range(num_layers), pct_lang2_model2, 's-', label=f'{name2} - {lang2_name}', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('Слой')
    axes[1, 1].set_ylabel('% от intermediate_size')
    axes[1, 1].set_title('Процент языко-специфичных нейронов по слоям')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_detailed.png', dpi=150, bbox_inches='tight')
    print(f"Сохранено: {output_prefix}_detailed.png")
    plt.close()
    
    # Create third figure for top neurons comparison
    fig3, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    top1 = data1.get('top_lang1_specific_by_rate_diff', [])[:30]
    top2 = data2.get('top_lang1_specific_by_rate_diff', [])[:30]
    
    # Model 1 top neurons
    diffs1 = [n['rate_diff'] for n in top1]
    axes[0].barh(range(len(diffs1)), diffs1, color='#2E86AB', alpha=0.7)
    axes[0].set_xlabel(f'Rate difference ({lang1_name} - {lang2_name})')
    axes[0].set_ylabel('Ранг')
    axes[0].set_title(f'{name1}: Топ-30 нейронов по разнице активации')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Model 2 top neurons
    diffs2 = [n['rate_diff'] for n in top2]
    axes[1].barh(range(len(diffs2)), diffs2, color='#F18F01', alpha=0.7)
    axes[1].set_xlabel(f'Rate difference ({lang1_name} - {lang2_name})')
    axes[1].set_ylabel('Ранг')
    axes[1].set_title(f'{name2}: Топ-30 нейронов по разнице активации')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_top_neurons.png', dpi=150, bbox_inches='tight')
    print(f"Сохранено: {output_prefix}_top_neurons.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Сравнение результатов обнаружения языко-специфичных нейронов для двух моделей"
    )
    parser.add_argument("--json1", type=str, required=True, help="Путь к первому JSON файлу")
    parser.add_argument("--json2", type=str, required=True, help="Путь ко второму JSON файлу")
    parser.add_argument("--name1", type=str, default="Model 1", help="Имя первой модели для отображения")
    parser.add_argument("--name2", type=str, default="Model 2", help="Имя второй модели для отображения")
    parser.add_argument("--output", type=str, default="model_comparison", help="Префикс для выходных файлов")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("СРАВНЕНИЕ ЯЗЫКО-СПЕЦИФИЧНЫХ НЕЙРОНОВ")
    print("=" * 80)
    
    # Load data
    data1 = load_json(args.json1)
    data2 = load_json(args.json2)
    
    # Extract language names from JSON files
    lang1_name = data1.get('columns', {}).get('lang1', 'lang1')
    lang2_name = data1.get('columns', {}).get('lang2', 'lang2')
    
    print(f"Языки: Lang1 = {lang1_name}, Lang2 = {lang2_name}")
    
    # Generate reports
    report_parts = []
    
    # Basic statistics
    report_parts.append(compare_basic_stats(data1, data2, args.name1, args.name2, lang1_name, lang2_name))
    
    # Language-specific neurons
    report_parts.append(compare_language_specific_neurons(data1, data2, args.name1, args.name2, lang1_name, lang2_name))
    
    # Layer distribution
    layer_report, layer_dist1, layer_dist2 = analyze_layer_distribution(data1, data2, args.name1, args.name2, lang1_name, lang2_name)
    report_parts.append(layer_report)
    
    # Top neurons
    report_parts.append(analyze_top_neurons(data1, data2, args.name1, args.name2, lang1_name, lang2_name))
    
    # Overlap analysis
    overlap_data, overlap_report = calculate_overlap(data1, data2, args.name1, args.name2, lang1_name, lang2_name)
    report_parts.append(overlap_report)
    
    # Combine all reports
    full_report = "\n".join(report_parts)
    
    # Print to console
    print(full_report)
    
    # Save to file
    report_file = f"{args.output}_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(full_report)
    print(f"\n{'-'*80}")
    print(f"Отчёт сохранён: {report_file}")
    
    # Generate visualizations
    print(f"\nГенерация визуализаций...")
    create_visualizations(data1, data2, args.name1, args.name2, 
                         layer_dist1, layer_dist2, overlap_data, args.output, lang1_name, lang2_name)
    
    print(f"\n{'='*80}")
    print("ГОТОВО!")
    print(f"{'='*80}")
    print(f"Файлы созданы:")
    print(f"  - {report_file}")
    print(f"  - {args.output}_comparison.png")
    print(f"  - {args.output}_detailed.png")
    print(f"  - {args.output}_top_neurons.png")


if __name__ == "__main__":
    main()

