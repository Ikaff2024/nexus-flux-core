#!/usr/bin/env python3
"""
Outil de visualisation de la convergence pour nexus-flux-core
Génère des graphiques de H_V, momentum et stabilité par itération
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_iteration_data(filepath):
    """Charge les données d'itérations depuis un fichier JSONL"""
    iterations = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'iteration' in data or 'step' in data:
                        iterations.append(data)
                except json.JSONDecodeError:
                    continue
    return iterations

def extract_metrics_v2(filepath):
    """Extrait les métriques des fichiers de résultats V2"""
    h_v_history = []
    momentum_history = []
    stability_history = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'H_V' in data:
                        h_v_history.append(data.get('H_V', 0))
                    if 'momentum' in data:
                        momentum_history.append(data.get('momentum', 0))
                    if 'stability' in data:
                        stability_history.append(data.get('stability', 0))
                except json.JSONDecodeError:
                    continue
    
    return h_v_history, momentum_history, stability_history

def create_mock_data(n_iterations=10):
    """Crée des données de démonstration si aucun fichier n'est disponible"""
    np.random.seed(42)
    
    # H_V décroît exponentiellement vers 0.95
    h_v = [0.0 + (0.95 - 0.0) * (1 - np.exp(-0.5 * i)) + np.random.randn() * 0.05 
           for i in range(n_iterations)]
    h_v = [max(0, min(1, x)) for x in h_v]
    
    # Momentum décroît vers 0
    momentum = [0.7 * np.exp(-0.4 * i) + np.random.randn() * 0.03 
                for i in range(n_iterations)]
    momentum = [max(0, x) for x in momentum]
    
    # Stabilité croît vers 0.97
    stability = [0.3 + (0.97 - 0.3) * (1 - np.exp(-0.6 * i)) + np.random.randn() * 0.03 
                 for i in range(n_iterations)]
    stability = [max(0, min(1, x)) for x in stability]
    
    return h_v, momentum, stability

def plot_convergence(h_v_history, momentum_history, stability_history, output_path='convergence_plot.png'):
    """Génère le graphique de convergence"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Nexus-Flux V2 - Convergence Metrics', fontsize=16, fontweight='bold')
    
    iterations = list(range(len(h_v_history)))
    
    # Graphique H_V (Entropie vectorielle)
    ax1 = axes[0]
    ax1.plot(iterations, h_v_history, 'b-o', linewidth=2, markersize=8, label='H_V (Entropie)')
    ax1.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='Seuil convergence (0.9)')
    ax1.set_ylabel('Entropie Vectorielle (H_V)', fontsize=12)
    ax1.set_title('Évolution de l\'entropie vectorielle - Plus haut = meilleure convergence', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.0])
    
    # Graphique Momentum
    ax2 = axes[1]
    ax2.plot(iterations, momentum_history, 'r-s', linewidth=2, markersize=8, label='Momentum')
    ax2.axhline(y=0.05, color='g', linestyle='--', alpha=0.7, label='Seuil convergence (0.05)')
    ax2.set_ylabel('Momentum', fontsize=12)
    ax2.set_title('Évolution du momentum - Plus bas = système plus stable', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(momentum_history) * 1.2 if momentum_history else 1])
    
    # Graphique Stabilité
    ax3 = axes[2]
    ax3.plot(iterations, stability_history, 'g-^', linewidth=2, markersize=8, label='Stabilité')
    ax3.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='Seuil cible (0.9)')
    ax3.set_ylabel('Stabilité', fontsize=12)
    ax3.set_xlabel('Itération', fontsize=12)
    ax3.set_title('Évolution de la stabilité - Plus haut = agents mieux alignés', fontsize=12)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Graphique généré : {output_path}")

def plot_comparison(output_path='comparison_v1_v2.png'):
    """Génère un graphique comparatif V1 vs V2"""
    
    # Données V1 (depuis analyze_results)
    v1_stability = 0.2921
    v1_hv_drop = 0.0058
    v1_convergence = 0
    
    # Données V2 (depuis analyze_results)
    v2_stability = 0.9666
    v2_hv_drop = 0.7289
    v2_convergence = 100  # pourcentage
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Comparaison Nexus-Flux V1 vs V2', fontsize=16, fontweight='bold')
    
    x = np.arange(2)  # V1 et V2
    width = 0.35
    
    # Graphique 1: Stabilité
    bars1 = axes[0].bar(x - width/2, [v1_stability, v2_stability], width, 
                        label='V1', color='#ff6b6b', alpha=0.8)
    axes[0].set_ylabel('Score (0-1)')
    axes[0].set_title(f'Stabilité\nV2: +{((v2_stability/v1_stability)-1)*100:.0f}% vs V1', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['V1', 'V2'])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1.0])
    
    for bar, val in zip(bars1, [v1_stability, v2_stability]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Graphique 2: H_V Drop
    bars2 = axes[1].bar(x - width/2, [v1_hv_drop, v2_hv_drop], width,
                        label='V1', color='#4ecdc4', alpha=0.8)
    axes[1].set_ylabel('Réduction H_V')
    axes[1].set_title(f'Entropie (H_V Drop)\nV2: {(v2_hv_drop/v1_hv_drop):.1f}x mieux', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['V1', 'V2'])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, [v1_hv_drop, v2_hv_drop]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Graphique 3: Taux de convergence
    bars3 = axes[2].bar(x - width/2, [v1_convergence, v2_convergence], width,
                        label='V1', color='#ffe66d', alpha=0.8)
    axes[2].set_ylabel('Taux de convergence (%)')
    axes[2].set_title(f'Convergence\nV2: {v2_convergence}% vs {v1_convergence}%', fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(['V1', 'V2'])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_ylim([0, 100])
    
    for bar, val in zip(bars3, [v1_convergence, v2_convergence]):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Graphique comparatif généré : {output_path}")

def main():
    print("=" * 70)
    print("  VISUALISATION DE CONVERGENCE - NEXUS-FLUX-CORE")
    print("=" * 70)
    print()
    
    # Essayer de charger les données réelles
    v2_file = Path('nexus_flux_v2_iterations.jsonl')
    
    if v2_file.exists():
        print(f"Chargement des données depuis {v2_file}...")
        try:
            h_v, momentum, stability = extract_metrics_v2(str(v2_file))
            if len(h_v) == 0:
                print("Aucune donnée trouvée, utilisation de données de démonstration...")
                h_v, momentum, stability = create_mock_data()
        except Exception as e:
            print(f"Erreur de lecture: {e}, utilisation de données de démonstration...")
            h_v, momentum, stability = create_mock_data()
    else:
        print("Fichier non trouvé, utilisation de données de démonstration...")
        h_v, momentum, stability = create_mock_data()
    
    print(f"Nombre d'itérations: {len(h_v)}")
    print()
    
    # Générer les graphiques
    plot_convergence(h_v, momentum, stability)
    plot_comparison()
    
    print()
    print("=" * 70)
    print("  VISUALISATIONS GÉNÉRÉES AVEC SUCCÈS")
    print("=" * 70)
    print()
    print("Fichiers créés:")
    print("  - convergence_plot.png (évolution des métriques)")
    print("  - comparison_v1_v2.png (comparaison V1 vs V2)")

if __name__ == "__main__":
    main()
