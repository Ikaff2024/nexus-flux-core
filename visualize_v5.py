#!/usr/bin/env python3
"""
Visualisation de la convergence Nexus-Flux V5 - Guided Flux
Génère des graphiques comparant l'évolution de la variance, de l'alignement et du momentum.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_results(filename="v5_results.json"):
    with open(filename, 'r') as f:
        return json.load(f)

def plot_convergence(data):
    history = data['history_summary']
    iterations = [h['iteration'] if 'iteration' in h else i+1 for i, h in enumerate(history)]
    
    variances = [h['variance'] for h in history]
    alignments = [h['alignment'] for h in history]
    momentums = [h['momentum'] for h in history]
    qualities = [h['avg_quality'] for h in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Nexus-Flux V5 - Convergence Guidée\n"Système de transport urbain durable et autonome"', fontsize=14, fontweight='bold')
    
    # Graphique 1: Variance (Cohésion)
    ax1 = axes[0, 0]
    ax1.plot(iterations, variances, 'b-o', linewidth=2, markersize=6, label='Variance')
    ax1.axhline(y=0.05, color='g', linestyle='--', alpha=0.7, label='Seuil convergence (0.05)')
    ax1.set_xlabel('Itération')
    ax1.set_ylabel('Variance')
    ax1.set_title('Cohésion du Groupe (Baisse de variance)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Annotation finale
    ax1.annotate(f'Final: {variances[-1]:.4f}', 
                xy=(iterations[-1], variances[-1]), 
                xytext=(iterations[-1]*0.7, variances[-1]*1.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')
    
    # Graphique 2: Alignement avec la cible (North Star)
    ax2 = axes[0, 1]
    ax2.plot(iterations, alignments, 'g-o', linewidth=2, markersize=6, label='Alignement')
    ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='Excellent (>0.9)')
    ax2.set_xlabel('Itération')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Alignement avec l\'Idéal (North Star)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    ax2.annotate(f'Final: {alignments[-1]:.4f}', 
                xy=(iterations[-1], alignments[-1]), 
                xytext=(iterations[-1]*0.7, alignments[-1]*0.85),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')
    
    # Graphique 3: Momentum (Vélocité)
    ax3 = axes[1, 0]
    ax3.plot(iterations, momentums, 'r-o', linewidth=2, markersize=6, label='Momentum')
    ax3.axhline(y=0.5, color='purple', linestyle='--', alpha=0.7, label='Stabilisation (<0.5)')
    ax3.set_xlabel('Itération')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Momentum (Vitesse de changement)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    
    ax3.annotate(f'Final: {momentums[-1]:.4f}', 
                xy=(iterations[-1], momentums[-1]), 
                xytext=(iterations[-1]*0.7, momentums[-1]*1.2),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue', fontweight='bold')
    
    # Graphique 4: Qualité moyenne
    ax4 = axes[1, 1]
    ax4.plot(iterations, qualities, 'm-o', linewidth=2, markersize=6, label='Qualité')
    ax4.set_xlabel('Itération')
    ax4.set_ylabel('Score (0-1)')
    ax4.set_title('Qualité Moyenne des Idées')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.3, 0.7)
    
    ax4.annotate(f'Final: {qualities[-1]:.4f}', 
                xy=(iterations[-1], qualities[-1]), 
                xytext=(iterations[-1]*0.7, qualities[-1]*1.15),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    
    # Sauvegarde
    output_file = "v5_convergence_plot.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé : {output_file}")
    plt.show()

def print_summary(data):
    print("\n" + "="*60)
    print("📊 RÉSUMÉ NEXUS-FLUX V5 - GUIDED FLUX")
    print("="*60)
    print(f"Statut       : {data['status'].upper()}")
    print(f"Itérations   : {data['iterations']}")
    print(f"Temps        : {data['time_elapsed']:.3f}s")
    print(f"\n🎯 Métriques Finales:")
    print(f"   Cohésion     : {1.0 - data['final_variance']:.4f} ({data['final_variance']:.4f} variance)")
    print(f"   Alignement   : {data['final_alignment']:.4f} (avec North Star)")
    print(f"   Qualité      : {data['final_quality']:.4f}")
    print(f"\n💡 Idées Convergentes:")
    for i, idea in enumerate(data['final_ideas']):
        print(f"   [{i}] {idea}")
    print("="*60)

if __name__ == "__main__":
    try:
        data = load_results()
        print_summary(data)
        plot_convergence(data)
    except FileNotFoundError:
        print("❌ Erreur: v5_results.json non trouvé.")
        print("Exécutez d'abord: python3 nexus_flux_v5.py")
    except Exception as e:
        print(f"❌ Erreur: {e}")
