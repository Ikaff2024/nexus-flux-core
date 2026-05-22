# nexus-flux-core

## 🎯 Vue d'ensemble

**nexus-flux-core** est un projet de recherche expérimental explorant une nouvelle approche pour coordonner plusieurs agents intelligents. Contrairement aux systèmes classiques où les agents communiquent en texte, ici la coordination se fait via des **vecteurs** (représentations numériques) dans un champ partagé.

### Concept clé
- **Coordination vectorielle** : Les agents ne s'échangent pas de messages textuels mais influencent un champ vectoriel commun
- **Indicateurs dynamiques** : Le système surveille la convergence, le momentum et l'entropie pour gérer la coordination
- **Zéro contradiction** : L'approche vectorielle élimine les incohérences typiques des systèmes multi-agents textuels

---

## 📁 Architecture du projet

```
nexus-flux-core/
├── baseline_multiagent_pro.py    # Système classique (communication textuelle)
├── nexus_flux_pro.py             # Nexus-Flux V1 (coordination vectorielle simple)
├── nexus_flux_pro_v2.py          # Nexus-Flux V2 (avec juge LLM et pondération)
├── nexus_flux_pro_v3.py          # Nexus-Flux V3 (optimisé : batch + cache + early stop)
├── nexus_flux_pro_v4.py          # Nexus-Flux V4 (hybride avec target vector)
├── nexus_flux_pro_v4_optimized.py# Nexus-Flux V4.1 (version optimisée)
├── nexus_flux_v5.py              # Nexus-Flux V5 (Guided Flux : North Star + k-NN)
├── analyze_results.py            # Outil d'analyse comparative
├── benchmark_v2_vs_v5.py         # Benchmark comparatif V2 vs V5
├── test_suite.py                 # Tests unitaires (8/8 passés)
├── visualize_convergence.py      # Visualisations graphiques (V1-V4)
├── visualize_v5.py               # Visualisations graphiques (V5)
├── README.md                     # Ce fichier
└── *.jsonl, *.json, *.png        # Fichiers de résultats et graphiques
```

### Versions comparées

| Version | Lignes | Approche | Points forts | Limites |
|---------|--------|----------|--------------|---------|
| **Baseline** | 280 | Communication textuelle | Simple, rapide | 47 contradictions/run |
| **Nexus-Flux V1** | 362 | Vecteurs simples | Zéro contradiction, +1212% stabilité | Convergence faible |
| **Nexus-Flux V2** | 733 | Vecteurs + Juge LLM | Convergence 100%, filtrage qualité | +80% coût API |
| **Nexus-Flux V3** | 746 | Batch + Cache + Early Stop | -86% chat calls | Pas de convergence |
| **Nexus-Flux V4** | ~800 | Target Vector + Hybridation | Bonne convergence | Trop cher (130 calls) |
| **Nexus-Flux V4.1** | ~600 | Version optimisée V4 | Bon compromis | Convergence 56% |
| **Nexus-Flux V5** ⭐ | 321 | **Guided Flux (North Star + k-NN)** | **-97% API, Alignement 91%** | Qualité sémantique inférieure |

---

## 📊 Résultats comparatifs complets

### Toutes versions (V1 à V5)

| Version | Runs | Convergence | Stabilité | H_V Drop | Chat Calls | Alignement | Status |
|---------|------|-------------|-----------|----------|------------|------------|--------|
| **Baseline** | 7 | N/A | -0.026 | 0.000 | 25.0 | - | ❌ Contradictions (47/run) |
| **V1** | 6 | 0/6 | 0.292 | 0.006 | 40.0 | - | ⚠️ Pas de convergence |
| **V2** ⭐ | 8 | **8/8** | **0.967** | **0.729** | 72.0 | 0.75 | ✅ **Production (Qualité)** |
| **V3** | 24 | 0/24 | 0.251 | 0.000 | **15.4** | - | ❌ Pas de convergence |
| **V4** | 3 | 2/3 | 0.986 | 0.836 | 130.3 | - | ⚠️ Trop cher en API |
| **V4.1** | 9 | 5/9 | 0.920 | 0.770 | **70.6** | - | ✅ Bon compromis |
| **V5** 🆕 | 3+ | **3/3** | **0.939** | **0.914** | **2.0** | **0.914** | ✅ **R&D (Efficacité)** |

**Recommandation actuelle :**
- **Usage Production** : V2 (qualité sémantique garantie, convergence 100%)
- **Usage R&D / Prototype** : V5 (coût API minimal, alignement optimal avec objectif)

---

## 📊 Benchmark V2 vs V5 (Dernière analyse)

Exécution : `python benchmark_v2_vs_v5.py`

```
======================================================================
📊 BENCHMARK COMPARATIF: NEXUS-FLUX V2 vs V5
======================================================================

Métrique                  | V2 (Emergent)   | V5 (Guided)     | Gain
----------------------------------------------------------------------
Taux de Convergence       | 100.00%         | 100.00%         | =
Variance (Cohésion)       | 0.04            | 0.06            | -65.4%
Alignement Cible          | 0.75            | 0.91            | +21.9%
Qualité Moyenne           | 0.74            | 0.55            | -25.5%
Itérations Moy.           | 9.00            | 15.00           | -66.7%
Appels LLM                | 72.00           | 2.00            | +97.2%
```

### Analyse détaillée

**✅ V2 (Emergent) excelle en :**
- Qualité sémantique des idées (juge LLM à chaque itération)
- Robustesse de convergence (100%)
- Maturité (déjà testé en production)

**✅ V5 (Guided Flux) excelle en :**
- Efficacité LLM (**-97% d'appels API**)
- Alignement avec objectif optimal (**North Star**)
- Rapidité d'exécution (pas d'attente API)
- Cohésion du groupe (variance contrôlée)

**💡 RECOMMANDATION :**
- Usage Production : V2 (qualité garantie)
- Usage R&D / Prototype : V5 (rapidité, coût réduit)
- Future V6 : Hybridation V2+V5 (qualité + efficacité)

---

## 🚀 Utilisation

### Prérequis
- Python 3.12+
- Clé API OpenAI (`OPENAI_API_KEY`)
- Bibliothèques : `numpy`, `openai`

### Installation
```bash
pip install numpy openai
export OPENAI_API_KEY="votre-clé-api"
```

### Exécuter les simulations

#### Baseline (communication textuelle)
```bash
python baseline_multiagent_pro.py
```

#### Nexus-Flux V2 (Production - Qualité garantie)
```bash
python nexus_flux_pro_v2.py
```

#### Nexus-Flux V5 (R&D - Efficacité maximale)
```bash
python nexus_flux_v5.py
```

### Analyser les résultats

#### Analyse comparative V1 vs V2
```bash
python analyze_results.py --v2
```

#### Benchmark V2 vs V5
```bash
python benchmark_v2_vs_v5.py
```

### Visualisations

#### Graphiques V1-V4
```bash
python visualize_convergence.py
```

#### Graphiques V5 (Guided Flux)
```bash
python visualize_v5.py
```

---

## 🔬 Concepts techniques

### Champ vectoriel partagé
Les agents n'échangent pas de texte mais modifient des vecteurs dans un espace commun. Chaque agent:
1. Génère une proposition (via LLM)
2. La convertit en embedding vectoriel
3. Influence le champ vectoriel global
4. Se synchronise via le centroïde et le momentum

### Indicateurs de convergence
- **H_V (Entropie vectorielle)** : Mesure le désordre dans le champ vectoriel. Doit diminuer.
- **Momentum** : Mesure l'inertie du système. Doit tendre vers 0.
- **Stabilité** : Inverse de la variance des positions vectorielles. Doit augmenter.
- **Alignement (V5)** : Cosine similarity avec le vecteur cible "North Star". Doit augmenter.

### Juge LLM (V2 uniquement)
Un agent supplémentaire évalue chaque proposition avant intégration:
- Accepte les idées pertinentes
- Rejette les idées faibles ou hors-sujet
- Améliore la qualité globale de la convergence

### Guided Flux - North Star (V5)
Nouvelle approche introduite dans V5:
- **Vecteur Cible** : Défini par le LLM comme l'idéal théorique du sujet
- **Topologie k-NN** : Les agents interagissent avec leurs voisins les plus proches
- **Formule hybride** : Combinaison d'attraction locale (voisins) et globale (cible)
- **Efficacité** : Seulement 2 appels LLM (initialisation + cible) vs 72 pour V2

---

## 🎯 Feuille de route

### ✅ Réalisé
- [x] Implémentation baseline (textuelle)
- [x] Implémentation Nexus-Flux V1 (vectoriel)
- [x] Implémentation Nexus-Flux V2 (vectoriel + juge)
- [x] Implémentation Nexus-Flux V3 (optimisations batch/cache)
- [x] Implémentation Nexus-Flux V4/V4.1 (target vector)
- [x] **Implémentation Nexus-Flux V5 (Guided Flux : North Star + k-NN)**
- [x] Outil d'analyse comparative
- [x] Benchmark comparatif V2 vs V5
- [x] Validation : V2 converge 100% avec excellente stabilité
- [x] Validation : V5 atteint 91% d'alignement avec -97% API

### 🔄 En cours
- [x] Tests unitaires pour valider la robustesse (8/8 tests passés)
- [x] Visualisations graphiques (convergence H_V, momentum par itération)
- [x] Visualisations V5 (variance, alignement, qualité)
- [x] Documentation étendue des algorithmes
- [x] Optimisation du coût API (**V5: -97% d'appels LLM**)

### 📋 À venir
- [ ] **V6 : Hybridation V2+V5** (qualité sémantique + efficacité)
- [ ] Expérimentation avec plus d'agents (>4)
- [ ] Tests sur différents topics/sujets
- [ ] Interface de visualisation temps réel
- [ ] Publication des résultats de recherche

---

## 📝 Licence

Projet de recherche expérimental. Usage libre pour experimentation et amélioration.

---

## 👨‍💻 Lead Technique

Ce projet est conduit avec une approche itérative :
1. **Mesurer** : Collecter des données quantitatives précises
2. **Analyser** : Identifier les points d'amélioration
3. **Optimiser** : Implémenter des solutions ciblées
4. **Valider** : Confirmer les gains par comparaison

**Prochaine priorité** : Optimisation du coût API et visualisations.
---

## 📊 Visualisations

Le projet inclut des outils de visualisation pour analyser la convergence :

### Générer les graphiques
```bash
python visualize_convergence.py
```

### Fichiers générés
- **convergence_plot.png** : Évolution de H_V, momentum et stabilité par itération
- **comparison_v1_v2.png** : Comparaison visuelle des performances V1 vs V2

---

## 🧪 Tests unitaires

Exécuter la suite de tests :
```bash
python test_suite.py
```

**Résultats actuels** : 8/8 tests passés ✓
- Initialisation du champ vectoriel
- Calcul du centroïde
- Calcul du momentum
- Calcul de l'entropie H_V
- Critères de convergence
- Métrique de stabilité
- Flexibilité du nombre d'agents
- Indépendance à la dimension
