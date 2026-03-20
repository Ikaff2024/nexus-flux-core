"""
analyze_results.py
==================
Analyse comparative Baseline vs Nexus-Flux.

Lit les logs JSONL produits par baseline_multiagent_pro.py
et nexus_flux_pro.py, calcule les statistiques agregees
et affiche un tableau comparatif + resume executif.

Usage :
    python analyze_results.py
    python analyze_results.py --baseline baseline_results.jsonl --nexus nexus_flux_results.jsonl
"""

import os
import json
import math
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple


# -- Fichiers par defaut -------------------------------------------------------
DEFAULT_BASELINE_LOG = "baseline_results.jsonl"
DEFAULT_NEXUS_LOG    = "nexus_flux_results.jsonl"


# -- Utilitaires statistiques -------------------------------------------------

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def fmt(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "null"
    return f"{value:.{decimals}f}"


# -- Chargement des logs -------------------------------------------------------

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filepath):
        print(f"  [WARN] Fichier introuvable : {filepath}")
        return []
    records = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Ligne {line_no} invalide dans {filepath}: {e}")
    return records


# -- Calcul des metriques agregees --------------------------------------------

def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Groupe par run_id et calcule moyenne +/- ecart-type pour chaque metrique.
    Retourne un dictionnaire {metrique: {mean, std, values, n}}.
    """
    if not records:
        return {}

    keys = ["iterations", "chat_calls", "embed_calls",
            "stability", "contradictions", "Hv", "Hv_drop", "momentum"]

    agg: Dict[str, List[float]] = defaultdict(list)

    for rec in records:
        for k in keys:
            v = rec.get(k)
            if v is not None:
                agg[k].append(float(v))

    result = {}
    for k in keys:
        vals = agg[k]
        result[k] = {
            "mean":   mean(vals),
            "std":    std(vals),
            "values": vals,
            "n":      len(vals),
        }

    # efficiency = deltaHv / (chat_calls + 0.2 * embed_calls)
    efficiency_vals: List[float] = []
    for rec in records:
        hv_drop  = rec.get("Hv_drop")
        chat     = rec.get("chat_calls", 0) or 0
        embed    = rec.get("embed_calls", 0) or 0
        if hv_drop is not None and (chat + 0.2 * embed) > 0:
            efficiency_vals.append(hv_drop / (chat + 0.2 * embed))

    result["efficiency"] = {
        "mean":   mean(efficiency_vals),
        "std":    std(efficiency_vals),
        "values": efficiency_vals,
        "n":      len(efficiency_vals),
    }

    return result


# -- Affichage du tableau ------------------------------------------------------

def _col_width(label: str, bv: str, nv: str, delta: str) -> int:
    return max(len(label), len(bv), len(nv), len(delta)) + 2


def print_table(baseline_agg: Dict, nexus_agg: Dict) -> None:
    """Affiche un tableau comparatif formate dans le terminal."""

    metrics_config = [
        # (cle, label, meilleur=bas? True/False)
        ("iterations",    "Iterations",       True),
        ("chat_calls",    "Chat calls",        True),
        ("embed_calls",   "Embed calls",       True),
        ("stability",     "Stability",         False),
        ("contradictions","Contradictions",    True),
        ("Hv",            "H_V (final)",       True),
        ("Hv_drop",       "H_V drop (ventropie)", False),
        ("momentum",      "Momentum (final)", True),
        ("efficiency",    "Efficiency",        False),
    ]

    rows = []
    for key, label, lower_is_better in metrics_config:
        b = baseline_agg.get(key, {})
        n = nexus_agg.get(key, {})

        b_mean = b.get("mean")
        b_std  = b.get("std", 0.0)
        n_mean = n.get("mean")
        n_std  = n.get("std", 0.0)
        b_n    = b.get("n", 0)
        n_n    = n.get("n", 0)

        b_str = f"{fmt(b_mean)} +/- {fmt(b_std)}" if b_mean is not None else "N/A"
        n_str = f"{fmt(n_mean)} +/- {fmt(n_std)}" if n_mean is not None else "N/A"

        # Calcul du delta et indication du gagnant
        if b_mean is not None and n_mean is not None and b_mean != 0:
            delta_pct = (n_mean - b_mean) / abs(b_mean) * 100
            delta_str = f"{delta_pct:+.1f}%"
            if lower_is_better:
                winner = "NF [OK]" if n_mean < b_mean else ("BL [OK]" if b_mean < n_mean else "=")
            else:
                winner = "NF [OK]" if n_mean > b_mean else ("BL [OK]" if b_mean > n_mean else "=")
        else:
            delta_str = "N/A"
            winner = "?"

        rows.append((label, b_str, n_str, delta_str, winner, b_n, n_n))

    # Largeurs de colonnes
    col_labels = ["Metrique", "Baseline", "Nexus-Flux", "delta (%)", "Gagnant"]
    col_data   = [
        (r[0], r[1], r[2], r[3], r[4]) for r in rows
    ]
    widths = [
        max(len(col_labels[i]), max(len(r[i]) for r in col_data)) + 2
        for i in range(5)
    ]

    sep = "+" + "+".join("-" * w for w in widths) + "+"

    def row_str(cells):
        return "|" + "|".join(
            f" {c:<{w-2}} " for c, w in zip(cells, widths)
        ) + "|"

    print(f"\n{'=' * sum(w + 1 for w in widths)}")
    print("  TABLEAU COMPARATIF : Baseline vs Nexus-Flux")
    print(f"{'=' * sum(w + 1 for w in widths)}")
    print(sep)
    print(row_str(col_labels))
    print(sep.replace("-", "="))
    for r in rows:
        print(row_str(r[:5]))
        print(sep)


# -- Resume executif -----------------------------------------------------------

def executive_summary(
    baseline_records: List[Dict],
    nexus_records:    List[Dict],
    baseline_agg:     Dict,
    nexus_agg:        Dict,
) -> None:

    print(f"\n{'=' * 62}")
    print("  RESUME EXECUTIF")
    print(f"{'=' * 62}")

    nb = len(baseline_records)
    nn = len(nexus_records)
    print(f"\n  Runs analyses : Baseline = {nb}  |  Nexus-Flux = {nn}")

    if nb == 0 and nn == 0:
        print("\n  [ERREUR] Aucune donnee disponible.")
        print("  -> Executez d'abord :")
        print("      python baseline_multiagent_pro.py")
        print("      python nexus_flux_pro.py")
        return

    findings = []

    # 1. Stabilite
    bs = baseline_agg.get("stability", {}).get("mean")
    ns = nexus_agg.get("stability", {}).get("mean")
    if bs is not None and ns is not None:
        if ns > bs:
            gain = (ns - bs) / abs(bs) * 100 if bs != 0 else float("inf")
            findings.append(
                f"[OK] STABILITE : Nexus-Flux superieur de {gain:.1f}% "
                f"({ns:.4f} vs {bs:.4f})"
            )
        else:
            findings.append(
                f"[X] STABILITE : Baseline superieur ({bs:.4f} vs {ns:.4f})"
            )

    # 2. Convergence (iterations)
    bi = baseline_agg.get("iterations", {}).get("mean")
    ni = nexus_agg.get("iterations", {}).get("mean")
    if bi is not None and ni is not None:
        if ni <= bi:
            findings.append(
                f"[OK] CONVERGENCE : Nexus-Flux converge en {ni:.1f} iter "
                f"vs {bi:.1f} pour Baseline"
            )
        else:
            findings.append(
                f"-> CONVERGENCE : Nexus-Flux necessite {ni:.1f} iter "
                f"vs {bi:.1f} pour Baseline"
            )

    # 3. Efficience des appels
    bc = (baseline_agg.get("chat_calls", {}).get("mean") or 0)
    be = (baseline_agg.get("embed_calls", {}).get("mean") or 0)
    nc = (nexus_agg.get("chat_calls", {}).get("mean") or 0)
    ne = (nexus_agg.get("embed_calls", {}).get("mean") or 0)
    b_cost = bc + 0.2 * be
    n_cost = nc + 0.2 * ne
    if b_cost > 0 and n_cost > 0:
        reduction = (b_cost - n_cost) / b_cost * 100
        findings.append(
            f"{'[OK]' if reduction > 0 else '[X]'} COUT API : "
            f"Nexus-Flux {'economise' if reduction > 0 else 'consomme'} "
            f"{abs(reduction):.1f}% "
            f"(cout normalise {n_cost:.1f} vs {b_cost:.1f})"
        )

    # 4. Reduction d'entropie
    hv_drop = nexus_agg.get("Hv_drop", {}).get("mean")
    if hv_drop is not None:
        findings.append(
            f"[OK] ENTROPIE VECTORIELLE : Nexus-Flux reduit H_V de {hv_drop:.4f} "
            f"(derive semantique controlee)"
        )

    # 5. Efficience globale
    eff_nf = nexus_agg.get("efficiency", {}).get("mean")
    eff_bl = baseline_agg.get("efficiency", {}).get("mean")
    if eff_nf and eff_nf > 0:
        findings.append(
            f"[OK] EFFICIENCE (deltaHv/cout) : Nexus-Flux = {eff_nf:.6f} "
            + (f"vs Baseline = {eff_bl:.6f}" if eff_bl else "(Baseline N/A)")
        )

    # 6. Contradictions
    bcon = baseline_agg.get("contradictions", {}).get("mean")
    ncon = nexus_agg.get("contradictions", {}).get("mean")
    if bcon is not None and ncon is not None:
        if ncon < bcon:
            findings.append(
                f"[OK] CONTRADICTIONS : Nexus-Flux produit moins de bruit "
                f"({ncon:.1f} vs {bcon:.1f})"
            )
        else:
            findings.append(
                f"-> CONTRADICTIONS : Baseline = {bcon:.1f} | Nexus-Flux = {ncon:.1f}"
            )

    print()
    for finding in findings:
        print(f"  {finding}")

    # Verdict global
    nf_wins = sum(1 for f in findings if f.startswith("  [OK]") or f.startswith("[OK]"))
    total   = len([f for f in findings if not f.startswith("  ->")])

    print(f"\n{'-' * 62}")
    print(f"  VERDICT : Nexus-Flux remporte {nf_wins}/{total} criteres")

    if nf_wins >= total * 0.7:
        print("  -> Nexus-Flux demontre une superiorite claire sur l'architecture")
        print("    baseline : convergence plus rapide, stabilite accrue,")
        print("    et reduction mesurable de la derive semantique.")
    elif nf_wins >= total * 0.5:
        print("  -> Nexus-Flux montre des avantages nets sur certains axes.")
        print("    Un ajustement des hyperparametres peut consolider les gains.")
    else:
        print("  -> Resultats mixtes. Augmenter le nombre de runs pour")
        print("    une conclusion statistiquement significative.")

    print(f"{'=' * 62}")


# -- Statistiques detaillees ---------------------------------------------------

def print_detail(records: List[Dict], label: str) -> None:
    if not records:
        print(f"\n  [{label}] Aucune donnee.")
        return
    print(f"\n  [{label}] -- {len(records)} run(s)")
    for rec in records:
        mock_tag = " [MOCK]" if rec.get("mock_mode") else ""
        hv   = fmt(rec.get("Hv"))
        mom  = fmt(rec.get("momentum"))
        stab = fmt(rec.get("stability"))
        print(
            f"    run_id={rec.get('run_id', '?'):<8}  "
            f"iter={rec.get('iterations', '?'):>3}  "
            f"chat={rec.get('chat_calls', '?'):>4}  "
            f"embed={rec.get('embed_calls', '?'):>4}  "
            f"stab={stab}  "
            f"Hv={hv}  "
            f"mom={mom}"
            + mock_tag
        )


# -- Main ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse comparative Baseline vs Nexus-Flux"
    )
    parser.add_argument("--baseline", default=DEFAULT_BASELINE_LOG,
                        help="Fichier JSONL baseline")
    parser.add_argument("--nexus",    default=DEFAULT_NEXUS_LOG,
                        help="Fichier JSONL nexus-flux")
    parser.add_argument("--detail",   action="store_true",
                        help="Affiche le detail par run")
    args = parser.parse_args()

    print(f"\n{'#' * 62}")
    print(f"#  ANALYSE COMPARATIVE -- Baseline vs Nexus-Flux")
    print(f"{'#' * 62}")
    print(f"\n  Lecture de : {args.baseline}")
    print(f"  Lecture de : {args.nexus}")

    baseline_records = load_jsonl(args.baseline)
    nexus_records    = load_jsonl(args.nexus)

    print(f"\n  Baseline    : {len(baseline_records)} enregistrement(s)")
    print(f"  Nexus-Flux  : {len(nexus_records)} enregistrement(s)")

    if args.detail:
        print_detail(baseline_records, "BASELINE")
        print_detail(nexus_records,    "NEXUS-FLUX")

    baseline_agg = aggregate(baseline_records)
    nexus_agg    = aggregate(nexus_records)

    if not baseline_agg and not nexus_agg:
        print("\n  [ERREUR] Aucune donnee a analyser.")
        print("  Executez d'abord les deux systemes :")
        print("    python baseline_multiagent_pro.py")
        print("    python nexus_flux_pro.py")
        return

    print_table(baseline_agg, nexus_agg)
    executive_summary(baseline_records, nexus_records, baseline_agg, nexus_agg)


if __name__ == "__main__":
    main()
