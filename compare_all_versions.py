#!/usr/bin/env python3
"""Comparaison complète de toutes les versions Nexus-Flux"""
import json

def load_results(filename):
    results = []
    try:
        with open(filename) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    except FileNotFoundError:
        return None
    return results

def avg(lst, key):
    if not lst:
        return 0
    return sum(r.get(key, 0) for r in lst) / len(lst)

print("="*80)
print("  COMPARAISON COMPLETE : V1 vs V2 vs V3 vs V4")
print("="*80)

versions = {
    "V1": load_results("nexus_flux_v1_results.jsonl"),
    "V2": load_results("nexus_flux_v2_results.jsonl"),
    "V3": load_results("nexus_flux_v3_results.jsonl"),
    "V4": load_results("nexus_flux_v4_results.jsonl"),
}

print("\n" + "="*80)
print(f"  {'Version':<8} | {'Runs':<5} | {'Conv.':<6} | {'Stab.':<8} | {'H_V drop':<9} | {'Chat':<6} | {'Iter':<5}")
print("="*80)

for name, results in versions.items():
    if not results:
        continue
    
    runs = len(results)
    conv = sum(1 for r in results if r.get('converged', False))
    stab = avg(results, 'stability')
    hv = avg(results, 'h_v_drop')
    chat = avg(results, 'chat_calls')
    iters = avg(results, 'iterations')
    
    conv_rate = f"{conv}/{runs}" if runs > 0 else "N/A"
    
    print(f"  {name:<8} | {runs:<5} | {conv_rate:<6} | {stab:<8.4f} | {hv:<9.4f} | {chat:<6.1f} | {iters:<5.1f}")

print("="*80)

# Analyse détaillée V2 vs V4
print("\nAnalyse V2 (ref) vs V4 (nouveau):")
v2 = versions.get("V2")
v4 = versions.get("V4")

if v2 and v4:
    print(f"  Stabilité   : V2={avg(v2,'stability'):.4f}  V4={avg(v4,'stability'):.4f}  -> {'V4+' if avg(v4,'stability')>avg(v2,'stability') else 'V2+'}")
    print(f"  H_V drop    : V2={avg(v2,'h_v_drop'):.4f}  V4={avg(v4,'h_v_drop'):.4f}  -> {'V4+' if avg(v4,'h_v_drop')>avg(v2,'h_v_drop') else 'V2+'}")
    print(f"  Chat calls  : V2={avg(v2,'chat_calls'):.1f}  V4={avg(v4,'chat_calls'):.1f}  -> {'V4+ éco' if avg(v4,'chat_calls')<avg(v2,'chat_calls') else 'V2+ éco'}")
    print(f"  Convergence : V2={sum(1 for r in v2 if r.get('converged'))}/{len(v2)}  V4={sum(1 for r in v4 if r.get('converged'))}/{len(v4)}")

# Ajouter V4.1 si disponible
v41 = load_results("nexus_flux_v4_optimized_results.jsonl")
if v41:
    print("\n  V4.1     |", len(v41), "runs |", 
          f"{sum(1 for r in v41 if r.get('converged'))}/{len(v41)}", "conv |",
          f"{avg(v41,'stability'):.4f}", "stab |",
          f"{avg(v41,'h_v_drop'):.4f}", "H_V |",
          f"{avg(v41,'chat_calls'):.1f}", "chat |",
          f"{avg(v41,'iterations'):.1f}", "iter")
    
    print("\nAnalyse V2 vs V4.1:")
    print(f"  Stabilité   : V2={avg(v2,'stability'):.4f}  V4.1={avg(v41,'stability'):.4f}")
    print(f"  H_V drop    : V2={avg(v2,'h_v_drop'):.4f}  V4.1={avg(v41,'h_v_drop'):.4f}")
    print(f"  Chat calls  : V2={avg(v2,'chat_calls'):.1f}  V4.1={avg(v41,'chat_calls'):.1f}  -> {'V4.1+ éco' if avg(v41,'chat_calls')<avg(v2,'chat_calls') else 'V2+ éco'}")
    print(f"  Convergence : V2={sum(1 for r in v2 if r.get('converged'))}/{len(v2)}  V4.1={sum(1 for r in v41 if r.get('converged'))}/{len(v41)}")
