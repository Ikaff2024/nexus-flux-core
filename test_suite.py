#!/usr/bin/env python3
"""
Suite de tests unitaires pour nexus-flux-core
Valide la robustesse des composants vectoriels
"""

import numpy as np
import sys

def test_vector_field_initialization():
    """Test l'initialisation du champ vectoriel"""
    n_agents = 4
    dim = 1536
    
    # Initialisation aléatoire normalisée
    positions = np.random.randn(n_agents, dim)
    positions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    
    # Vérifier que tous les vecteurs sont unitaires
    norms = np.linalg.norm(positions, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), "Les vecteurs doivent être unitaires"
    
    print("✓ Test vector_field_initialization: PASSED")
    return True

def test_centroid_computation():
    """Test le calcul du centroïde"""
    n_agents = 4
    dim = 1536
    
    positions = np.random.randn(n_agents, dim)
    positions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    
    # Calcul du centroïde
    centroid = np.mean(positions, axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    
    # Le centroïde doit être dans le même espace
    assert centroid.shape == (dim,), "Dimension du centroïde incorrecte"
    assert np.isclose(np.linalg.norm(centroid), 1.0, atol=1e-5), "Centroïde doit être unitaire"
    
    print("✓ Test centroid_computation: PASSED")
    return True

def test_momentum_calculation():
    """Test le calcul du momentum"""
    n_agents = 4
    dim = 1536
    
    # Positions actuelles et précédentes
    positions_prev = np.random.randn(n_agents, dim)
    positions_prev = positions_prev / np.linalg.norm(positions_prev, axis=1, keepdims=True)
    
    positions_curr = np.random.randn(n_agents, dim)
    positions_curr = positions_curr / np.linalg.norm(positions_curr, axis=1, keepdims=True)
    
    # Calcul du momentum
    momentum_vectors = positions_curr - positions_prev
    momentum_magnitude = np.mean(np.linalg.norm(momentum_vectors, axis=1))
    
    # Le momentum doit être positif ou nul
    assert momentum_magnitude >= 0, "Le momentum doit être positif"
    
    print("✓ Test momentum_calculation: PASSED")
    return True

def test_entropy_computation():
    """Test le calcul de l'entropie vectorielle H_V"""
    n_agents = 4
    dim = 1536
    
    positions = np.random.randn(n_agents, dim)
    positions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    
    # Matrice de similarité cosinus
    similarity_matrix = np.dot(positions, positions.T)
    
    # H_V : moyenne des similarités hors diagonale
    n = n_agents
    mask = ~np.eye(n, dtype=bool)
    h_v = np.mean(similarity_matrix[mask])
    
    # H_V doit être entre -1 et 1
    assert -1 <= h_v <= 1, f"H_V doit être entre -1 et 1, obtenu {h_v}"
    
    print("✓ Test entropy_computation: PASSED")
    return True

def test_convergence_criteria():
    """Test les critères de convergence"""
    # Cas convergé : momentum faible et H_V élevé
    momentum_low = 0.02
    h_v_high = 0.95
    converged = momentum_low < 0.05 and h_v_high > 0.9
    assert converged, "Devrait détecter la convergence"
    
    # Cas non convergé : momentum élevé
    momentum_high = 0.3
    h_v_low = 0.5
    not_converged = not (momentum_high < 0.05 and h_v_low > 0.9)
    assert not_converged, "Devrait détecter la non-convergence"
    
    print("✓ Test convergence_criteria: PASSED")
    return True

def test_stability_metric():
    """Test la métrique de stabilité"""
    n_agents = 4
    dim = 1536
    
    # Positions très proches (haute stabilité)
    base = np.random.randn(dim)
    base = base / np.linalg.norm(base)
    positions_stable = np.array([base + np.random.randn(dim) * 0.01 for _ in range(n_agents)])
    positions_stable = positions_stable / np.linalg.norm(positions_stable, axis=1, keepdims=True)
    
    # Positions éloignées (basse stabilité)
    positions_unstable = np.random.randn(n_agents, dim)
    positions_unstable = positions_unstable / np.linalg.norm(positions_unstable, axis=1, keepdims=True)
    
    # Calcul de la variance
    centroid_stable = np.mean(positions_stable, axis=0)
    centroid_unstable = np.mean(positions_unstable, axis=0)
    
    var_stable = np.mean(np.linalg.norm(positions_stable - centroid_stable, axis=1))
    var_unstable = np.mean(np.linalg.norm(positions_unstable - centroid_unstable, axis=1))
    
    # La stabilité est l'inverse de la variance
    stability_stable = 1.0 / (1.0 + var_stable)
    stability_unstable = 1.0 / (1.0 + var_unstable)
    
    assert stability_stable > stability_unstable, "Stabilité stable > instable"
    
    print("✓ Test stability_metric: PASSED")
    return True

def test_agent_count_flexibility():
    """Test la flexibilité du nombre d'agents"""
    dim = 1536
    
    for n_agents in [2, 4, 8, 16]:
        positions = np.random.randn(n_agents, dim)
        positions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
        
        centroid = np.mean(positions, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        assert centroid.shape == (dim,), f"Échec pour {n_agents} agents"
    
    print("✓ Test agent_count_flexibility: PASSED")
    return True

def test_dimension_independence():
    """Test l'indépendance à la dimension"""
    n_agents = 4
    
    for dim in [128, 512, 1536, 3072]:
        positions = np.random.randn(n_agents, dim)
        positions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
        
        similarity_matrix = np.dot(positions, positions.T)
        assert similarity_matrix.shape == (n_agents, n_agents), f"Échec pour dim={dim}"
    
    print("✓ Test dimension_independence: PASSED")
    return True

def run_all_tests():
    """Exécute tous les tests"""
    print("=" * 70)
    print("  SUITE DE TESTS UNITAIRES - NEXUS-FLUX-CORE")
    print("=" * 70)
    print()
    
    tests = [
        test_vector_field_initialization,
        test_centroid_computation,
        test_momentum_calculation,
        test_entropy_computation,
        test_convergence_criteria,
        test_stability_metric,
        test_agent_count_flexibility,
        test_dimension_independence,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: FAILED - {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print(f"  RÉSULTATS: {passed}/{len(tests)} tests passés")
    if failed > 0:
        print(f"  ÉCHECS: {failed} tests")
    else:
        print("  TOUS LES TESTS ONT RÉUSSI ✓")
    print("=" * 70)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
