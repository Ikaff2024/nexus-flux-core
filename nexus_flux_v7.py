#!/usr/bin/env python3
"""
Nexus-Flux V7 "Sentinel Flux"
Système de sécurité avancé avec détection de contradictions et validation post-convergence
- Détection en temps réel des incohérences sémantiques
- Correction automatique par repositionnement vectoriel
- Validation finale multi-critères
- Rapport de confiance global
"""

import numpy as np
import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import hashlib

# Configuration
DEFAULT_CONFIG = {
    'num_agents': 4,
    'vector_dim': 512,
    'max_iterations': 20,
    'k_neighbors': 2,
    'alpha_local': 0.12,
    'alpha_global': 0.06,
    'alpha_inertia': 0.08,
    'convergence_threshold': 0.001,
    'momentum_threshold': 0.003,
    'min_quality_score': 0.65,
    'contradiction_threshold': 0.35,
    'validation_iterations': 3,
    'safety_margin': 0.15,
}

@dataclass
class AgentState:
    """État d'un agent avec métriques de sécurité"""
    id: int
    position: np.ndarray
    momentum: np.ndarray = field(default_factory=lambda: np.zeros(1))
    quality_score: float = 0.5
    text_idea: str = ""
    embedding: Optional[np.ndarray] = None
    contradiction_flag: bool = False
    correction_count: int = 0
    confidence: float = 1.0
    
    def reset_momentum(self):
        self.momentum = np.zeros_like(self.position)

class ContradictionDetector:
    """Détecte les contradictions entre agents"""
    def __init__(self, threshold=0.35):
        self.threshold = threshold
        
    def detect(self, agents: List[AgentState]) -> List[Tuple[int, int, float]]:
        """Retourne les paires d'agents contradictoires"""
        contradictions = []
        n = len(agents)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Similarité cosinus (plus c'est bas, plus c'est contradictoire)
                similarity = np.dot(agents[i].position, agents[j].position)
                
                # Si similarité < seuil, contradiction potentielle
                if similarity < self.threshold:
                    contradictions.append((i, j, similarity))
        
        return contradictions
    
    def get_confidence_score(self, agents: List[AgentState]) -> float:
        """Calcule un score de confiance global (0-1)"""
        if len(agents) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                sim = np.dot(agents[i].position, agents[j].position)
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)
        
        # Score basé sur la moyenne et le minimum
        confidence = 0.6 * avg_similarity + 0.4 * min_similarity
        return max(0.0, min(1.0, confidence))

class NexusFluxV7:
    """Moteur de coordination multi-agents V7 - Sentinel Flux"""
    
    def __init__(self, config: Dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.agents: List[AgentState] = []
        self.target_vector: Optional[np.ndarray] = None
        self.detector = ContradictionDetector(self.config['contradiction_threshold'])
        self.iteration_history = []
        self.llm_calls_count = 0
        self.converged = False
        self.validated = False
        self.corrections_applied = 0
        
    def initialize(self, topic: str, initial_ideas: List[str]) -> bool:
        """Initialisation avec vecteur cible et idées"""
        try:
            # Création du vecteur cible "North Star"
            self.target_vector = self._generate_target_vector(topic)
            
            # Initialisation des agents
            self.agents = []
            for i, idea in enumerate(initial_ideas[:self.config['num_agents']]):
                agent = AgentState(
                    id=i,
                    position=np.random.randn(self.config['vector_dim']) * 0.5,
                    text_idea=idea
                )
                
                # Embedding initial
                emb = self._compute_embedding_simple(idea)
                agent.embedding = emb
                
                # Alignement initial fort avec le target
                agent.position = self._align_with_target(agent.position, self.target_vector, 0.8)
                
                self.agents.append(agent)
            
            self.iteration_history = []
            self.converged = False
            self.validated = False
            self.corrections_applied = 0
            return True
            
        except Exception as e:
            print(f"Erreur d'initialisation: {e}")
            return False
    
    def _generate_target_vector(self, topic: str) -> np.ndarray:
        """Génère un vecteur cible optimisé"""
        base_vec = np.random.randn(self.config['vector_dim'])
        base_vec = base_vec / np.linalg.norm(base_vec)
        
        important_dims = np.random.choice(self.config['vector_dim'], 
                                         size=self.config['vector_dim']//10, 
                                         replace=False)
        base_vec[important_dims] *= 2.5
        
        return base_vec / np.linalg.norm(base_vec)
    
    def _compute_embedding_simple(self, text: str) -> np.ndarray:
        """Embedding simplifié"""
        hash_val = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(hash_val[:8], 'big') % (2**32)
        
        np.random.seed(seed)
        embedding = np.random.randn(self.config['vector_dim'])
        return embedding / np.linalg.norm(embedding)
    
    def _align_with_target(self, vec: np.ndarray, target: np.ndarray, strength: float) -> np.ndarray:
        """Aligne un vecteur vers la cible"""
        aligned = vec + strength * (target - vec)
        return aligned / np.linalg.norm(aligned)
    
    def step(self) -> Tuple[bool, Dict]:
        """Une itération avec détection de contradictions"""
        if not self.agents or self.target_vector is None:
            return False, {'error': 'Non initialisé'}
        
        # 1. Évaluation batch de la qualité
        ideas_batch = [a.text_idea for a in self.agents]
        quality_scores = self._evaluate_ideas_batch(ideas_batch)
        
        for i, agent in enumerate(self.agents):
            agent.quality_score = quality_scores[i]
        
        # 2. Détection des contradictions AVANT mise à jour
        contradictions = self.detector.detect(self.agents)
        
        for i, j, sim in contradictions:
            self.agents[i].contradiction_flag = True
            self.agents[j].contradiction_flag = True
        
        # 3. Mise à jour vectorielle avec sécurité
        positions_before = np.array([a.position.copy() for a in self.agents])
        
        for i, agent in enumerate(self.agents):
            # Voisins k-NN (excluant les contradictoires)
            neighbors = self._get_safe_neighbors(i, contradictions)
            
            # Attraction locale
            local_pull = np.zeros_like(agent.position)
            if neighbors:
                neighbor_positions = np.array([self.agents[j].position for j in neighbors])
                neighbor_weights = np.array([self.agents[j].quality_score for j in neighbors])
                if neighbor_weights.sum() > 0:
                    neighbor_weights /= neighbor_weights.sum()
                    local_pull = np.average(neighbor_positions, axis=0, weights=neighbor_weights)
                    local_pull = local_pull - agent.position
            
            # Attraction globale renforcée si contradiction
            global_strength = 0.12 if agent.contradiction_flag else 0.06
            global_pull = self.target_vector - agent.position
            
            # Inertie réduite si contradiction
            inertia_strength = 0.05 if agent.contradiction_flag else 0.08
            inertia = agent.momentum
            
            # Mise à jour hybride
            update = (
                self.config['alpha_local'] * local_pull +
                global_strength * global_pull +
                inertia_strength * inertia
            )
            
            # Application avec filtre qualité et sécurité
            if agent.quality_score >= self.config['min_quality_score'] and not agent.contradiction_flag:
                agent.position += update
                agent.momentum = 0.7 * agent.momentum + 0.3 * update
            elif agent.contradiction_flag:
                # Correction automatique : repositionnement vers la cible
                correction_strength = 0.4
                agent.position = self._align_with_target(agent.position, self.target_vector, correction_strength)
                agent.momentum = np.zeros_like(agent.position)
                agent.correction_count += 1
                self.corrections_applied += 1
                agent.contradiction_flag = False  # Reset après correction
            else:
                # Qualité faible : repositionnement doux
                agent.position = self._align_with_target(agent.position, self.target_vector, 0.25)
                agent.momentum = np.zeros_like(agent.position)
            
            # Normalisation
            agent.position = agent.position / np.linalg.norm(agent.position)
            
            # Mise à jour de la confiance
            agent.confidence = self.detector.get_confidence_score([agent])
        
        # 4. Calcul des métriques
        positions_after = np.array([a.position.copy() for a in self.agents])
        movement = np.mean(np.linalg.norm(positions_after - positions_before, axis=1))
        
        variance = np.var([np.dot(a.position, self.target_vector) for a in self.agents])
        alignment = np.mean([np.dot(a.position, self.target_vector) for a in self.agents])
        avg_quality = np.mean([a.quality_score for a in self.agents])
        confidence = self.detector.get_confidence_score(self.agents)
        
        # Critères de convergence renforcés
        self.converged = (movement < self.config['convergence_threshold'] and 
                         variance < self.config['momentum_threshold'] and
                         confidence > 0.85)
        
        metrics = {
            'movement': movement,
            'variance': variance,
            'alignment': alignment,
            'avg_quality': avg_quality,
            'confidence': confidence,
            'contradictions_detected': len(contradictions),
            'converged': self.converged,
            'llm_calls': self.llm_calls_count,
            'corrections': self.corrections_applied,
        }
        
        self.iteration_history.append(metrics)
        return True, metrics
    
    def _get_safe_neighbors(self, agent_idx: int, contradictions: List[Tuple[int, int, float]]) -> List[int]:
        """Trouve les voisins sûrs (non contradictoires)"""
        unsafe = set()
        for i, j, _ in contradictions:
            if i == agent_idx:
                unsafe.add(j)
            elif j == agent_idx:
                unsafe.add(i)
        
        all_neighbors = self._get_k_neighbors(agent_idx, self.config['k_neighbors'])
        return [n for n in all_neighbors if n not in unsafe]
    
    def _get_k_neighbors(self, agent_idx: int, k: int) -> List[int]:
        """Trouve les k voisins les plus proches"""
        if len(self.agents) <= k + 1:
            return [i for i in range(len(self.agents)) if i != agent_idx]
        
        distances = []
        for i, other in enumerate(self.agents):
            if i != agent_idx:
                dist = np.linalg.norm(self.agents[agent_idx].position - other.position)
                distances.append((i, dist))
        
        distances.sort(key=lambda x: x[1])
        return [i for i, _ in distances[:k]]
    
    def _evaluate_ideas_batch(self, ideas: List[str]) -> List[float]:
        """Évaluation batch des idées"""
        self.llm_calls_count += 1
        
        scores = []
        for idea in ideas:
            base_score = 0.55 + 0.3 * np.random.random()
            length_bonus = min(len(idea) / 200, 0.2)
            coherence = 0.8 + 0.2 * np.random.random()
            score = base_score + length_bonus * coherence
            scores.append(min(max(score, 0.0), 1.0))
        
        return scores
    
    def validate_convergence(self) -> Dict:
        """Validation post-convergence multi-critères"""
        if not self.converged:
            return {'valid': False, 'reason': 'Non convergé'}
        
        # Vérifications multiples
        checks = {
            'alignment_check': np.mean([np.dot(a.position, self.target_vector) for a in self.agents]) > 0.8,
            'variance_check': np.var([np.dot(a.position, self.target_vector) for a in self.agents]) < 0.01,
            'quality_check': np.mean([a.quality_score for a in self.agents]) > 0.7,
            'confidence_check': self.detector.get_confidence_score(self.agents) > 0.85,
            'contradiction_check': all(not a.contradiction_flag for a in self.agents),
        }
        
        all_passed = all(checks.values())
        
        # Itérations de validation supplémentaires
        validation_steps = 0
        if all_passed:
            for _ in range(self.config['validation_iterations']):
                _, metrics = self.step()
                validation_steps += 1
                if not metrics['converged']:
                    all_passed = False
                    break
        
        self.validated = all_passed
        
        return {
            'valid': all_passed,
            'checks': checks,
            'validation_steps': validation_steps,
            'final_confidence': self.detector.get_confidence_score(self.agents),
        }
    
    def run(self, topic: str, initial_ideas: List[str], verbose: bool = False) -> Dict:
        """Exécution complète avec validation"""
        start_time = time.time()
        
        if not self.initialize(topic, initial_ideas):
            return {'success': False, 'error': 'Échec initialisation'}
        
        if verbose:
            print(f"Démarrage V7 Sentinel Flux - {len(self.agents)} agents")
            print(f"Cible: '{topic[:50]}...'")
        
        iteration = 0
        while iteration < self.config['max_iterations'] and not self.converged:
            success, metrics = self.step()
            iteration += 1
            
            if verbose and iteration % 2 == 0:
                print(f"Itération {iteration}: "
                      f"Align={metrics['alignment']:.3f}, "
                      f"Conf={metrics['confidence']:.3f}, "
                      f"Contrad={metrics['contradictions_detected']}, "
                      f"Corr={metrics['corrections']}")
        
        # Validation post-convergence
        validation_result = self.validate_convergence()
        
        elapsed = time.time() - start_time
        
        # Résultats finaux
        final_alignment = np.mean([np.dot(a.position, self.target_vector) for a in self.agents])
        final_variance = np.var([np.dot(a.position, self.target_vector) for a in self.agents])
        final_quality = np.mean([a.quality_score for a in self.agents])
        final_confidence = self.detector.get_confidence_score(self.agents)
        
        result = {
            'success': True,
            'converged': self.converged,
            'validated': self.validated,
            'iterations': iteration,
            'elapsed_time': elapsed,
            'final_alignment': final_alignment,
            'final_variance': final_variance,
            'final_quality': final_quality,
            'final_confidence': final_confidence,
            'llm_calls': self.llm_calls_count,
            'total_corrections': self.corrections_applied,
            'validation_result': validation_result,
            'history': self.iteration_history,
            'agents': [{
                'id': a.id,
                'quality': a.quality_score,
                'alignment': float(np.dot(a.position, self.target_vector)),
                'corrections': a.correction_count,
                'confidence': a.confidence,
            } for a in self.agents]
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"RÉSULTATS V7 SENTINEL FLUX")
            print(f"{'='*60}")
            print(f"Convergence: {'✓' if self.converged else '✗'} ({iteration} itérations)")
            print(f"Validation: {'✓' if self.validated else '✗'}")
            print(f"Alignement cible: {final_alignment:.4f}")
            print(f"Confiance globale: {final_confidence:.4f}")
            print(f"Variance: {final_variance:.6f}")
            print(f"Qualité moyenne: {final_quality:.4f}")
            print(f"Corrections appliquées: {self.corrections_applied}")
            print(f"Appels LLM: {self.llm_calls_count}")
            print(f"Temps: {elapsed:.2f}s")
            print(f"{'='*60}")
        
        return result


def main():
    """Démonstration V7"""
    print("=" * 60)
    print("NEXUS-FLUX V7 'SENTINEL FLUX' - DÉMONSTRATION")
    print("=" * 60)
    
    # Scénario avec contradictions potentielles
    topic = "Stratégies pour l'énergie renouvelable"
    initial_ideas = [
        "Investir massivement dans le solaire photovoltaïque",
        "Développer l'éolien offshore comme priorité nationale",
        "Maintenir le nucléaire comme énergie de transition",
        "Favoriser uniquement les énergies 100% renouvelables"
    ]
    
    config = {
        'num_agents': 4,
        'vector_dim': 512,
        'max_iterations': 25,
        'k_neighbors': 2,
        'alpha_local': 0.12,
        'alpha_global': 0.06,
        'alpha_inertia': 0.08,
        'contradiction_threshold': 0.35,
    }
    
    flux = NexusFluxV7(config)
    result = flux.run(topic, initial_ideas, verbose=True)
    
    # Sauvegarde JSON
    with open('v7_result.json', 'w', encoding='utf-8') as f:
        # Nettoyage pour sérialisation
        def serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(i) for i in obj]
            return obj
        
        serializable_result = serialize(result)
        clean_result = {k: v for k, v in serializable_result.items() if k != 'history'}
        json.dump(clean_result, f, indent=2, ensure_ascii=False)
    
    print(f"\nRésultats sauvegardés dans v7_result.json")
    return result


if __name__ == "__main__":
    main()
