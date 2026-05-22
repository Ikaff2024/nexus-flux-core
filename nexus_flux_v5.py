#!/usr/bin/env python3
"""
Nexus-Flux V5: Guided Flux Architecture
----------------------------------------
Améliorations clés :
1. Vecteur Cible ("North Star") : Guide la convergence vers un optimum sémantique.
2. Topologie Dynamique (k-NN) : Les agents interagissent avec leurs voisins proches.
3. Formule Hybride : Combinaison d'attraction locale, globale et d'inertie.
4. Efficacité : Réduction des appels LLM via une initialisation ciblée.
"""

import numpy as np
import time
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import hashlib

# --- Configuration ---
@dataclass
class Config:
    num_agents: int = 4
    vector_dim: int = 1536  # Dimension des embeddings OpenAI
    max_iterations: int = 15
    k_neighbors: int = 2    # Nombre de voisins pour la topologie dynamique
    
    # Hyperparamètres de convergence guidée
    alpha_local: float = 0.3      # Poids de l'attraction vers les voisins
    alpha_global: float = 0.4     # Poids de l'attraction vers la cible (North Star)
    momentum_factor: float = 0.2  # Inertie du mouvement précédent
    noise_decay: float = 0.9      # Décroissance du bruit d'exploration
    
    # Seuils
    convergence_threshold: float = 0.05
    min_momentum_drop: float = 0.1
    
    # API
    use_mock_api: bool = True  # Mettre à False pour utiliser OpenAI réel
    model_name: str = "text-embedding-3-small"

config = Config()

# --- Mock API pour démonstration (Remplaçable par OpenAI réel) ---
class MockOpenAI:
    def __init__(self):
        self.cache = {}

    def get_embedding(self, text: str) -> np.ndarray:
        # Cache simple basé sur le hachage du texte
        h = hashlib.md5(text.encode()).hexdigest()
        if h in self.cache:
            return self.cache[h]
        
        # Simulation d'un embedding cohérent (pseudo-aléatoire déterministe)
        np.random.seed(int(h[:8], 16))
        vec = np.random.randn(config.vector_dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        self.cache[h] = vec
        return vec

    def generate_idea(self, topic: str, context_vector: np.ndarray) -> str:
        # Simulation d'une idée basée sur le contexte
        ideas = [
            f"Système décentralisé pour {topic}",
            f"Approche hybride IA/Humain pour {topic}",
            f"Optimisation en temps réel de {topic}",
            f"Protocole de sécurité avancé pour {topic}",
            f"Interface intuitive dédiée à {topic}"
        ]
        # Choix pseudo-aléatoire basé sur le contexte
        idx = int(np.sum(context_vector[:10]) * 100) % len(ideas)
        return ideas[idx]

    def evaluate_quality(self, idea: str, target_vector: np.ndarray) -> float:
        # Simulation d'un score de qualité (cosine similarity avec la cible)
        idea_vec = self.get_embedding(idea)
        sim = np.dot(idea_vec, target_vector)
        # Mapping de [-1, 1] vers [0, 1] avec un peu de bruit
        return max(0.0, min(1.0, (sim + 1) / 2 + np.random.normal(0, 0.05)))

api = MockOpenAI()

# --- Structures de Données ---
@dataclass
class Agent:
    id: int
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    current_idea: str = ""
    quality_score: float = 0.0

@dataclass
class NexusState:
    agents: List[Agent]
    target_vector: np.ndarray  # Le "North Star"
    iteration: int = 0
    history: List[Dict] = field(default_factory=list)

# --- Moteur de Convergence Guidée ---
class GuidedFluxEngine:
    def __init__(self, topic: str):
        self.topic = topic
        self.state: Optional[NexusState] = None
        
    def initialize(self) -> NexusState:
        print(f"\n🚀 Initialisation Nexus-Flux V5 pour : '{self.topic}'")
        print(f"   - Agents: {config.num_agents}")
        print(f"   - Dimension: {config.vector_dim}")
        print(f"   - Mode: Guided Flux (Target + k-NN)")

        # 1. Génération du vecteur cible ("North Star")
        # On demande au LLM de définir l'idéal théorique pour le sujet
        ideal_prompt = f"Définis le concept idéal et parfait pour : {self.topic}. Un seul paragraphe dense."
        if config.use_mock_api:
            target_vec = api.get_embedding(ideal_prompt)
        else:
            # Appel réel à implémenter ici
            target_vec = api.get_embedding(ideal_prompt)
        
        print(f"   - Vecteur Cible généré (North Star).")

        # 2. Initialisation des agents
        agents = []
        for i in range(config.num_agents):
            # Chaque agent part d'une idée aléatoire liée au sujet
            prompt = f"Idée initiale aléatoire pour {topic} (angle {i}): "
            idea_text = api.generate_idea(self.topic, target_vec + np.random.normal(0, 0.5, config.vector_dim))
            pos = api.get_embedding(idea_text)
            
            # Normalisation
            pos /= np.linalg.norm(pos)
            
            agents.append(Agent(id=i, position=pos, velocity=np.zeros(config.vector_dim), current_idea=idea_text))
            
        self.state = NexusState(agents=agents, target_vector=target_vec)
        return self.state

    def get_neighbors(self, agent_idx: int) -> List[int]:
        """Trouve les k voisins les plus proches (cosine distance)"""
        if len(self.state.agents) <= config.k_neighbors:
            return [i for i in range(len(self.state.agents)) if i != agent_idx]
        
        current_pos = self.state.agents[agent_idx].position
        distances = []
        for i, other in enumerate(self.state.agents):
            if i == agent_idx: continue
            dist = 1.0 - np.dot(current_pos, other.position) # Cosine distance
            distances.append((i, dist))
        
        distances.sort(key=lambda x: x[1])
        return [x[0] for x in distances[:config.k_neighbors]]

    def step(self) -> bool:
        """Une itération de mise à jour vectorielle guidée"""
        if not self.state: return False
        
        t = self.state.iteration
        agents = self.state.agents
        target = self.state.target_vector
        
        # Décroissance des paramètres pour stabiliser
        current_alpha_local = config.alpha_local * (0.95 ** t)
        current_alpha_global = config.alpha_global * (1.0 + 0.1 * t) # On renforce l'attraction globale avec le temps
        current_momentum = config.momentum_factor
        noise_scale = 0.05 * (config.noise_decay ** t)

        new_positions = []
        new_velocities = []

        for i, agent in enumerate(agents):
            # 1. Attraction Locale (Voisins)
            neighbors_idx = self.get_neighbors(i)
            local_center = np.mean([agents[j].position for j in neighbors_idx], axis=0)
            local_center /= np.linalg.norm(local_center)
            force_local = local_center - agent.position
            
            # 2. Attraction Globale (North Star)
            force_global = target - agent.position
            
            # 3. Calcul de la nouvelle vélocité (Physique simplifiée)
            # v_new = momentum * v_old + alpha_local * F_local + alpha_global * F_global + Noise
            acceleration = (current_alpha_local * force_local) + (current_alpha_global * force_global)
            noise = np.random.normal(0, noise_scale, config.vector_dim)
            
            new_vel = (current_momentum * agent.velocity) + acceleration + noise
            new_pos = agent.position + new_vel
            
            # Normalisation (Projection sur hypersphère unitaire)
            new_pos /= np.linalg.norm(new_pos)
            
            new_positions.append(new_pos)
            new_velocities.append(new_vel)
            
            # Mise à jour de l'idée textuelle (si déplacement significatif)
            displacement = 1.0 - np.dot(agent.position, new_pos)
            if displacement > 0.02 or t == 0:
                # On régénère l'idée pour qu'elle corresponde au nouveau vecteur
                # Dans une vraie implé, on ferait un appel LLM "inverse" ou une recherche dans une base
                # Ici on simule un raffinement
                agent.current_idea = api.generate_idea(self.topic, new_pos)
                agent.quality_score = api.evaluate_quality(agent.current_idea, target)
            
            agent.position = new_pos
            agent.velocity = new_vel

        # Mise à jour de l'état
        for i, agent in enumerate(agents):
            agent.position = new_positions[i]
            agent.velocity = new_velocities[i]
            
        self.state.iteration += 1
        return True

    def compute_metrics(self) -> Dict:
        """Calcule les métriques de convergence"""
        agents = self.state.agents
        target = self.state.target_vector
        
        # 1. Cohésion (Variance interne)
        positions = np.array([a.position for a in agents])
        centroid = np.mean(positions, axis=0)
        centroid /= np.linalg.norm(centroid)
        variance = np.mean([1.0 - np.dot(p, centroid) for p in positions])
        
        # 2. Alignement avec la Cible (Qualité Globale)
        alignment = np.mean([np.dot(a.position, target) for a in agents])
        
        # 3. Momentum moyen
        momentum_mag = np.mean([np.linalg.norm(a.velocity) for a in agents])
        
        # 4. Qualité moyenne des idées
        avg_quality = np.mean([a.quality_score for a in agents])
        
        return {
            "variance": variance,
            "alignment": alignment,
            "momentum": momentum_mag,
            "avg_quality": avg_quality,
            "iteration": self.state.iteration
        }

    def run(self) -> Dict:
        """Exécute la simulation complète"""
        self.initialize()
        
        print("\n⏳ Démarrage de la convergence guidée...")
        start_time = time.time()
        
        metrics_history = []
        
        for _ in range(config.max_iterations):
            self.step()
            metrics = self.compute_metrics()
            metrics_history.append(metrics)
            
            # Affichage progressif
            status = (
                f"It {metrics['iteration']:02d} | "
                f"Var: {metrics['variance']:.4f} | "
                f"Align: {metrics['alignment']:.4f} | "
                f"Qual: {metrics['avg_quality']:.2f} | "
                f"Mom: {metrics['momentum']:.4f}"
            )
            print(status)
            
            # Critère d'arrêt anticipé
            if metrics['variance'] < config.convergence_threshold and metrics['momentum'] < 0.05:
                print("   --> Convergence atteinte !")
                break
                
        elapsed = time.time() - start_time
        
        # Rapport final
        final_metrics = metrics_history[-1]
        report = {
            "status": "success" if final_metrics['variance'] < 0.1 else "partial",
            "final_variance": final_metrics['variance'],
            "final_alignment": final_metrics['alignment'],
            "final_quality": final_metrics['avg_quality'],
            "iterations": final_metrics['iteration'],
            "time_elapsed": elapsed,
            "history": metrics_history,
            "final_ideas": [a.current_idea for a in self.state.agents]
        }
        
        self.print_report(report)
        return report

    def print_report(self, report: Dict):
        print("\n" + "="*50)
        print("📊 RAPPORT FINAL NEXUS-FLUX V5")
        print("="*50)
        print(f"Statut       : {report['status'].upper()}")
        print(f"Itérations   : {report['iterations']}")
        print(f"Temps        : {report['time_elapsed']:.2f}s")
        print(f"Cohésion     : {1.0 - report['final_variance']:.4f} (Plus haut est mieux)")
        print(f"Alignement   : {report['final_alignment']:.4f} (Proximité avec l'idéal)")
        print(f"Qualité Moy. : {report['final_quality']:.4f}")
        print("\n💡 Idées Convergentes :")
        for i, idea in enumerate(report['final_ideas']):
            print(f"   [{i}] {idea}")
        print("="*50)

# --- Point d'entrée ---
if __name__ == "__main__":
    topic = "Système de transport urbain durable et autonome"
    engine = GuidedFluxEngine(topic)
    result = engine.run()
    
    # Sauvegarde optionnelle des résultats
    with open("v5_results.json", "w") as f:
        # Nettoyage pour sérialisation JSON (retrait des numpy arrays bruts)
        clean_result = {k: v for k, v in result.items() if k != 'history'}
        clean_result['history_summary'] = [
            {k: float(v[k]) for k in ['variance', 'alignment', 'momentum', 'avg_quality']} 
            for v in result['history']
        ]
        # Conversion explicite des types numpy restants
        for key in ['final_variance', 'final_alignment', 'final_quality', 'time_elapsed']:
            if key in clean_result:
                clean_result[key] = float(clean_result[key])
        json.dump(clean_result, f, indent=2)
    print("\n✅ Résultats sauvegardés dans v5_results.json")
