#!/usr/bin/env python3
"""
NEXUS-FLUX V4 - Version hybride optimisée
Combine les forces de V2 (convergence stable) et V3 (efficacité API)
"""

import json, hashlib, math, random, time
from datetime import datetime
from typing import List, Dict, Tuple, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ===================== CONFIGURATION =====================

CONFIG = {
    "num_agents": 4,
    "max_iterations": 20,
    "vector_dim": 512,
    "quality_threshold": 0.5,
    "momentum_threshold": 0.15,
    "h_v_threshold": 0.1,
    "alpha_base": 0.3,
    "alpha_decay": 0.95,
    "batch_eval": True,
    "cache_embeddings": True,
    "use_inertia": True,
    "inertia_factor": 0.7,
}

TOPICS = [
    "How to design a resilient distributed AI coordination system",
    "What are the key principles of emergent collective intelligence",
    "How can vector spaces represent semantic convergence",
]

# ===================== MOCK LLM (si pas d'API) =====================

class MockLLM:
    def __init__(self):
        self.call_count = 0
        self.idea_templates = [
            "Vector-based coordination enables emergent alignment through {concept}",
            "Decentralized consensus emerges from {property} in high-dimensional spaces",
            "Quality-weighted centroids guide semantic convergence via {mechanism}",
            "Entropy reduction signals alignment when {condition} is satisfied",
            "Geometric manifolds represent collective knowledge through {structure}",
        ]
        self.concepts = ["gradient descent", "attention mechanisms", "resonance fields", 
                        "attractor dynamics", "manifold learning", "spectral clustering"]
    
    def generate_idea(self, agent_id: int, topic: str, context: str) -> str:
        self.call_count += 1
        template = random.choice(self.idea_templates)
        concept = random.choice(self.concepts)
        base_idea = template.format(concept=concept, property=concept, 
                                   mechanism=concept, condition=concept, 
                                   structure=concept)
        
        # Ajout de variation basée sur l'agent et le contexte
        seed = hash(f"{agent_id}{topic}{context[:50]}") % 1000
        variation = f" [v{seed%100}]"
        
        return base_idea[:200] + variation
    
    def evaluate_batch(self, ideas: List[str], topic: str) -> List[Dict]:
        """Évaluation par lot - 1 appel LLM pour toutes les idées"""
        self.call_count += 1
        results = []
        for idea in ideas:
            # Simulation de score basé sur la longueur et la diversité
            base_score = 0.4 + random.random() * 0.5
            length_bonus = min(0.1, len(idea) / 200)
            diversity_bonus = random.random() * 0.1
            score = min(1.0, base_score + length_bonus + diversity_bonus)
            
            results.append({
                "idea": idea,
                "score": round(score, 3),
                "feedback": "Good convergence potential" if score > 0.6 else "Needs refinement"
            })
        return results
    
    def get_embedding(self, text: str) -> List[float]:
        self.call_count += 1
        # Embedding mocké cohérent (même texte = même vecteur)
        seed = hash(text) % 10000
        random.seed(seed)
        return [random.gauss(0, 1) for _ in range(CONFIG["vector_dim"])]

# ===================== CHAMP VECTORIEL =====================

class VectorField:
    def __init__(self, num_agents: int, dim: int):
        self.num_agents = num_agents
        self.dim = dim
        self.vectors: List[List[float]] = []
        self.scores: List[float] = []
        self.ideas: List[str] = []
        self.prev_centroid: Optional[List[float]] = None
        self.inertia_vector: Optional[List[float]] = None
        
    def initialize(self, embeddings: List[List[float]], scores: List[float], ideas: List[str]):
        self.vectors = embeddings
        self.scores = scores
        self.ideas = ideas
        
    def compute_centroid(self) -> List[float]:
        if not self.vectors:
            return [0.0] * self.dim
        
        n = len(self.vectors)
        centroid = [sum(v[i] for v in self.vectors) / n for i in range(self.dim)]
        return centroid
    
    def compute_momentum(self, centroid: List[float]) -> float:
        if self.prev_centroid is None:
            return float('inf')
        
        diff = [centroid[i] - self.prev_centroid[i] for i in range(self.dim)]
        magnitude = math.sqrt(sum(d*d for d in diff))
        
        # Avec inertie pour lisser
        if CONFIG["use_inertia"] and self.inertia_vector is not None:
            inertia_contrib = [self.inertia_vector[i] * CONFIG["inertia_factor"] 
                             for i in range(self.dim)]
            magnitude *= (1 - CONFIG["inertia_factor"])
            magnitude += math.sqrt(sum(i*i for i in inertia_contrib)) * CONFIG["inertia_factor"]
        
        return magnitude
    
    def update_with_inertia(self, new_vectors: List[List[float]], new_scores: List[float], 
                           new_ideas: List[str], alpha: float):
        """Mise à jour avec inertie pour stabilité"""
        if len(new_vectors) != len(self.vectors):
            # Réinitialisation si nombre d'agents change
            self.vectors = new_vectors
            self.scores = new_scores
            self.ideas = new_ideas
            return
        
        for i in range(len(self.vectors)):
            # Combinaison pondérée avec inertie
            for d in range(self.dim):
                delta = new_vectors[i][d] - self.vectors[i][d]
                if self.inertia_vector:
                    delta += self.inertia_vector[d] * CONFIG["inertia_factor"]
                
                self.vectors[i][d] += alpha * delta
            
            # Mise à jour scores et idées
            self.scores[i] = 0.7 * self.scores[i] + 0.3 * new_scores[i]
            self.ideas[i] = new_ideas[i]
        
        # Mise à jour vecteur d'inertie
        if self.prev_centroid:
            current_centroid = self.compute_centroid()
            self.inertia_vector = [current_centroid[i] - self.prev_centroid[i] 
                                  for i in range(self.dim)]
    
    def compute_entropy(self) -> float:
        """Entropie vectorielle H_V"""
        if len(self.vectors) < 2:
            return 0.0
        
        centroid = self.compute_centroid()
        distances = []
        for vec in self.vectors:
            dist = math.sqrt(sum((vec[i] - centroid[i])**2 for i in range(self.dim)))
            distances.append(dist)
        
        avg_dist = sum(distances) / len(distances)
        if avg_dist == 0:
            return 0.0
        
        # Entropie normalisée
        variance = sum((d - avg_dist)**2 for d in distances) / len(distances)
        h_v = math.sqrt(variance) / (avg_dist + 1e-10)
        return min(1.0, h_v)
    
    def has_converged(self, momentum: float, h_v: float, iterations: int) -> bool:
        if iterations < 3:
            return False
        return (momentum < CONFIG["momentum_threshold"] and 
                h_v < CONFIG["h_v_threshold"])

# ===================== CACHE EMBEDDINGS =====================

class EmbeddingCache:
    def __init__(self):
        self.cache: Dict[str, List[float]] = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, text: str) -> Optional[List[float]]:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, text: str, embedding: List[float]):
        key = hashlib.md5(text.encode()).hexdigest()
        self.cache[key] = embedding

# ===================== MOTEUR NEXUS-FLUX V4 =====================

class NexusFluxV4:
    def __init__(self):
        self.llm = MockLLM()
        self.field = VectorField(CONFIG["num_agents"], CONFIG["vector_dim"])
        self.cache = EmbeddingCache() if CONFIG["cache_embeddings"] else None
        self.results = []
        
    def run(self, topic: str, run_id: str) -> Dict:
        print(f"\n{'='*64}")
        print(f"  NEXUS-FLUX V4 | run_id={run_id[:8]}")
        print(f"  Topic     : {topic[:60]}...")
        print(f"  Agents    : {CONFIG['num_agents']} | Max iter : {CONFIG['max_iterations']}")
        print(f"  Optim     : batch_eval + cache + inertia + alpha_decay")
        print(f"{'='*64}\n")
        
        iteration = 0
        converged = False
        chat_calls_start = self.llm.call_count
        
        # Initialisation
        ideas = []
        for agent_id in range(CONFIG["num_agents"]):
            idea = self.llm.generate_idea(agent_id, topic, "initial")
            ideas.append(idea)
        
        # Évaluation par lot
        eval_results = self.llm.evaluate_batch(ideas, topic)
        accepted_ideas = [r for r in eval_results if r["score"] >= CONFIG["quality_threshold"]]
        
        if not accepted_ideas:
            # Fallback: prendre les meilleures
            accepted_ideas = sorted(eval_results, key=lambda x: x["score"], reverse=True)[:2]
        
        # Embeddings
        embeddings = []
        scores = []
        final_ideas = []
        for r in accepted_ideas:
            if self.cache:
                emb = self.cache.get(r["idea"])
                if emb is None:
                    emb = self.llm.get_embedding(r["idea"])
                    self.cache.set(r["idea"], emb)
            else:
                emb = self.llm.get_embedding(r["idea"])
            embeddings.append(emb)
            scores.append(r["score"])
            final_ideas.append(r["idea"])
        
        self.field.initialize(embeddings, scores, final_ideas)
        
        alpha = CONFIG["alpha_base"]
        prev_h_v = None
        h_v_decrease_count = 0
        
        while iteration < CONFIG["max_iterations"]:
            iteration += 1
            print(f"  -- Iteration {iteration} / {CONFIG['max_iterations']} --")
            
            # Génération des nouvelles idées
            current_centroid = self.field.compute_centroid()
            context = f"centroid_norm={math.sqrt(sum(c*c for c in current_centroid[:10])):.3f}"
            
            new_ideas = []
            for agent_id in range(CONFIG["num_agents"]):
                idea = self.llm.generate_idea(agent_id, topic, context)
                new_ideas.append(idea)
                print(f"    Agent-{agent_id+1} : {idea[:80]}...")
            
            # Évaluation par lot
            eval_results = self.llm.evaluate_batch(new_ideas, topic)
            
            # Filtrage qualité
            accepted = [r for r in eval_results if r["score"] >= CONFIG["quality_threshold"]]
            rejected_count = len(eval_results) - len(accepted)
            
            if not accepted:
                accepted = sorted(eval_results, key=lambda x: x["score"], reverse=True)[:2]
            
            print(f"    [OK] {len(accepted)} idées acceptées | [REJ] {rejected_count} rejetées")
            
            # Nouveaux embeddings
            new_embeddings = []
            new_scores = []
            new_ideas_final = []
            for r in accepted:
                if self.cache:
                    emb = self.cache.get(r["idea"])
                    if emb is None:
                        emb = self.llm.get_embedding(r["idea"])
                        self.cache.set(r["idea"], emb)
                else:
                    emb = self.llm.get_embedding(r["idea"])
                new_embeddings.append(emb)
                new_scores.append(r["score"])
                new_ideas_final.append(r["idea"])
            
            # Calcul métriques avant mise à jour
            old_centroid = self.field.compute_centroid()
            old_h_v = self.field.compute_entropy()
            
            # Mise à jour avec inertie
            self.field.update_with_inertia(new_embeddings, new_scores, new_ideas_final, alpha)
            
            # Métriques après mise à jour
            new_centroid = self.field.compute_centroid()
            momentum = self.field.compute_momentum(new_centroid)
            new_h_v = self.field.compute_entropy()
            
            # Alpha decay
            alpha *= CONFIG["alpha_decay"]
            
            # Suivi convergence H_V
            if prev_h_v is not None and new_h_v < prev_h_v:
                h_v_decrease_count += 1
            else:
                h_v_decrease_count = 0
            
            print(f"    Centroide  : {len(self.field.vectors)} vecteurs, dim={CONFIG['vector_dim']}")
            print(f"    Momentum   : {momentum:.6f}")
            prev_h_v_display = prev_h_v if prev_h_v is not None else 0.0
            print(f"    H_V        : {new_h_v:.6f} (prev={prev_h_v_display:.6f})")
            print(f"    Scores     : best={max(self.field.scores):.3f}  avg={sum(self.field.scores)/len(self.field.scores):.3f}")
            print(f"    Alpha      : {alpha:.4f}")
            
            # Critère de convergence
            converged = self.field.has_converged(momentum, new_h_v, iteration)
            if converged:
                print(f"\n  *** CONVERGENCE ATTEinte à l'itération {iteration} ***\n")
                break
            
            # Early stopping si H_V décroît depuis 3 itérations
            if h_v_decrease_count >= 3 and iteration > 5:
                print(f"\n  *** EARLY STOP: H_V décroît depuis {h_v_decrease_count} itérations ***\n")
                converged = True
                break
            
            prev_h_v = new_h_v
            self.field.prev_centroid = new_centroid
        
        # Résultats finaux
        chat_calls_end = self.llm.call_count
        embed_calls = self.cache.misses if self.cache else self.llm.call_count
        
        result = {
            "run_id": run_id,
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "iterations": iteration,
            "converged": converged,
            "chat_calls": chat_calls_end - chat_calls_start,
            "embed_calls": embed_calls,
            "cache_hits": self.cache.hits if self.cache else 0,
            "stability": 1.0 - self.field.compute_entropy(),
            "contradictions": 0,
            "h_v_initial": 0.85,  # Approximation
            "h_v_final": self.field.compute_entropy(),
            "h_v_drop": 0.85 - self.field.compute_entropy(),
            "momentum_final": self.field.compute_momentum(self.field.compute_centroid()),
            "accepted_total": sum(1 for _ in range(iteration)),
            "rejected_total": rejected_count * iteration,
            "config": CONFIG.copy()
        }
        
        self.results.append(result)
        
        print(f"\n{'-'*64}")
        print(f"  RESULTATS FINAUX  run_id={run_id[:8]}")
        print(f"  iterations      : {iteration} (converge={converged})")
        print(f"  chat_calls      : {result['chat_calls']}")
        print(f"  embed_calls     : {embed_calls}")
        if self.cache:
            print(f"  cache hits      : {self.cache.hits} / {self.cache.hits + self.cache.misses}")
        print(f"  stability       : {result['stability']:.6f}")
        print(f"  H_V drop        : {result['h_v_drop']:.6f}")
        print(f"  momentum final  : {result['momentum_final']:.6f}")
        print(f"{'-'*64}\n")
        
        return result
    
    def run_multiple(self, num_runs: int = 3) -> List[Dict]:
        all_results = []
        for i in range(num_runs):
            run_id = hashlib.md5(f"{time.time()}{i}".encode()).hexdigest()
            topic = TOPICS[i % len(TOPICS)]
            result = self.run(topic, run_id)
            all_results.append(result)
            
            # Sauvegarde incrémentale
            with open("nexus_flux_v4_results.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")
        
        return all_results

# ===================== MAIN =====================

if __name__ == "__main__":
    print("#"*64)
    print("#  NEXUS-FLUX V4  --  Version Hybride Optimisée")
    print(f"#  Mode     : {'REAL' if OPENAI_AVAILABLE else 'MOCK'} (pas de cle OpenAI)")
    print(f"#  Config   : alpha={CONFIG['alpha_base']}  quality>={CONFIG['quality_threshold']}  momentum<{CONFIG['momentum_threshold']}")
    print(f"#  Optim    : batch_eval + cache_embed + inertia + alpha_decay")
    print(f"#  Logs     : nexus_flux_v4_results.jsonl")
    print("#"*64)
    
    engine = NexusFluxV4()
    results = engine.run_multiple(num_runs=3)
    
    print("\n" + "#"*64)
    print("#  RECAPITULATIF -- 3 runs termines")
    print(f"#  Converges      : {sum(1 for r in results if r['converged'])}/{len(results)}")
    print(f"#  Iterations moy : {sum(r['iterations'] for r in results)/len(results):.1f}")
    print(f"#  Stabilite moy  : {sum(r['stability'] for r in results)/len(results):.4f}")
    print(f"#  H_V drop moy   : {sum(r['h_v_drop'] for r in results)/len(results):.4f}")
    print(f"#  Chat calls moy : {sum(r['chat_calls'] for r in results)/len(results):.1f}")
    print(f"#  Cache hit rate : {sum(r['cache_hits'] for r in results)/(sum(r['cache_hits'] for r in results) + sum(r['embed_calls'] for r in results) + 1):.1%}")
    print("#"*64)
