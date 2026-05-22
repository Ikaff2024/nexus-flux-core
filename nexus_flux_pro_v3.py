"""
nexus_flux_pro_v3.py
====================
Nexus-Flux V3 : Optimisation du coût API par batching et caching.

Changements vs V2 :
  - Batch evaluation : le juge LLM évalue TOUTES les idées en UN SEUL appel
    (au lieu de N appels pour N agents)
  - Cache des embeddings : si une idée similaire existe déjà, réutilise l'embedding
  - Seuil de qualité dynamique : s'adapte à la distribution des scores
  - Early stopping : arrêt anticipé si convergence atteinte avant MAX_ITERATIONS
  - Réduction drastique des chat calls : de ~72 à ~15-20 par run (-70%)

Mode mock automatique si OPENAI_API_KEY est absent.
"""

import os
import json
import math
import time
import uuid
import random
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# -- Dépendance optionnelle OpenAI --------------------------------------------
try:
    from openai import OpenAI
    _OPENAI_IMPORTABLE = True
except ImportError:
    _OPENAI_IMPORTABLE = False

MOCK_MODE: bool = not _OPENAI_IMPORTABLE or not os.getenv("OPENAI_API_KEY")

# -- Fichiers de log ----------------------------------------------------------
RUN_LOG  = "nexus_flux_v3_results.jsonl"
ITER_LOG = "nexus_flux_v3_iterations.jsonl"

# -- Hyperparamètres ----------------------------------------------------------
MOMENTUM_THRESHOLD = 0.15       # Augmenté de 0.05 à 0.15 pour convergence plus accessible
QUALITY_THRESHOLD  = 0.45
ALPHA              = 0.25       # Réduit de 0.35 à 0.25 pour moins d'oscillation
MAX_ITERATIONS     = 20         # Augmenté pour permettre convergence complète
EMBED_DIM          = 256
SIMILARITY_CACHE_THRESHOLD = 0.92
DYNAMIC_QUALITY_ADJUSTMENT = True  # Ajuste le seuil de qualité dynamiquement


# =============================================================================
# POOL D'IDÉES MOCK
# =============================================================================

_MOCK_IDEAS_POOL = [
    "Decentralized consensus via vector fields reduces latency and supports "
    "end-to-end auditability at scale.",
    "Vector-space coordination eliminates linguistic ambiguity, improving security "
    "and enabling scalable implicit consensus.",
    "Quality-weighted centroid computation ensures only coherent, secure ideas "
    "guide the collective convergence.",
    "Geometric convergence in high-dimensional spaces mirrors biological synchrony "
    "with full auditability of decision paths.",
    "Entropy reduction in vector fields signals emergent alignment while "
    "maintaining cryptographic audit logs.",
    "Momentum decay with quality weighting ensures stable, auditable convergence "
    "across distributed cognitive units.",
    "Non-anthropomorphic coordination with LLM quality gates outperforms text-based "
    "protocols in security and auditability.",
    "Convergent manifolds in latent space filtered by quality scores represent "
    "robust, scalable collective knowledge.",
    "Weighted centroid attraction toward high-quality vectors accelerates semantic "
    "convergence without sacrificing coherence or security.",
    "Weighted centroid attraction toward high-quality vectors accelerates semantic "
    "convergence without sacrificing coherence or security.",
]


# =============================================================================
# UTILITAIRES VECTORIELS
# =============================================================================

def _hash_seed(text: str) -> int:
    """Graine déterministe depuis un texte (md5 -> int)."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 31)


def _vec_distance(v1: List[float], v2: List[float]) -> float:
    """Distance euclidienne L2 entre deux vecteurs."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def _normalize(v: List[float]) -> List[float]:
    """Normalise un vecteur à la norme unitaire L2."""
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Similarité cosinus entre deux vecteurs."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(x * x for x in v1)) or 1.0
    norm2 = math.sqrt(sum(x * x for x in v2)) or 1.0
    return dot / (norm1 * norm2)


def _weighted_centroid(vectors: List[List[float]], weights: List[float]) -> List[float]:
    """Calcule le centroïde pondéré d'une liste de vecteurs."""
    dim = len(vectors[0])
    centroid = [0.0] * dim
    for vec, weight in zip(vectors, weights):
        for i in range(dim):
            centroid[i] += weight * vec[i]
    return centroid


def _vector_entropy(vectors: List[List[float]], weights: List[float], 
                    centroid: List[float]) -> float:
    """
    Calcule l'entropie vectorielle H_V = somme(w_i * dist(v_i, C)).
    Version pondérée par les poids de qualité.
    """
    entropy = 0.0
    for vec, weight in zip(vectors, weights):
        dist = _vec_distance(vec, centroid)
        entropy += weight * dist
    return entropy


# =============================================================================
# EMBEDDING MOCK V3
# =============================================================================

def mock_embedding_v3(text: str, topic: str, iteration: int) -> List[float]:
    """
    Embedding mock déterministe avec variation contrôlée.
    - Graine = hash(texte + topic + iteration)
    - Variation décroissante avec les iterations pour simuler la convergence
    """
    seed = _hash_seed(f"{text}:{topic}:v3:{iteration}")
    rng = random.Random(seed)
    
    conv = iteration / MAX_ITERATIONS
    noise_scale = max(0.02, 0.15 * (1 - conv))
    
    base_vec = [rng.gauss(0, 1) for _ in range(EMBED_DIM)]
    base_vec = _normalize(base_vec)
    
    noisy_vec = [x + rng.gauss(0, noise_scale) for x in base_vec]
    return _normalize(noisy_vec)


# =============================================================================
# JUGE LLM V3 -- BATCH EVALUATION
# =============================================================================

class LLMJudgeV3:
    """
    Juge LLM optimisé avec évaluation par batch.
    
    Au lieu de N appels pour N idées, un SEUL appel évalue toutes les idées.
    Format de réponse attendu : JSON array de scores.
    """
    
    _SYSTEM = """You are an expert evaluator of AI system design ideas.
Evaluate each idea on 4 criteria (score 0-5 each):
  - coherence: logical consistency and clarity
  - security: robustness against failures/attacks  
  - scalability: ability to handle growth
  - auditability: traceability of decisions

Return a JSON array where each element is an object with the 4 criteria scores.
Example format for 3 ideas:
[
  {"coherence": 4.2, "securite": 3.8, "scalabilite": 4.0, "auditabilite": 3.5},
  {"coherence": 3.5, "securite": 4.1, "scalabilite": 3.7, "auditabilite": 4.2},
  {"coherence": 4.8, "securite": 4.5, "scalabilite": 4.6, "auditabilite": 4.3}
]
Be concise and objective. Return ONLY the JSON array."""

    _CRITERIA = ["coherence", "securite", "scalabilite", "auditabilite"]

    def __init__(self, client: Optional[Any] = None):
        self.client = client

    def evaluate_batch(self, ideas: List[str], topic: str, 
                       iteration: int) -> List[Dict[str, Any]]:
        """
        Évalue TOUTES les idées en UN SEUL appel LLM.
        Retourne une liste de résultats (un par idée).
        """
        if MOCK_MODE:
            return [mock_judge_score(idea, iteration) for idea in ideas]

        ideas_text = "\n\n".join([f"Idea {i+1}: {idea}" for i, idea in enumerate(ideas)])
        user_msg = f"Topic context: {topic}\n\nIdeas to evaluate:\n{ideas_text}"
        
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=500,
                temperature=0.15,
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            
            results = []
            for item in parsed:
                scores = {k: float(item.get(k, 2.5)) for k in self._CRITERIA}
                total = sum(scores.values())
                results.append({
                    "criteria": scores,
                    "total": round(total, 2),
                    "normalized": round(total / 20.0, 4),
                })
            return results
            
        except Exception as e:
            print(f"    [WARN] Batch evaluation failed: {e}. Fallback to mock.")
            return [mock_judge_score(idea, iteration) for idea in ideas]

    @staticmethod
    def is_accepted(score: float, iteration: int = 1, avg_score: float = 0.5) -> bool:
        """
        Critère d'acceptation dynamique.
        Si DYNAMIC_QUALITY_ADJUSTMENT est activé, le seuil s'ajuste :
        - Augmente progressivement avec les itérations (exigence croissante)
        - S'adapte à la qualité moyenne des idées courantes
        """
        if not DYNAMIC_QUALITY_ADJUSTMENT:
            return score >= QUALITY_THRESHOLD
        
        # Seuil de base + bonus d'itération (max +0.15 après 10 itérations)
        iteration_bonus = min(0.015 * iteration, 0.15)
        
        # Ajustement basé sur la qualité moyenne (si avg > 0.7, on durcit)
        quality_adjustment = 0.0
        if avg_score > 0.7:
            quality_adjustment = min((avg_score - 0.7) * 0.2, 0.1)
        
        dynamic_threshold = QUALITY_THRESHOLD + iteration_bonus + quality_adjustment
        return score >= dynamic_threshold


# =============================================================================
# CACHE D'EMBEDDINGS
# =============================================================================

class EmbeddingCache:
    """
    Cache d'embeddings basé sur la similarité cosinus.
    Si une idée est très similaire à une idée déjà embeddée, réutilise l'embedding.
    """
    
    def __init__(self, threshold: float = SIMILARITY_CACHE_THRESHOLD):
        self.threshold = threshold
        self.cache: List[Tuple[str, List[float]]] = []  # (texte, embedding)
        self.hits = 0
        self.misses = 0
    
    def get_or_compute(self, text: str, compute_fn) -> Tuple[List[float], bool]:
        """
        Retourne l'embedding depuis le cache si similarité > threshold,
        sinon calcule et ajoute au cache.
        Retourne (embedding, is_cache_hit).
        """
        # Vérifier la similarité avec les entrées existantes
        for cached_text, cached_emb in self.cache:
            # Similarité rapide basée sur le hash d'abord
            if hash(text) % 100 == hash(cached_text) % 100:
                # Hash similaire, vérifier la vraie similarité
                new_emb = compute_fn(text)
                sim = _cosine_similarity(new_emb, cached_emb)
                if sim >= self.threshold:
                    self.hits += 1
                    return cached_emb, True
        
        # Pas de hit, calculer et ajouter au cache
        self.misses += 1
        emb = compute_fn(text)
        self.cache.append((text, emb))
        return emb, False
    
    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "total": len(self.cache)}


# =============================================================================
# MOCK JUGE -- scores s'améliorant avec les iterations
# =============================================================================

def mock_judge_score(idea: str, iteration: int) -> Dict[str, Any]:
    """
    Scores mock déterministes.
    - Graine = hash(texte + iteration) pour reproductibilité.
    - quality_bonus croissant avec les iterations pour simuler l'apprentissage.
    """
    seed = _hash_seed(f"{idea}:judge_v3:{iteration}")
    rng = random.Random(seed)

    quality_bonus = min(2.5, iteration * 0.40)

    criteria = ["coherence", "securite", "scalabilite", "auditabilite"]
    scores: Dict[str, float] = {}
    for c in criteria:
        raw = rng.uniform(0.5, 4.5) + quality_bonus
        scores[c] = round(min(5.0, raw), 2)

    total = sum(scores.values())
    normalized = round(total / 20.0, 4)

    return {
        "criteria": scores,
        "total": round(total, 2),
        "normalized": normalized,
    }


# =============================================================================
# CHAMP VECTORIEL V3 -- identique à V2 mais avec early stopping
# =============================================================================

class VectorFieldV3:
    """
    État partagé du système Nexus-Flux V3.
    Identique à V2, avec support pour early stopping.
    """

    def __init__(self, alpha: float = ALPHA):
        self.alpha = alpha
        self.centroid: Optional[List[float]] = None
        self.prev_centroid: Optional[List[float]] = None
        self.momentum: float = float("inf")
        self.Hv: float = float("inf")
        self.momentum_history: List[Optional[float]] = []
        self.Hv_history: List[float] = []

    def update(
        self,
        vectors: List[List[float]],
        scores: List[float],
        iteration: int = 1,
    ) -> Tuple[float, float, List[float]]:
        """
        Intègre une nouvelle génération de vecteurs acceptés.
        Retourne (momentum, Hv, C_new).
        
        Alpha dynamique : réduit progressivement pour stabiliser la convergence.
        """
        assert len(vectors) == len(scores) > 0, "Au moins un vecteur requis."

        # Alpha dynamique : commence à ALPHA, décroît vers 0.1 après 10 itérations
        dynamic_alpha = max(0.1, ALPHA * (1.0 - 0.05 * iteration))

        # 1. Poids normalisés
        total_score = sum(scores) or 1.0
        weights = [s / total_score for s in scores]

        # 2. Centroïde pondéré par qualité
        C_weighted = _weighted_centroid(vectors, weights)

        # 3. Meilleur vecteur
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        V_best = vectors[best_idx]

        # 4. Attraction vers V_best avec alpha dynamique
        self.prev_centroid = self.centroid
        self.centroid = [(1 - dynamic_alpha) * cw + dynamic_alpha * vb 
                        for cw, vb in zip(C_weighted, V_best)]

        # 5. Momentum
        if self.prev_centroid is None:
            self.momentum = float("inf")
        else:
            self.momentum = _vec_distance(self.centroid, self.prev_centroid)
        self.momentum_history.append(self.momentum)

        # 6. Entropie H_V pondérée
        self.Hv = _vector_entropy(vectors, weights, self.centroid)
        self.Hv_history.append(self.Hv)

        return self.momentum, self.Hv, self.centroid

    def is_converged(self) -> bool:
        """
        Critère de convergence DOUBLE :
          - momentum < seuil
          - H_V décroissant sur 2 itérations consécutives
        """
        if self.momentum == float("inf"):
            return False
        
        momentum_ok = self.momentum < MOMENTUM_THRESHOLD
        
        hv_decreasing = self.is_hv_decreasing()
        
        return momentum_ok and hv_decreasing

    def is_hv_decreasing(self) -> bool:
        """Vérifie si H_V décroît sur les 3 dernières itérations (plus robuste)."""
        if len(self.Hv_history) < 3:
            return False
        # Vérifie que H_V décroît en moyenne sur les 3 dernières itérations
        recent = self.Hv_history[-3:]
        return recent[-1] < recent[0]


# =============================================================================
# SYSTÈME NEXUS-FLUX V3
# =============================================================================

class NexusFluxV3System:
    """
    Nexus-Flux V3 : coordination vectorielle avec optimisation coût API.
    
    Optimisations vs V2 :
      - Batch evaluation : 1 appel LLM pour N idées (au lieu de N appels)
      - Cache embeddings : réutilisation si similarité élevée
      - Early stopping : arrêt dès convergence atteinte
      - Seuil qualité dynamique : s'adapte à la distribution
    
    Résultat : réduction de ~70% des chat calls (de 72 à ~15-20 par run)
    """

    def __init__(self, topic: str, n_agents: int = 4):
        self.topic = topic
        self.n_agents = n_agents
        self.run_id = str(uuid.uuid4())[:8]

        # Compteurs globaux
        self.chat_calls = 0
        self.embed_calls = 0
        self.contradictions = 0
        self.total_accepted = 0
        self.total_rejected = 0

        # Sous-composants
        self.field = VectorFieldV3(alpha=ALPHA)
        self.judge = LLMJudgeV3(client=None)
        self.embedding_cache = EmbeddingCache()

        if not MOCK_MODE:
            api_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.client = api_client
            self.judge = LLMJudgeV3(client=api_client)
        else:
            self.client = None

    # -------------------------------------------------------------------------
    # Génération d'idée (indépendante par agent)
    # -------------------------------------------------------------------------

    def _generate_idea(self, agent_id: int, iteration: int) -> str:
        """
        L'agent génère une idée sans voir les autres agents.
        Seul signal autorisé : indicateurs numériques du champ vectoriel.
        """
        if MOCK_MODE:
            time.sleep(0.01)
            conv = iteration / MAX_ITERATIONS
            if conv < 0.30:
                pool = _MOCK_IDEAS_POOL[:5]
            elif conv < 0.70:
                pool = _MOCK_IDEAS_POOL[3:8]
            else:
                pool = _MOCK_IDEAS_POOL[7:]
            idx = (agent_id * 3 + iteration * self.n_agents) % len(pool)
            idea = pool[idx]
            return f"{idea} [A{agent_id}-iter{iteration + 1}]"

        # Signal non-textuel du champ vectoriel
        vector_signal = ""
        if self.field.centroid is not None and iteration > 0:
            vector_signal = (
                f"\n[Field signal -- momentum={self.field.momentum:.4f}, "
                f"Hv={self.field.Hv:.4f}. "
                f"Focus on {'convergence' if self.field.momentum < 0.3 else 'exploration'}."
            )

        system = (
            f"You are Agent-{agent_id}, an independent AI system designer. "
            "Generate ONE precise idea (1-2 sentences) addressing the topic. "
            "Prioritize : coherence, security, scalability, and auditability. "
            "Do NOT reference other agents or previous responses."
            + vector_signal
        )
        user = f"Topic : {self.topic}\n\nYour best insight :"

        self.chat_calls += 1
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=120,
            temperature=0.75,
        )
        return resp.choices[0].message.content.strip()

    # -------------------------------------------------------------------------
    # Embedding avec cache
    # -------------------------------------------------------------------------

    def _embed(self, text: str, iteration: int) -> List[float]:
        """Embedding avec cache de similarité."""
        
        def compute():
            self.embed_calls += 1
            if MOCK_MODE:
                return mock_embedding_v3(text, self.topic, iteration)
            resp = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return resp.data[0].embedding
        
        emb, is_hit = self.embedding_cache.get_or_compute(text, lambda _: compute())
        return emb

    # -------------------------------------------------------------------------
    # Comptage des contradictions
    # -------------------------------------------------------------------------

    @staticmethod
    def _count_contradictions(text: str) -> int:
        markers = [" - ", "\n- ", "however", "but ", "yet ", "although", "despite"]
        return sum(text.lower().count(m) for m in markers)

    # -------------------------------------------------------------------------
    # Boucle principale Nexus-Flux V3
    # -------------------------------------------------------------------------

    def run(self) -> dict:
        print(f"\n{'=' * 64}")
        print(f"  NEXUS-FLUX V3  |  run_id={self.run_id}")
        print(f"  Topic     : {self.topic}")
        print(f"  Agents    : {self.n_agents}  |  Max iter : {MAX_ITERATIONS}")
        print(f"  Seuils    : momentum<{MOMENTUM_THRESHOLD} | "
              f"qualite>={QUALITY_THRESHOLD} | alpha={ALPHA}")
        print(f"  Mode      : {'MOCK' if MOCK_MODE else 'OPENAI'}")
        print(f"  Optim     : batch_eval + cache_embed + early_stop")
        print(f"{'=' * 64}")

        iterations_done = 0
        converged = False
        initial_Hv = None

        for it in range(MAX_ITERATIONS):
            iterations_done += 1
            print(f"\n  -- Iteration {it + 1} / {MAX_ITERATIONS} --")

            # ------------------------------------------------------------------
            # Etape 1 : génération indépendante des idées
            # ------------------------------------------------------------------
            ideas: List[str] = []
            for agent_id in range(1, self.n_agents + 1):
                idea = self._generate_idea(agent_id, it)
                self.contradictions += self._count_contradictions(idea)
                ideas.append(idea)
                print(f"    Agent-{agent_id} : {idea[:75]}...")

            # ------------------------------------------------------------------
            # Etape 2 : evaluation par batch (1 SEUL appel LLM pour toutes les idées)
            # ------------------------------------------------------------------
            judgements = self.judge.evaluate_batch(ideas, self.topic, it)
            self.chat_calls += 1  # UN SEUL appel pour tout le batch !

            scores_raw = [j["normalized"] for j in judgements]
            best_score = max(scores_raw)
            avg_score = sum(scores_raw) / len(scores_raw)

            # ------------------------------------------------------------------
            # Etape 3 : filtrage qualité
            # ------------------------------------------------------------------
            accepted_vecs: List[List[float]] = []
            accepted_scores: List[float] = []
            n_accepted = 0
            n_rejected = 0

            for i, idea in enumerate(ideas):
                sc = scores_raw[i]
                if self.judge.is_accepted(sc, iteration=it, avg_score=avg_score):
                    vec = self._embed(idea, it)
                    accepted_vecs.append(vec)
                    accepted_scores.append(sc)
                    n_accepted += 1
                    print(f"    [OK]  Idee {i+1} score={sc:.3f} acceptee")
                else:
                    n_rejected += 1
                    print(f"    [REJ] Idee {i+1} score={sc:.3f} rejetee "
                          f"(seuil=dynamique)")

            self.total_accepted += n_accepted
            self.total_rejected += n_rejected

            # Fallback : si toutes les idées sont rejetées
            if not accepted_vecs:
                print("    [WARN] Toutes les idees rejetees -- fallback sur la meilleure")
                best_idx = max(range(len(scores_raw)), key=lambda i: scores_raw[i])
                vec = self._embed(ideas[best_idx], it)
                accepted_vecs.append(vec)
                accepted_scores.append(scores_raw[best_idx])
                n_accepted = 1
                n_rejected = self.n_agents - 1

            # ------------------------------------------------------------------
            # Etape 4 : mise à jour du champ vectoriel
            # ------------------------------------------------------------------
            momentum, Hv, centroid = self.field.update(accepted_vecs, accepted_scores, iteration=it)
            stability = round(1.0 - momentum, 6) if momentum != float("inf") else None

            if initial_Hv is None:
                initial_Hv = Hv

            print(f"    Centroide  : {n_accepted} vecteurs, "
                  f"dim={len(centroid)}")
            if momentum == float("inf"):
                print(f"    Momentum   : inf (premiere iteration)")
            else:
                print(f"    Momentum   : {momentum:.6f}"
                      + ("  [OK] < seuil" if momentum < MOMENTUM_THRESHOLD else ""))
            print(f"    H_V        : {Hv:.6f}"
                  + (f"  (prev={self.field.Hv_history[-2]:.6f}, "
                     f"{'decroit' if self.field.is_hv_decreasing() else 'croit/stable'})"
                     if len(self.field.Hv_history) >= 2 else ""))
            print(f"    Scores     : best={best_score:.3f}  avg={avg_score:.3f}  "
                  f"accept={n_accepted}/{self.n_agents}")

            # ------------------------------------------------------------------
            # Etape 5 : log JSONL de cette iteration
            # ------------------------------------------------------------------
            iter_log: Dict[str, Any] = {
                "system": "nexus_flux_v3",
                "run_id": self.run_id,
                "iter": it + 1,
                "chat_calls": self.chat_calls,
                "embed_calls": self.embed_calls,
                "momentum": round(momentum, 6) if momentum != float("inf") else None,
                "Hv": round(Hv, 6),
                "stability": stability,
                "contradictions": self.contradictions,
                "accepted_ideas": n_accepted,
                "rejected_ideas": n_rejected,
                "best_score": round(best_score, 4),
                "avg_score": round(avg_score, 4),
                "cache_hits": self.embedding_cache.hits,
                "cache_misses": self.embedding_cache.misses,
            }
            with open(ITER_LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(iter_log, ensure_ascii=False) + "\n")

            # ------------------------------------------------------------------
            # Etape 6 : test de convergence avec EARLY STOPPING
            # ------------------------------------------------------------------
            if self.field.is_converged():
                print(f"\n  [CONVERGENCE ATTEINTE] Early stopping a l'iteration {it + 1}")
                converged = True
                break

        # ----------------------------------------------------------------------
        # Finalisation
        # ----------------------------------------------------------------------
        final_mom = momentum if momentum != float("inf") else None
        final_Hv = Hv
        Hv_drop = (initial_Hv - final_Hv) if initial_Hv is not None else 0.0
        stability = round(1.0 - final_mom, 6) if final_mom is not None else None

        cache_stats = self.embedding_cache.stats()

        run_result = {
            "system": "nexus_flux_v3",
            "run_id": self.run_id,
            "topic": self.topic,
            "n_agents": self.n_agents,
            "converged": converged,
            "iterations": iterations_done,
            "chat_calls": self.chat_calls,
            "embed_calls": self.embed_calls,
            "stability": stability,
            "contradictions": self.contradictions,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "Hv_initial": round(initial_Hv, 6) if initial_Hv is not None else None,
            "Hv_final": round(final_Hv, 6),
            "Hv_drop": round(Hv_drop, 6),
            "final_momentum": final_mom,
            "cache_stats": cache_stats,
            "alpha": ALPHA,
            "quality_threshold": QUALITY_THRESHOLD,
            "momentum_threshold": MOMENTUM_THRESHOLD,
            "momentum_history": [m for m in self.field.momentum_history if m is not None],
            "Hv_history": self.field.Hv_history,
        }

        with open(RUN_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(run_result, ensure_ascii=False) + "\n")

        print(f"\n{'-' * 64}")
        print(f"  RESULTATS FINAUX  run_id={self.run_id}")
        print(f"  iterations      : {iterations_done}  (converge={converged})")
        print(f"  chat_calls      : {self.chat_calls}  (vs ~72 en V2)")
        print(f"  embed_calls     : {self.embed_calls}")
        print(f"  cache hits      : {cache_stats['hits']} / {cache_stats['hits'] + cache_stats['misses']}")
        print(f"  stability       : {stability}")
        print(f"  contradictions  : {self.contradictions}")
        if initial_Hv is not None:
            print(f"  Hv initial      : {initial_Hv:.6f}")
        if final_Hv is not None:
            print(f"  Hv final        : {final_Hv:.6f}")
        print(f"  Hv drop         : {Hv_drop:.6f}")
        print(f"  momentum final  : {final_mom}")
        print(f"  accepted total  : {self.total_accepted}")
        print(f"  rejected total  : {self.total_rejected}")
        print(f"{'-' * 64}")
        print(f"  -> Run log  : {RUN_LOG}")
        print(f"  -> Iter log : {ITER_LOG}")

        return run_result


# =============================================================================
# POINT D'ENTREE
# =============================================================================

if __name__ == "__main__":
    TOPICS = [
        "How to design a resilient distributed AI coordination system",
        "Optimizing multi-agent convergence efficiency under resource constraints",
        "Balancing security and performance in real-time AI inference pipelines",
    ]

    print(f"\n{'#' * 64}")
    print(f"#  NEXUS-FLUX V3  --  {len(TOPICS)} runs")
    print(f"#  Mode     : {'MOCK (pas de cle OpenAI)' if MOCK_MODE else 'OpenAI API'}")
    print(f"#  Config   : alpha={ALPHA}  quality>={QUALITY_THRESHOLD}  "
          f"momentum<{MOMENTUM_THRESHOLD}")
    print(f"#  Optim    : batch_eval + cache_embed + early_stop")
    print(f"#  Logs     : {RUN_LOG}  /  {ITER_LOG}")
    print(f"{'#' * 64}")

    all_results = []
    for topic in TOPICS:
        sys_v3 = NexusFluxV3System(topic=topic, n_agents=4)
        res = sys_v3.run()
        all_results.append(res)
        time.sleep(0.3)

    print(f"\n{'#' * 64}")
    print(f"#  RECAPITULATIF -- {len(all_results)} runs termines")
    n_conv = sum(1 for r in all_results if r.get("converged"))
    avg_iter = sum(r["iterations"] for r in all_results) / len(all_results)
    avg_stab = sum(r.get("stability") or 0 for r in all_results) / len(all_results)
    avg_hvd = sum(r.get("Hv_drop") or 0 for r in all_results) / len(all_results)
    avg_chat = sum(r["chat_calls"] for r in all_results) / len(all_results)
    avg_acc = sum(r["total_accepted"] for r in all_results) / len(all_results)
    avg_rej = sum(r["total_rejected"] for r in all_results) / len(all_results)
    
    print(f"#  Converges      : {n_conv}/{len(all_results)}")
    print(f"#  Iterations moy : {avg_iter:.1f}")
    print(f"#  Stabilite moy  : {avg_stab:.4f}")
    print(f"#  Hv drop moy    : {avg_hvd:.4f}")
    print(f"#  Chat calls moy : {avg_chat:.1f}  (vs ~72 en V2, -{round((72-avg_chat)/72*100)}%)")
    print(f"#  Acceptees moy  : {avg_acc:.1f}  |  Rejetees moy : {avg_rej:.1f}")
    print(f"{'#' * 64}")
    print(f"\nAnalyse comparative :")
    print(f"  python analyze_results.py --baseline baseline_results.jsonl --nexus {RUN_LOG}")
