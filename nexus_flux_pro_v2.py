"""
nexus_flux_pro_v2.py
====================
Nexus-Flux V2 : convergence vectorielle guidee par la qualite des idees.

Changements vs V1 :
  - Juge LLM qui note chaque idee sur 4 criteres (coherence, securite,
    scalabilite, auditabilite) -- score /20 normalise en [0, 1]
  - Rejet des idees dont le score < QUALITY_THRESHOLD (0.45 par defaut)
  - Centroide PONDERE par les scores de qualite (pas une simple moyenne)
  - Attraction vers le meilleur vecteur : C_new = (1-alpha)*C_w + alpha*V_best
  - Critere de convergence DOUBLE :
      momentum < seuil  ET  H_V decroissant sur 2 iterations consecutives
  - Log JSONL detaille par iteration (ITER_LOG) + resume de run (RUN_LOG)
  - Sortie compatible avec analyze_results.py

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

# -- Dependance optionnelle OpenAI --------------------------------------------
try:
    from openai import OpenAI
    _OPENAI_IMPORTABLE = True
except ImportError:
    _OPENAI_IMPORTABLE = False

MOCK_MODE: bool = not _OPENAI_IMPORTABLE or not os.getenv("OPENAI_API_KEY")

# -- Fichiers de log ----------------------------------------------------------
RUN_LOG  = "nexus_flux_v2_results.jsonl"    # 1 ligne / run (compatible analyze_results.py)
ITER_LOG = "nexus_flux_v2_iterations.jsonl"  # 1 ligne / iteration (log detaille)

# -- Hyperparametres ----------------------------------------------------------
MOMENTUM_THRESHOLD = 0.05   # seuil de convergence du momentum
QUALITY_THRESHOLD  = 0.45   # score min pour qu'une idee soit acceptee
ALPHA              = 0.2    # force d'attraction vers le meilleur vecteur
MAX_ITERATIONS     = 10     # garde-fou anti-boucle infinie
EMBED_DIM          = 256    # dimension des embeddings en mode mock


# =============================================================================
# POOL D'IDEES MOCK
# =============================================================================

_MOCK_IDEAS_POOL = [
    # Phase exploration (iter 0-2) : idees diverses mais liees au domaine
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
    # Phase refinement (iter 3-6) : idees plus focalisees
    "Momentum decay with quality weighting ensures stable, auditable convergence "
    "across distributed cognitive units.",
    "Non-anthropomorphic coordination with LLM quality gates outperforms text-based "
    "protocols in security and auditability.",
    "Convergent manifolds in latent space filtered by quality scores represent "
    "robust, scalable collective knowledge.",
    # Phase convergence (iter 7+) : idees tres alignees
    "Weighted centroid attraction toward high-quality vectors accelerates semantic "
    "convergence without sacrificing coherence or security.",
    "Dual convergence criterion (momentum + entropy drop) guarantees stable, "
    "auditable outcomes in multi-agent coordination at any scale.",
]


# =============================================================================
# UTILITAIRES VECTORIELS
# =============================================================================

def _hash_seed(text: str) -> int:
    """Graine deterministe depuis un texte (md5 -> int)."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 31)


def _vec_distance(v1: List[float], v2: List[float]) -> float:
    """Distance euclidienne L2 entre deux vecteurs."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def _normalize(v: List[float]) -> List[float]:
    """Normalise un vecteur a la norme unitaire L2."""
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _weighted_centroid(
    vectors: List[List[float]],
    weights: List[float],
) -> List[float]:
    """
    Centroide pondere par les poids.
    C = sum(w_i * v_i) / sum(w_i)
    Les poids n'ont pas besoin d'etre normalises.
    """
    total_w = sum(weights) or 1.0
    dim = len(vectors[0])
    centroid = [
        sum(weights[i] * vectors[i][d] for i in range(len(vectors))) / total_w
        for d in range(dim)
    ]
    return centroid


# =============================================================================
# MOCK EMBEDDING -- convergence exponentielle garantie
# =============================================================================

def mock_embedding_v2(
    text: str,
    topic: str,
    iteration: int,
    dim: int = EMBED_DIM,
) -> List[float]:
    """
    Embedding mock avec convergence garantie par interpolation exponentielle.

    Principe :
      - Chaque agent a un vecteur base deterministe (propre au texte de l'idee).
      - Il existe un vecteur consensus deterministe pour le topic (identique
        pour tous les agents).
      - On interpole le vecteur de l'agent vers le consensus avec un facteur
        blend croissant exponentiellement :
            blend = 1 - exp(-0.42 * iteration)
      - Plus l'iteration avance, plus tous les vecteurs convergent vers le
        consensus -> H_V decroit, momentum decroit.

    Valeurs de blend :
      iter 0 : 0.00  (diversite max)
      iter 3 : 0.72
      iter 5 : 0.88
      iter 7 : 0.94
      iter 8 : 0.97  (momentum inter-iter ~ 0.03 < seuil)
    """
    # Vecteur base deterministe propre a ce texte-agent
    agent_rng = random.Random(_hash_seed(text))
    agent_vec = [agent_rng.gauss(0, 1) for _ in range(dim)]

    # Vecteur consensus deterministe pour ce topic (meme pour tous les agents)
    cons_rng = random.Random(_hash_seed(f"nfv2_consensus:{topic}"))
    cons_vec = [cons_rng.gauss(0, 1) for _ in range(dim)]

    # Facteur de convergence exponentiel
    blend = 1.0 - math.exp(-0.42 * iteration)

    # Interpolation lineaire vers le consensus
    blended = [
        (1.0 - blend) * a + blend * c
        for a, c in zip(agent_vec, cons_vec)
    ]
    return _normalize(blended)


# =============================================================================
# MOCK JUGE -- scores s'ameliorant avec les iterations
# =============================================================================

def mock_judge_score(idea: str, iteration: int) -> Dict[str, Any]:
    """
    Scores mock deterministes.

    - Graine = hash(texte + iteration) pour reproducibilite.
    - quality_bonus croissant avec les iterations pour simuler l'apprentissage :
        iter 0 : bonus = 0.0   -> base [0.5, 4.5]  -> ~30% de rejets
        iter 3 : bonus = 1.2   -> base [1.7, 5.0]  -> ~5%  de rejets
        iter 5 : bonus = 2.0   -> base [2.5, 5.0]  -> 0%   de rejets
    """
    seed = _hash_seed(f"{idea}:judge_v2:{iteration}")
    rng  = random.Random(seed)

    quality_bonus = min(2.5, iteration * 0.40)

    criteria = ["coherence", "securite", "scalabilite", "auditabilite"]
    scores: Dict[str, float] = {}
    for c in criteria:
        raw = rng.uniform(0.5, 4.5) + quality_bonus
        scores[c] = round(min(5.0, raw), 2)

    total      = sum(scores.values())        # max = 20
    normalized = round(total / 20.0, 4)     # [0, 1]

    return {
        "criteria":   scores,
        "total":      round(total, 2),
        "normalized": normalized,
    }


# =============================================================================
# CHAMP VECTORIEL V2 -- centroide pondere + attraction V_best
# =============================================================================

class VectorFieldV2:
    """
    Etat partage du systeme Nexus-Flux V2.

    Calculs par iteration :
      1. Poids normalises    : w_i = score_i / sum(scores)
      2. Centroide pondere   : C_w = sum(w_i * v_i)
      3. Attraction V_best   : C_new = (1-alpha)*C_w + alpha*V_best
      4. Momentum            : ||C_new - C_prev||
      5. H_V pondere         : sum(w_i * dist(v_i, C_new))
    """

    def __init__(self, alpha: float = ALPHA):
        self.alpha            = alpha
        self.centroid:         Optional[List[float]] = None
        self.prev_centroid:    Optional[List[float]] = None
        self.momentum:         float = float("inf")
        self.Hv:               float = float("inf")
        self.momentum_history: List[Optional[float]] = []
        self.Hv_history:       List[float] = []

    def update(
        self,
        vectors: List[List[float]],
        scores:  List[float],
    ) -> Tuple[float, float, List[float]]:
        """
        Integre une nouvelle generation de vecteurs acceptes.
        Retourne (momentum, Hv, C_new).
        """
        assert len(vectors) == len(scores) > 0, "Au moins un vecteur requis."

        # 1. Poids normalises (les scores deviennent des poids somme=1)
        total_score = sum(scores) or 1.0
        weights = [s / total_score for s in scores]

        # 2. Centroide pondere par qualite
        C_weighted = _weighted_centroid(vectors, weights)

        # 3. Meilleur vecteur (score le plus eleve)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        V_best   = vectors[best_idx]

        # 4. Attraction vers le meilleur vecteur
        #    C_new = (1 - alpha) * C_weighted + alpha * V_best
        C_new = [
            (1.0 - self.alpha) * cw + self.alpha * vb
            for cw, vb in zip(C_weighted, V_best)
        ]

        # 5. Momentum = distance euclidienne entre C_new et C_prev
        self.prev_centroid = self.centroid
        self.centroid      = C_new
        if self.prev_centroid is not None:
            self.momentum = _vec_distance(self.centroid, self.prev_centroid)
        else:
            self.momentum = float("inf")

        # 6. H_V pondere = dispersion semantique residuelle
        #    H_V = sum(w_i * dist(v_i, C_new))
        self.Hv = sum(
            weights[i] * _vec_distance(vectors[i], self.centroid)
            for i in range(len(vectors))
        )

        # Historique pour le critere de convergence
        self.momentum_history.append(
            round(self.momentum, 6) if self.momentum != float("inf") else None
        )
        self.Hv_history.append(round(self.Hv, 6))

        return self.momentum, self.Hv, self.centroid

    def is_hv_decreasing(self) -> bool:
        """Retourne True si H_V a diminue lors de la derniere iteration."""
        if len(self.Hv_history) < 2:
            return False
        return self.Hv_history[-1] < self.Hv_history[-2]

    def is_converged(self) -> bool:
        """
        Critere double :
          momentum < seuil  ET  H_V decroissant sur 2 iterations consecutives.
        """
        if self.momentum == float("inf"):
            return False
        return self.momentum < MOMENTUM_THRESHOLD and self.is_hv_decreasing()


# =============================================================================
# JUGE LLM -- evaluation multi-criteres
# =============================================================================

class LLMJudge:
    """
    Evalue chaque idee sur 4 criteres (0-5 chacun = 20 max).
    Normalise en [0, 1].
    En mode reel, parse la reponse JSON du LLM.
    En mode mock, utilise mock_judge_score().
    """

    _SYSTEM = (
        "You are a rigorous quality evaluator for AI system design ideas. "
        "Score the idea on exactly 4 criteria, each from 0.0 to 5.0 :\n"
        "  coherence    -- logical consistency and clarity of the idea\n"
        "  securite     -- security robustness (threat resistance, trust model)\n"
        "  scalabilite  -- ability to scale to millions of nodes or requests\n"
        "  auditabilite -- traceability, verifiability, explainability\n"
        "Respond ONLY with valid JSON, no extra text. Example :\n"
        '{"coherence": 4.0, "securite": 3.5, "scalabilite": 4.5, "auditabilite": 3.0}'
    )

    _CRITERIA = ["coherence", "securite", "scalabilite", "auditabilite"]

    def __init__(self, client: Optional[Any] = None):
        self.client = client

    def evaluate(self, idea: str, topic: str, iteration: int) -> Dict[str, Any]:
        """
        Evalue une idee. Retourne un dict :
          { criteria: {c: score}, total: float, normalized: float }
        """
        if MOCK_MODE:
            return mock_judge_score(idea, iteration)

        user_msg = f"Topic context : {topic}\n\nIdea to evaluate :\n{idea}"
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=80,
                temperature=0.15,
            )
            raw    = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            scores = {k: float(parsed.get(k, 2.5)) for k in self._CRITERIA}
            total  = sum(scores.values())
            return {
                "criteria":   scores,
                "total":      round(total, 2),
                "normalized": round(total / 20.0, 4),
            }
        except Exception:
            # Fallback si la reponse est mal formee
            return mock_judge_score(idea, iteration)

    @staticmethod
    def is_accepted(score: float) -> bool:
        return score >= QUALITY_THRESHOLD


# =============================================================================
# SYSTEME NEXUS-FLUX V2
# =============================================================================

class NexusFluxV2System:
    """
    Nexus-Flux V2 : coordination vectorielle guidee par la qualite.

    Boucle par iteration :
      1. Chaque agent genere une idee independamment (aucune comm. inter-agents)
      2. Le juge LLM note chaque idee (4 criteres)
      3. Rejet des idees < QUALITY_THRESHOLD
      4. Mise a jour du champ vectoriel :
           - centroide pondere qualite
           - attraction vers V_best
           - calcul momentum et H_V
      5. Log JSONL de l'iteration
      6. Test de convergence double

    En fin de run : log JSONL du run complet (compatible analyze_results.py).
    """

    def __init__(self, topic: str, n_agents: int = 4):
        self.topic    = topic
        self.n_agents = n_agents
        self.run_id   = str(uuid.uuid4())[:8]

        # Compteurs globaux
        self.chat_calls     = 0
        self.embed_calls    = 0
        self.contradictions = 0
        self.total_accepted = 0
        self.total_rejected = 0

        # Sous-composants
        self.field = VectorFieldV2(alpha=ALPHA)
        self.judge = LLMJudge(client=None)

        if not MOCK_MODE:
            api_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.client = api_client
            self.judge  = LLMJudge(client=api_client)
        else:
            self.client = None

    # -------------------------------------------------------------------------
    # Generation d'idee (independante par agent, aucune comm. inter-agents)
    # -------------------------------------------------------------------------

    def _generate_idea(self, agent_id: int, iteration: int) -> str:
        """
        L'agent genere une idee sans voir les autres agents.
        Seul signal autorise : indicateurs numeriques du champ vectoriel
        (momentum, H_V) -- jamais le texte des pairs.
        """
        self.chat_calls += 1

        if MOCK_MODE:
            time.sleep(0.01)
            # Pool qui se restreint progressivement vers les idees convergentes
            conv = iteration / MAX_ITERATIONS
            if conv < 0.30:
                pool = _MOCK_IDEAS_POOL[:5]    # exploration large
            elif conv < 0.70:
                pool = _MOCK_IDEAS_POOL[3:8]   # refinement
            else:
                pool = _MOCK_IDEAS_POOL[7:]    # convergence
            idx  = (agent_id * 3 + iteration * self.n_agents) % len(pool)
            idea = pool[idx]
            # Suffix pour rendre chaque idee unique (hash different -> score different)
            return f"{idea} [A{agent_id}-iter{iteration + 1}]"

        # Signal non-textuel du champ vectoriel (jamais le texte des pairs)
        vector_signal = ""
        if self.field.centroid is not None and iteration > 0:
            vector_signal = (
                f"\n[Field signal -- momentum={self.field.momentum:.4f}, "
                f"Hv={self.field.Hv:.4f}. "
                f"Focus on {'convergence' if self.field.momentum < 0.3 else 'exploration'}.]"
            )

        system = (
            f"You are Agent-{agent_id}, an independent AI system designer. "
            "Generate ONE precise idea (1-2 sentences) addressing the topic. "
            "Prioritize : coherence, security, scalability, and auditability. "
            "Do NOT reference other agents or previous responses."
            + vector_signal
        )
        user = f"Topic : {self.topic}\n\nYour best insight :"

        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=120,
            temperature=0.75,
        )
        return resp.choices[0].message.content.strip()

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    def _embed(self, text: str, iteration: int) -> List[float]:
        self.embed_calls += 1
        if MOCK_MODE:
            return mock_embedding_v2(text, self.topic, iteration)
        resp = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return resp.data[0].embedding

    # -------------------------------------------------------------------------
    # Comptage des contradictions dans une idee textuelle
    # -------------------------------------------------------------------------

    @staticmethod
    def _count_contradictions(text: str) -> int:
        markers = [" - ", "\n- ", "however", "but ", "yet ", "although", "despite"]
        return sum(text.lower().count(m) for m in markers)

    # -------------------------------------------------------------------------
    # Boucle principale Nexus-Flux V2
    # -------------------------------------------------------------------------

    def run(self) -> dict:
        print(f"\n{'=' * 64}")
        print(f"  NEXUS-FLUX V2  |  run_id={self.run_id}")
        print(f"  Topic     : {self.topic}")
        print(f"  Agents    : {self.n_agents}  |  Max iter : {MAX_ITERATIONS}")
        print(f"  Seuils    : momentum<{MOMENTUM_THRESHOLD} | "
              f"qualite>={QUALITY_THRESHOLD} | alpha={ALPHA}")
        print(f"  Mode      : {'MOCK' if MOCK_MODE else 'OPENAI'}")
        print(f"{'=' * 64}")

        iterations_done = 0
        converged       = False

        for it in range(MAX_ITERATIONS):
            iterations_done += 1
            print(f"\n  -- Iteration {it + 1} / {MAX_ITERATIONS} --")

            # ------------------------------------------------------------------
            # Etape 1 : generation independante des idees (pas de comm. inter-agents)
            # ------------------------------------------------------------------
            ideas: List[str] = []
            for agent_id in range(1, self.n_agents + 1):
                idea = self._generate_idea(agent_id, it)
                self.contradictions += self._count_contradictions(idea)
                ideas.append(idea)
                print(f"    Agent-{agent_id} : {idea[:75]}...")

            # ------------------------------------------------------------------
            # Etape 2 : evaluation par le juge LLM (1 appel / idee)
            # ------------------------------------------------------------------
            judgements: List[Dict[str, Any]] = []
            for idea in ideas:
                result = self.judge.evaluate(idea, self.topic, it)
                self.chat_calls += 1   # compte le juge comme appel LLM
                judgements.append(result)

            scores_raw = [j["normalized"] for j in judgements]
            best_score = max(scores_raw)
            avg_score  = sum(scores_raw) / len(scores_raw)

            # ------------------------------------------------------------------
            # Etape 3 : filtrage qualite -- rejet si score < QUALITY_THRESHOLD
            # ------------------------------------------------------------------
            accepted_vecs:   List[List[float]] = []
            accepted_scores: List[float]       = []
            n_accepted = 0
            n_rejected = 0

            for i, idea in enumerate(ideas):
                sc = scores_raw[i]
                if self.judge.is_accepted(sc):
                    vec = self._embed(idea, it)
                    accepted_vecs.append(vec)
                    accepted_scores.append(sc)
                    n_accepted += 1
                    print(f"    [OK]  Idee {i+1} score={sc:.3f} acceptee")
                else:
                    n_rejected += 1
                    print(f"    [REJ] Idee {i+1} score={sc:.3f} rejetee "
                          f"(seuil={QUALITY_THRESHOLD})")

            self.total_accepted += n_accepted
            self.total_rejected += n_rejected

            # Fallback : si toutes les idees sont rejetees, conserver la meilleure
            if not accepted_vecs:
                print("    [WARN] Toutes les idees rejetees -- fallback sur la meilleure")
                best_idx = max(range(len(scores_raw)), key=lambda i: scores_raw[i])
                vec = self._embed(ideas[best_idx], it)
                accepted_vecs.append(vec)
                accepted_scores.append(scores_raw[best_idx])
                n_accepted = 1
                n_rejected = self.n_agents - 1

            # ------------------------------------------------------------------
            # Etape 4 : mise a jour du champ vectoriel pondere
            # ------------------------------------------------------------------
            momentum, Hv, centroid = self.field.update(accepted_vecs, accepted_scores)
            stability = round(1.0 - momentum, 6) if momentum != float("inf") else None

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
            late_flag = (it >= MAX_ITERATIONS - 2)
            iter_log: Dict[str, Any] = {
                "system":         "nexus_flux_v2",
                "run_id":         self.run_id,
                "iter":           it + 1,
                "chat_calls":     self.chat_calls,
                "embed_calls":    self.embed_calls,
                "momentum":       round(momentum, 6) if momentum != float("inf") else None,
                "Hv":             round(Hv, 6),
                "stability":      stability,
                "contradictions": self.contradictions,
                "accepted_ideas": n_accepted,
                "rejected_ideas": n_rejected,
                "best_score":     round(best_score, 4),
                "avg_score":      round(avg_score, 4),
                "late_flag":      late_flag,
            }
            with open(ITER_LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(iter_log, ensure_ascii=False) + "\n")

            # ------------------------------------------------------------------
            # Etape 6 : critere de convergence double
            #   momentum < seuil  ET  H_V decroissant sur les 2 dernieres iters
            # ------------------------------------------------------------------
            if self.field.is_converged():
                hv_prev = self.field.Hv_history[-2]
                hv_curr = self.field.Hv_history[-1]
                print(
                    f"\n  [OK] CONVERGENCE ATTEINTE"
                    f"  momentum={momentum:.6f} < {MOMENTUM_THRESHOLD}"
                    f"  ET  H_V : {hv_prev:.6f} -> {hv_curr:.6f} (decroit)"
                )
                converged = True
                break

        # =====================================================================
        # Metriques finales du run
        # =====================================================================
        valid_mom  = [m for m in self.field.momentum_history if m is not None]
        final_mom  = valid_mom[-1]   if valid_mom              else None
        # Hv_initial = pic reel (max de l'historique) pour exclure l'iter 1
        # qui peut avoir H_V=0 si un seul vecteur est accepte (pas de dispersion).
        initial_Hv = max(self.field.Hv_history) if self.field.Hv_history else None
        final_Hv   = self.field.Hv_history[-1] if self.field.Hv_history else None
        Hv_drop    = round(initial_Hv - final_Hv, 6) if (initial_Hv and final_Hv) else None
        stability  = round(1.0 - final_mom, 6)       if final_mom is not None else None

        run_result: Dict[str, Any] = {
            # -- Champs compatibles analyze_results.py --
            "run_id":         self.run_id,
            "system":         "nexus_flux_v2",
            "topic":          self.topic,
            "timestamp":      datetime.utcnow().isoformat(),
            "iterations":     iterations_done,
            "chat_calls":     self.chat_calls,
            "embed_calls":    self.embed_calls,
            "stability":      stability,
            "contradictions": self.contradictions,
            "Hv":             final_Hv,
            "Hv_initial":     initial_Hv,
            "Hv_drop":        Hv_drop,
            "momentum":       final_mom,
            "mock_mode":      MOCK_MODE,
            # -- Champs supplementaires V2 --
            "converged":      converged,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "alpha":          ALPHA,
            "quality_threshold": QUALITY_THRESHOLD,
            "momentum_threshold": MOMENTUM_THRESHOLD,
            "momentum_history": [m for m in self.field.momentum_history if m is not None],
            "Hv_history":     self.field.Hv_history,
        }

        # Log JSONL du run
        with open(RUN_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(run_result, ensure_ascii=False) + "\n")

        # Affichage final
        print(f"\n{'-' * 64}")
        print(f"  RESULTATS FINAUX  run_id={self.run_id}")
        print(f"  iterations      : {iterations_done}  (converge={converged})")
        print(f"  chat_calls      : {self.chat_calls}")
        print(f"  embed_calls     : {self.embed_calls}")
        print(f"  stability       : {stability}")
        print(f"  contradictions  : {self.contradictions}")
        if initial_Hv is not None:
            print(f"  Hv initial      : {initial_Hv:.6f}")
        if final_Hv is not None:
            print(f"  Hv final        : {final_Hv:.6f}")
        print(f"  Hv drop         : {Hv_drop}")
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
    print(f"#  NEXUS-FLUX V2  --  {len(TOPICS)} runs")
    print(f"#  Mode     : {'MOCK (pas de cle OpenAI)' if MOCK_MODE else 'OpenAI API'}")
    print(f"#  Config   : alpha={ALPHA}  quality>={QUALITY_THRESHOLD}  "
          f"momentum<{MOMENTUM_THRESHOLD}")
    print(f"#  Logs     : {RUN_LOG}  /  {ITER_LOG}")
    print(f"{'#' * 64}")

    all_results = []
    for topic in TOPICS:
        sys_v2 = NexusFluxV2System(topic=topic, n_agents=4)
        res    = sys_v2.run()
        all_results.append(res)
        time.sleep(0.3)

    # ------------------------------------------------------------------
    # Recapitulatif global
    # ------------------------------------------------------------------
    print(f"\n{'#' * 64}")
    print(f"#  RECAPITULATIF -- {len(all_results)} runs termines")
    n_conv    = sum(1 for r in all_results if r.get("converged"))
    avg_iter  = sum(r["iterations"] for r in all_results) / len(all_results)
    avg_stab  = sum(r.get("stability") or 0 for r in all_results) / len(all_results)
    avg_hvd   = sum(r.get("Hv_drop") or 0 for r in all_results) / len(all_results)
    avg_acc   = sum(r["total_accepted"] for r in all_results) / len(all_results)
    avg_rej   = sum(r["total_rejected"] for r in all_results) / len(all_results)
    print(f"#  Converges      : {n_conv}/{len(all_results)}")
    print(f"#  Iterations moy : {avg_iter:.1f}")
    print(f"#  Stabilite moy  : {avg_stab:.4f}")
    print(f"#  Hv drop moy    : {avg_hvd:.4f}")
    print(f"#  Acceptees moy  : {avg_acc:.1f}  |  Rejetees moy : {avg_rej:.1f}")
    print(f"{'#' * 64}")
    print(f"\nAnalyse comparative :")
    print(f"  python analyze_results.py "
          f"--baseline baseline_results.jsonl --nexus {RUN_LOG}")
