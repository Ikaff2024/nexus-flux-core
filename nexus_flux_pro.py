"""
nexus_flux_pro.py
=================
Nexus-Flux : coordination vectorielle non-anthropomorphique.

Les agents NE communiquent PAS entre eux par texte.
Chaque agent produit une idee -> transformee en vecteur (embedding).
La coordination emerge du champ vectoriel partage :
  - Centroide C  = moyenne des vecteurs courants
  - Momentum     = ||C_t - C_{t-1}||
  - H_V          = moyenne des distances au centroide (entropie vectorielle)
  - Convergence  : momentum < seuil  OU  10 iterations

Mode mock automatique si OPENAI_API_KEY est absent.
"""

import os
import json
import time
import uuid
import math
import random
import hashlib
from datetime import datetime
from typing import List, Optional, Tuple

# -- Dependance optionnelle ----------------------------------------------------
try:
    from openai import OpenAI
    _OPENAI_IMPORTABLE = True
except ImportError:
    _OPENAI_IMPORTABLE = False

MOCK_MODE: bool = not _OPENAI_IMPORTABLE or not os.getenv("OPENAI_API_KEY")
LOG_FILE = "nexus_flux_results.jsonl"

# -- Hyperparametres Nexus-Flux ------------------------------------------------
MOMENTUM_THRESHOLD = 0.05   # seuil de convergence
MAX_ITERATIONS     = 10     # garde-fou
EMBED_DIM          = 256    # dimension mock (text-embedding-3-small = 1536)

# -- Idees mock par domaine ----------------------------------------------------
_MOCK_IDEAS_POOL = [
    # Convergence lente -> represente des concepts proches
    "Decentralized coordination reduces latency through parallel consensus mechanisms.",
    "Vector-space coordination eliminates linguistic ambiguity in agent communication.",
    "Shared embedding spaces enable implicit consensus without explicit message passing.",
    "Geometric convergence in high-dimensional spaces mirrors biological neural synchrony.",
    "Entropy reduction in vector fields signals emergent collective intelligence.",
    "Centroid stability indicates semantic alignment across independent cognitive units.",
    "Momentum decay in embedding space corresponds to decreasing information surprise.",
    "Non-anthropomorphic coordination outperforms text-based protocols at scale.",
    "Dimensional compression of idea vectors preserves topological relationships.",
    "Convergent manifolds in latent space represent robust collective knowledge states.",
]


# -- Utilitaires vectoriels ----------------------------------------------------

def _hash_seed(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 31)


def mock_embedding(text: str, noise_scale: float = 0.0, dim: int = EMBED_DIM) -> List[float]:
    """
    Embedding deterministe + bruit optionnel pour simuler la variance
    inter-agents a une meme iteration.
    """
    rng = random.Random(_hash_seed(text))
    base = [rng.gauss(0, 1) for _ in range(dim)]
    if noise_scale > 0:
        noise_rng = random.Random()  # non-deterministe -> diversite reelle
        base = [b + noise_rng.gauss(0, noise_scale) for b in base]
    norm = math.sqrt(sum(x * x for x in base)) or 1.0
    return [x / norm for x in base]


def _vec_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 1.0


def _vec_distance(v1: List[float], v2: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def _centroid(vectors: List[List[float]]) -> List[float]:
    """Moyenne arithmetique des vecteurs (non normalisee -> magnitude porteuse de sens)."""
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[d] for v in vectors) / n for d in range(dim)]


# -- Etat partage Nexus-Flux ---------------------------------------------------

class VectorField:
    """
    L'etat partage du systeme Nexus-Flux.
    Stocke les vecteurs de tous les agents a chaque iteration.
    Calcule centroide, momentum et entropie vectorielle.
    """

    def __init__(self):
        self.vectors: List[List[float]] = []   # vecteurs de l'iteration courante
        self.prev_centroid: Optional[List[float]] = None
        self.centroid: Optional[List[float]] = None
        self.momentum: float = float("inf")
        self.Hv: float = float("inf")
        self.momentum_history: List[float] = []
        self.Hv_history: List[float] = []

    def update(self, new_vectors: List[List[float]]) -> Tuple[float, float]:
        """
        Integre une nouvelle generation de vecteurs.
        Retourne (momentum, Hv).
        """
        self.vectors = new_vectors
        self.prev_centroid = self.centroid

        # 1. Centroide global
        self.centroid = _centroid(new_vectors)

        # 2. Momentum = ||C_t - C_{t-1}||
        if self.prev_centroid is not None:
            self.momentum = _vec_distance(self.centroid, self.prev_centroid)
        else:
            self.momentum = float("inf")

        # 3. H_V = moyenne des distances au centroide
        self.Hv = sum(_vec_distance(v, self.centroid) for v in new_vectors) / len(new_vectors)

        self.momentum_history.append(
            round(self.momentum, 6) if self.momentum != float("inf") else None
        )
        self.Hv_history.append(round(self.Hv, 6))

        return self.momentum, self.Hv


# -- Classe principale ---------------------------------------------------------

class NexusFluxSystem:
    """
    Nexus-Flux : chaque agent genere une idee independamment,
    la transforme en vecteur. La convergence emerge de la dynamique
    du champ vectoriel partage -- sans aucun echange textuel entre agents.
    """

    def __init__(self, topic: str, n_agents: int = 4):
        self.topic = topic
        self.n_agents = n_agents
        self.run_id = str(uuid.uuid4())[:8]

        self.chat_calls = 0
        self.embed_calls = 0
        self.contradictions = 0

        self.field = VectorField()
        self.ideas_per_iteration: List[List[str]] = []

        if not MOCK_MODE:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.client = None

    # -- Appels LLM/Embeddings -------------------------------------------------

    def _generate_idea(self, agent_id: int, iteration: int) -> str:
        """
        L'agent genere une idee SANS voir les autres agents.
        Il ne recoit que le topic et (en iterations avancees) un signal
        de convergence vectorielle agrege -- jamais le texte des pairs.
        """
        self.chat_calls += 1

        if MOCK_MODE:
            time.sleep(0.015)
            # Mock : ideas convergent progressivement avec les iterations
            # Facteur de convergence croissant -> topics plus proches
            convergence_factor = iteration / MAX_ITERATIONS  # 0 -> 1
            if convergence_factor < 0.3:
                # Phase 1 : exploration diversifiee
                pool_indices = list(range(len(_MOCK_IDEAS_POOL)))
            elif convergence_factor < 0.7:
                # Phase 2 : reduction de la variance
                pool_indices = list(range(3, len(_MOCK_IDEAS_POOL)))
            else:
                # Phase 3 : convergence
                pool_indices = [5, 6, 7]

            idx = pool_indices[(agent_id + iteration * self.n_agents) % len(pool_indices)]
            idea = _MOCK_IDEAS_POOL[idx]
            # Legere variation par agent pour ne pas avoir des vecteurs identiques
            return f"{idea} [Agent-{agent_id} perspective, iteration {iteration + 1}]"

        # Signal vectoriel optionnel (resume non-textuel du centroide precedent)
        centroid_signal = ""
        if self.field.centroid is not None and iteration > 0:
            centroid_signal = (
                f"\n[Vector convergence signal: momentum={self.field.momentum:.4f}, "
                f"Hv={self.field.Hv:.4f} -- adjust your perspective accordingly]"
            )

        system = (
            f"You are Agent-{agent_id}, an autonomous cognitive unit. "
            "Generate ONE concise insight (1-2 sentences) about the topic. "
            "Do NOT reference other agents. Work independently."
            + centroid_signal
        )
        user = f"Topic: {self.topic}\n\nGenerate your best insight."

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=120,
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()

    def _embed(self, text: str, iteration: int = 0, agent_id: int = 0) -> List[float]:
        self.embed_calls += 1
        if MOCK_MODE:
            # Plus on avance dans les iterations, plus les vecteurs convergent
            # Noise scale decroit avec les iterations -> convergence simulee
            noise = max(0.05, 1.5 * (1 - iteration / MAX_ITERATIONS))
            return mock_embedding(text, noise_scale=noise)
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    # -- Comptage des contradictions -------------------------------------------

    @staticmethod
    def _count_contradictions(text: str) -> int:
        markers = [" - ", "\n- ", "however", "but ", "yet ", "although", "despite"]
        return sum(text.lower().count(m) for m in markers)

    # -- Boucle Nexus-Flux -----------------------------------------------------

    def run(self) -> dict:
        print(f"\n{'=' * 60}")
        print(f"  NEXUS-FLUX  |  run_id={self.run_id}")
        print(f"  Topic       : {self.topic}")
        print(f"  Agents      : {self.n_agents}  |  Max iter: {MAX_ITERATIONS}")
        print(f"  Seuil conv. : momentum < {MOMENTUM_THRESHOLD}")
        print(f"  Mode        : {'MOCK' if MOCK_MODE else 'OPENAI'}")
        print(f"{'=' * 60}")

        iterations_done = 0
        converged = False

        for it in range(MAX_ITERATIONS):
            iterations_done += 1
            print(f"\n  -- Iteration {it + 1} / {MAX_ITERATIONS} --")

            # ? Generation independante des idees (AUCUNE communication inter-agents)
            ideas: List[str] = []
            for agent_id in range(1, self.n_agents + 1):
                idea = self._generate_idea(agent_id, it)
                self.contradictions += self._count_contradictions(idea)
                ideas.append(idea)
                print(f"    Agent-{agent_id} idea: {idea[:80]}...")

            self.ideas_per_iteration.append(ideas)

            # ? Transformation idees -> vecteurs (embeddings)
            vectors: List[List[float]] = []
            for agent_id, idea in enumerate(ideas, start=1):
                vec = self._embed(idea, iteration=it, agent_id=agent_id)
                vectors.append(vec)

            # ? Mise a jour du champ vectoriel partage
            momentum, Hv = self.field.update(vectors)

            print(f"    Centroide  : calcule ({len(vectors)} vecteurs, dim={len(vectors[0])})")
            print(f"    Momentum   : {momentum:.6f}" if momentum != float("inf")
                  else f"    Momentum   : inf (premiere iteration)")
            print(f"    H_V        : {Hv:.6f}")

            # ? Critere de convergence
            if momentum != float("inf") and momentum < MOMENTUM_THRESHOLD:
                print(f"\n  [OK] CONVERGENCE  momentum={momentum:.6f} < seuil={MOMENTUM_THRESHOLD}")
                converged = True
                break

        # -- Metriques finales -------------------------------------------------
        valid_momentums = [m for m in self.field.momentum_history if m is not None]
        final_momentum  = valid_momentums[-1] if valid_momentums else None
        initial_Hv      = self.field.Hv_history[0] if self.field.Hv_history else None
        final_Hv        = self.field.Hv_history[-1] if self.field.Hv_history else None
        Hv_drop         = round(initial_Hv - final_Hv, 6) if (initial_Hv and final_Hv) else None
        stability       = round(1.0 - final_momentum, 6) if final_momentum is not None else None

        result = {
            "run_id": self.run_id,
            "system": "nexus_flux",
            "topic": self.topic,
            "timestamp": datetime.utcnow().isoformat(),
            "iterations": iterations_done,
            "chat_calls": self.chat_calls,
            "embed_calls": self.embed_calls,
            "stability": stability,
            "contradictions": self.contradictions,
            "Hv": final_Hv,
            "Hv_initial": initial_Hv,
            "Hv_drop": Hv_drop,
            "momentum": final_momentum,
            "momentum_history": [m for m in self.field.momentum_history if m is not None],
            "Hv_history": self.field.Hv_history,
            "converged": converged,
            "mock_mode": MOCK_MODE,
        }

        # Log JSONL
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"\n{'-' * 60}")
        print(f"  RESULTATS run_id={self.run_id}")
        print(f"  iterations     : {iterations_done}  (converge={converged})")
        print(f"  chat_calls     : {self.chat_calls}")
        print(f"  embed_calls    : {self.embed_calls}")
        print(f"  stability      : {stability}")
        print(f"  contradictions : {self.contradictions}")
        print(f"  Hv initial     : {initial_Hv:.6f}" if initial_Hv else "  Hv initial     : N/A")
        print(f"  Hv final       : {final_Hv:.6f}"   if final_Hv   else "  Hv final       : N/A")
        print(f"  Hv drop        : {Hv_drop}")
        print(f"  momentum final : {final_momentum}")
        print(f"{'-' * 60}")
        print(f"  -> Logge dans {LOG_FILE}")

        return result


# -- Point d'entree ------------------------------------------------------------

if __name__ == "__main__":
    TOPICS = [
        "How to design a resilient distributed AI coordination system",
        "Optimizing multi-agent convergence efficiency under resource constraints",
        "Balancing security and performance in real-time AI inference pipelines",
    ]

    print(f"\n{'#' * 60}")
    print(f"#  NEXUS-FLUX  -  {len(TOPICS)} runs")
    print(f"#  Mode : {'MOCK (pas de cle OpenAI)' if MOCK_MODE else 'OpenAI API'}")
    print(f"{'#' * 60}")

    all_results = []
    for topic in TOPICS:
        system = NexusFluxSystem(topic=topic, n_agents=4)
        result = system.run()
        all_results.append(result)
        time.sleep(0.3)

    print(f"\n{'#' * 60}")
    print(f"#  TOUS LES RUNS TERMINES  -  {len(all_results)} resultats dans {LOG_FILE}")
    print(f"{'#' * 60}")
