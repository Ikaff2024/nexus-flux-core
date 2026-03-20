"""
baseline_multiagent_pro.py
==========================
Systeme multi-agent textuel classique (architecture de reference).
Les agents communiquent par texte via une synthese partagee.

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
from typing import Optional, List

# -- Dependance optionnelle ----------------------------------------------------
try:
    from openai import OpenAI
    _OPENAI_IMPORTABLE = True
except ImportError:
    _OPENAI_IMPORTABLE = False

MOCK_MODE: bool = not _OPENAI_IMPORTABLE or not os.getenv("OPENAI_API_KEY")
LOG_FILE = "baseline_results.jsonl"
EMBED_DIM = 256  # dimension reduite pour le mock

# -- Contenu mock realiste -----------------------------------------------------
_MOCK_AGENT_TEMPLATES = [
    "From a systems perspective, {topic} requires robust fault-tolerance mechanisms "
    "and clear ownership boundaries between components. Decentralization reduces "
    "bottlenecks - but introduces consistency challenges.",
    "Performance-wise, {topic} benefits from asynchronous pipelines and lazy evaluation. "
    "Resource contention is the primary bottleneck; however, smart caching can mitigate "
    "this - yet adds operational complexity.",
    "Security for {topic} demands zero-trust principles and end-to-end encryption. "
    "Every trust boundary is a potential attack surface - we must validate all inputs "
    "but avoid over-engineering the threat model.",
    "User adoption of {topic} hinges on simplicity. Complex architectures alienate "
    "developers; however, premature simplification sacrifices power - balance is key.",
    "Scalability for {topic} points toward event-driven microservices. Horizontal "
    "scaling works well - but stateful components remain a hard constraint to manage.",
]

_MOCK_SYNTHESIS_TEMPLATES = [
    "Synthesis {i}: Converging on a resilient, event-driven architecture for {topic}. "
    "Key tensions: decentralization vs consistency, security vs simplicity. "
    "Recommended path: async coordination with strong ownership contracts.",
    "Synthesis {i}: The group aligns on performance-first design for {topic}, "
    "accepting some operational overhead. Security layers should be composable - "
    "not monolithic. Scalability via horizontal partitioning.",
    "Synthesis {i}: Final convergence toward a hybrid approach for {topic}: "
    "centralized orchestration at the meta-level, decentralized execution at leaf nodes. "
    "Monitoring and observability are non-negotiable.",
]


# -- Utilitaires vectoriels ----------------------------------------------------

def _hash_seed(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 31)


def mock_embedding(text: str, dim: int = EMBED_DIM) -> List[float]:
    """Embedding deterministe base sur le contenu du texte."""
    rng = random.Random(_hash_seed(text))
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1)) or 1.0
    n2 = math.sqrt(sum(b * b for b in v2)) or 1.0
    return dot / (n1 * n2)


# -- Classe principale ---------------------------------------------------------

class BaselineMultiAgent:
    """
    Architecture baseline : boucle sequentielle ou chaque agent
    recoit la synthese precedente et produit une reponse textuelle.
    """

    def __init__(self, topic: str, n_agents: int = 4, max_iterations: int = 5):
        self.topic = topic
        self.n_agents = n_agents
        self.max_iterations = max_iterations
        self.run_id = str(uuid.uuid4())[:8]

        # Compteurs de metriques
        self.chat_calls = 0
        self.embed_calls = 0
        self.contradictions = 0

        # Historique
        self.synthesis_history: List[str] = []
        self.embedding_history: List[List[float]] = []

        if not MOCK_MODE:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.client = None

    # -- Appels LLM/Embeddings -------------------------------------------------

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        self.chat_calls += 1
        if MOCK_MODE:
            time.sleep(0.02)
            # Selection du template selon agent et iteration
            idx = (self.chat_calls - 1) % len(_MOCK_AGENT_TEMPLATES)
            return _MOCK_AGENT_TEMPLATES[idx].format(topic=self.topic)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=220,
            temperature=0.75,
        )
        return response.choices[0].message.content.strip()

    def _synthesize_chat(self, responses: List[str], iteration: int) -> str:
        self.chat_calls += 1
        if MOCK_MODE:
            time.sleep(0.02)
            idx = iteration % len(_MOCK_SYNTHESIS_TEMPLATES)
            return _MOCK_SYNTHESIS_TEMPLATES[idx].format(i=iteration + 1, topic=self.topic)
        combined = "\n".join(f"Agent-{i + 1}: {r}" for i, r in enumerate(responses))
        system = (
            "You are a synthesis coordinator. Merge the following agent responses "
            "into one concise, coherent synthesis (3-4 sentences max). "
            "Highlight consensus and flag remaining tensions."
        )
        user = f"Topic: {self.topic}\n\nAgent responses:\n{combined}"
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=300,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()

    def _embed(self, text: str) -> List[float]:
        self.embed_calls += 1
        if MOCK_MODE:
            return mock_embedding(text)
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

    # -- Boucle principale -----------------------------------------------------

    def run(self) -> dict:
        print(f"\n{'=' * 60}")
        print(f"  BASELINE MULTI-AGENT  |  run_id={self.run_id}")
        print(f"  Topic   : {self.topic}")
        print(f"  Agents  : {self.n_agents}  |  Iterations: {self.max_iterations}")
        print(f"  Mode    : {'MOCK' if MOCK_MODE else 'OPENAI'}")
        print(f"{'=' * 60}")

        synthesis = f"Open discussion on: {self.topic}"
        prev_embedding: Optional[List[float]] = None
        stabilities: List[float] = []

        for it in range(self.max_iterations):
            print(f"\n  -- Iteration {it + 1} / {self.max_iterations} --")

            # Phase 1 : chaque agent lit la synthese et repond par texte
            responses: List[str] = []
            for agent_id in range(1, self.n_agents + 1):
                system_p = (
                    f"You are Agent-{agent_id}, a critical expert. "
                    "Respond concisely (2-3 sentences). Challenge weak points."
                )
                user_p = (
                    f"Topic: {self.topic}\n\n"
                    f"Current synthesis:\n{synthesis}\n\n"
                    "Provide your analysis and most important insight."
                )
                resp = self._chat(system_p, user_p)
                self.contradictions += self._count_contradictions(resp)
                responses.append(resp)
                print(f"    Agent-{agent_id}: {resp[:90]}...")

            # Phase 2 : synthese de toutes les reponses
            synthesis = self._synthesize_chat(responses, it)
            self.contradictions += self._count_contradictions(synthesis)
            self.synthesis_history.append(synthesis)
            print(f"    Synthesis : {synthesis[:100]}...")

            # Phase 3 : mesure de stabilite (similarite cosinus entre syntheses)
            emb = self._embed(synthesis)
            self.embedding_history.append(emb)
            if prev_embedding is not None:
                sim = cosine_similarity(prev_embedding, emb)
                stabilities.append(sim)
                print(f"    Stability : {sim:.4f}")
            prev_embedding = emb

        # -- Metriques finales -------------------------------------------------
        avg_stability = sum(stabilities) / len(stabilities) if stabilities else 0.0

        result = {
            "run_id": self.run_id,
            "system": "baseline",
            "topic": self.topic,
            "timestamp": datetime.utcnow().isoformat(),
            "iterations": self.max_iterations,
            "chat_calls": self.chat_calls,
            "embed_calls": self.embed_calls,
            "stability": round(avg_stability, 6),
            "contradictions": self.contradictions,
            "Hv": None,
            "momentum": None,
            "mock_mode": MOCK_MODE,
            "final_synthesis": synthesis[:600],
        }

        # Log JSONL
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"\n{'-' * 60}")
        print(f"  RESULTATS run_id={self.run_id}")
        print(f"  chat_calls     : {self.chat_calls}")
        print(f"  embed_calls    : {self.embed_calls}")
        print(f"  stability      : {avg_stability:.4f}")
        print(f"  contradictions : {self.contradictions}")
        print(f"  Hv             : null")
        print(f"  momentum       : null")
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
    print(f"#  BASELINE MULTI-AGENT  -  {len(TOPICS)} runs")
    print(f"#  Mode : {'MOCK (pas de cle OpenAI)' if MOCK_MODE else 'OpenAI API'}")
    print(f"{'#' * 60}")

    all_results = []
    for topic in TOPICS:
        system = BaselineMultiAgent(topic=topic, n_agents=4, max_iterations=5)
        result = system.run()
        all_results.append(result)
        time.sleep(0.3)

    print(f"\n{'#' * 60}")
    print(f"#  TOUS LES RUNS TERMINES  -  {len(all_results)} resultats dans {LOG_FILE}")
    print(f"{'#' * 60}")
