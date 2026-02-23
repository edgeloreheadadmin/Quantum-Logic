"""
cognitive.py — Cognitive Processing Unit (CPU)

A self-modifying cognitive architecture that:
  • Maintains attention allocation across topics
  • Builds a live knowledge graph from LLM conversations
  • Tracks cognitive load, mode (focused / diffuse / creative)
  • Dynamically upgrades its own processing modules when the LLM
    returns relevant architectural insights
  • Feeds its state into the NeuroEvolution fitness signal
"""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .core import SharedVirtualStorage, SystemBus


# ─── Cognitive modes ─────────────────────────────────────────────────────────

class CogMode(Enum):
    FOCUSED  = auto()   # single-topic deep attention
    DIFFUSE  = auto()   # broad parallel attention
    CREATIVE = auto()   # low-inhibition exploration
    RESET    = auto()   # clearing / consolidation phase


# ─── Knowledge graph ─────────────────────────────────────────────────────────

@dataclass
class KNode:
    concept: str
    weight: float = 1.0          # relevance accumulated
    last_seen: float = field(default_factory=time.time)
    access_count: int = 0

@dataclass
class KEdge:
    src: str
    dst: str
    strength: float = 1.0
    co_occur: int = 1


class KnowledgeGraph:
    """Incrementally built semantic graph from conversation tokens."""

    def __init__(self) -> None:
        self.nodes: Dict[str, KNode] = {}
        self.edges: Dict[Tuple[str, str], KEdge] = {}
        self._lock = threading.RLock()

    def ingest(self, text: str, weight: float = 1.0) -> None:
        """Extract noun/code concepts and update graph."""
        # Very simple extraction: long tokens that look like identifiers
        raw = re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", text)
        # Deduplicate, lowercase
        tokens = list(dict.fromkeys(t.lower() for t in raw))[:30]
        if not tokens:
            return
        with self._lock:
            for t in tokens:
                if t in self.nodes:
                    self.nodes[t].weight += weight * 0.1
                    self.nodes[t].access_count += 1
                    self.nodes[t].last_seen = time.time()
                else:
                    self.nodes[t] = KNode(concept=t, weight=weight)
            # Co-occurrence edges (sliding window of 3)
            for i in range(len(tokens)):
                for j in range(i + 1, min(i + 4, len(tokens))):
                    key = (tokens[i], tokens[j])
                    if key in self.edges:
                        self.edges[key].co_occur += 1
                        self.edges[key].strength = min(10.0, self.edges[key].strength + 0.1)
                    else:
                        self.edges[key] = KEdge(src=tokens[i], dst=tokens[j])

    def top_concepts(self, n: int = 10) -> List[KNode]:
        with self._lock:
            return sorted(self.nodes.values(), key=lambda n: n.weight, reverse=True)[:n]

    def top_edges(self, n: int = 10) -> List[KEdge]:
        with self._lock:
            return sorted(self.edges.values(), key=lambda e: e.strength, reverse=True)[:n]

    def stats(self) -> dict:
        with self._lock:
            return {"nodes": len(self.nodes), "edges": len(self.edges)}


# ─── Processing Module ────────────────────────────────────────────────────────

@dataclass
class CogModule:
    name: str
    description: str
    capacity: float = 1.0      # 0–1, upgradeable
    load: float = 0.0          # current utilisation
    version: int = 1
    active: bool = True
    upgrade_log: List[str] = field(default_factory=list)

    def upgrade(self, delta: float = 0.05, note: str = "") -> None:
        self.capacity = min(2.0, self.capacity + delta)
        self.version += 1
        if note:
            self.upgrade_log.append(f"v{self.version}: {note}")


# ─── Cognitive Processing Unit ───────────────────────────────────────────────

class CognitiveCPU:
    """
    Orchestrates attention, knowledge, and cognitive modules.
    Self-upgrades based on pattern-matched LLM responses.
    """

    LOAD_DECAY = 0.92           # cognitive load decay per tick
    ATTENTION_DECAY = 0.98      # attention decay per tick

    # Keyword → module_name mapping for upgrade triggers
    _UPGRADE_PATTERNS: Dict[str, str] = {
        r"\bneural\b|\bnetwork\b|\bdeep learning\b": "neural_processor",
        r"\bmemory\b|\bcache\b|\bbuffer\b":            "memory_manager",
        r"\battention\b|\btransformer\b|\bhead\b":     "attention_unit",
        r"\bevolution\b|\bmutation\b|\bgenetic\b":     "evolution_tracker",
        r"\bgraph\b|\bknowledge\b|\bontology\b":       "knowledge_engine",
        r"\boptimis[ei]\b|\bgradient\b|\bloss\b":      "optimizer",
        r"\bquantum\b|\bqubit\b|\bsuperposition\b":    "quantum_bridge",
    }

    def __init__(self, bus: SystemBus, storage: SharedVirtualStorage) -> None:
        self.bus = bus
        self.storage = storage
        self._lock = threading.RLock()

        self.mode = CogMode.DIFFUSE
        self.cognitive_load: float = 0.0   # 0–1
        self.creativity: float = 0.5
        self.focus_depth: float = 0.5

        self.attention: Dict[str, float] = defaultdict(float)  # topic → weight
        self.knowledge = KnowledgeGraph()

        self.modules: Dict[str, CogModule] = {}
        self._init_modules()

        # History
        self.load_history: deque[float] = deque(maxlen=120)
        self.mode_history: deque[str] = deque(maxlen=120)
        self.upgrade_history: deque[dict] = deque(maxlen=50)
        self.thought_stream: deque[str] = deque(maxlen=30)  # internal monologue

        # Persist state
        self._persist()
        bus.publish("cpu.init", {"modules": list(self.modules.keys())})

    # ── Module init ───────────────────────────────────────────────────────────

    def _init_modules(self) -> None:
        specs = [
            ("attention_unit",    "Multi-head attention allocation",   1.0),
            ("memory_manager",    "Working memory & cache steering",   1.0),
            ("neural_processor",  "Neural layer coordination",         1.0),
            ("evolution_tracker", "Fitness signal integration",        0.8),
            ("knowledge_engine",  "Semantic graph maintenance",        0.9),
            ("optimizer",         "Gradient & loss management",        0.8),
            ("quantum_bridge",    "Quantum state interface (future)",  0.5),
            ("meta_learner",      "Self-modification orchestration",   0.7),
        ]
        for name, desc, cap in specs:
            self.modules[name] = CogModule(name=name, description=desc, capacity=cap)

    # ── Processing ────────────────────────────────────────────────────────────

    def process_exchange(
        self,
        user_text: str,
        assistant_text: str,
        npu_metrics: Optional[dict] = None,
    ) -> dict:
        """
        Called after each LLM round-trip.
        Updates all cognitive state and returns a summary.
        """
        with self._lock:
            # 1. Knowledge graph update
            self.knowledge.ingest(user_text, weight=0.8)
            self.knowledge.ingest(assistant_text, weight=1.2)

            # 2. Attention allocation
            combined = (user_text + " " + assistant_text).lower()
            for concept, node in list(self.knowledge.nodes.items())[:20]:
                if concept in combined:
                    self.attention[concept] = min(
                        1.0, self.attention[concept] + node.weight * 0.05
                    )

            # Decay all attention
            for k in list(self.attention.keys()):
                self.attention[k] *= self.ATTENTION_DECAY
                if self.attention[k] < 0.01:
                    del self.attention[k]

            # 3. Cognitive load
            token_count = len(assistant_text.split())
            load_spike = min(0.4, token_count / 500.0)
            self.cognitive_load = min(1.0, self.cognitive_load * self.LOAD_DECAY + load_spike)
            self.load_history.append(self.cognitive_load)

            # 4. Mode selection
            if self.cognitive_load > 0.75:
                self.mode = CogMode.FOCUSED
            elif self.cognitive_load < 0.25:
                self.mode = CogMode.CREATIVE
            else:
                self.mode = CogMode.DIFFUSE
            self.mode_history.append(self.mode.name)

            # 5. Module load distribution
            for name, mod in self.modules.items():
                if mod.active:
                    mod.load = min(1.0, self.cognitive_load * mod.capacity * 0.8 +
                                   np.random.rand() * 0.1)

            # 6. Check for upgrade triggers
            upgrades = self._check_upgrades(assistant_text)

            # 7. Generate thought
            thought = self._generate_thought(user_text, assistant_text, upgrades)
            self.thought_stream.append(thought)

            # 8. NPU feedback loop
            if npu_metrics:
                util = npu_metrics.get("utilization", 0.0)
                self.modules["neural_processor"].load = util / 100.0

        self._persist()
        result = {
            "mode": self.mode.name,
            "cognitive_load": self.cognitive_load,
            "upgrades": upgrades,
            "top_attention": self._top_attention(5),
            "knowledge": self.knowledge.stats(),
            "thought": thought,
        }
        self.bus.publish("cpu.process", result)
        return result

    # ── Upgrade triggers ──────────────────────────────────────────────────────

    def _check_upgrades(self, text: str) -> List[str]:
        upgraded: List[str] = []
        for pattern, module_name in self._UPGRADE_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                mod = self.modules.get(module_name)
                if mod and mod.version < 20:   # cap upgrades
                    mod.upgrade(delta=0.05, note=f"LLM mention: '{pattern[:20]}'")
                    upgraded.append(module_name)
                    self.upgrade_history.append({
                        "module": module_name,
                        "version": mod.version,
                        "capacity": mod.capacity,
                        "ts": time.time(),
                    })
                    self.bus.publish("cpu.upgrade", {
                        "module": module_name,
                        "version": mod.version,
                        "capacity": mod.capacity,
                    })
        return upgraded

    # ── Internal monologue ────────────────────────────────────────────────────

    def _generate_thought(
        self, user: str, assistant: str, upgrades: List[str]
    ) -> str:
        """Synthesise a brief internal thought describing the cognitive state."""
        top = self._top_attention(3)
        top_str = ", ".join(f"'{c}'" for c, _ in top) if top else "nothing specific"

        lines = [f"[{self.mode.name}] Cognitive load: {self.cognitive_load:.2f}"]
        lines.append(f"Attending to: {top_str}")
        if upgrades:
            lines.append(f"Upgraded modules: {', '.join(upgrades)}")
        kg = self.knowledge.stats()
        lines.append(f"Knowledge graph: {kg['nodes']} nodes, {kg['edges']} edges")
        return " | ".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _top_attention(self, n: int = 5) -> List[Tuple[str, float]]:
        return sorted(self.attention.items(), key=lambda x: x[1], reverse=True)[:n]

    def _persist(self) -> None:
        # Persist module capacities as a vector
        caps = np.array(
            [m.capacity for m in self.modules.values()], dtype=np.float32
        )
        if self.storage.get("cpu.module_caps") is None:
            self.storage.alloc("cpu.module_caps", caps.shape, owner="cpu", init=caps)
        else:
            self.storage.put("cpu.module_caps", caps)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            module_info = {
                name: {"cap": f"{m.capacity:.2f}", "load": f"{m.load:.2f}",
                       "v": m.version, "active": m.active}
                for name, m in self.modules.items()
            }
        return {
            "mode": self.mode.name,
            "cognitive_load": self.cognitive_load,
            "avg_load": float(np.mean(list(self.load_history))) if self.load_history else 0.0,
            "attention_topics": len(self.attention),
            "top_attention": self._top_attention(5),
            "knowledge_nodes": self.knowledge.stats()["nodes"],
            "knowledge_edges": self.knowledge.stats()["edges"],
            "modules": module_info,
            "upgrade_events": len(self.upgrade_history),
            "last_thought": list(self.thought_stream)[-1] if self.thought_stream else "",
        }
