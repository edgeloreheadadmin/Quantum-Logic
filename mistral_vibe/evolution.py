"""
evolution.py — Dynamic & Adaptive Neuro-Evolution
NEAT-inspired engine that co-evolves the CyberneticNN and VirtualNPU.

Key concepts
────────────
Genome     : encodes a network topology (node genes + connection genes)
Species    : group of genomes sharing structural similarity
Population : collection of species
Fitness    : evaluated after each LLM exchange (quality + efficiency)

Operations: mutation (add node, add conn, weight perturb), crossover,
            speciation by compatibility distance, champion elitism.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import SystemBus
from .neural import CyberneticNN
from .npu import VirtualNPU


# ─── Genome data structures ───────────────────────────────────────────────────

@dataclass
class NodeGene:
    node_id: int
    layer: int      # 0 = input, -1 = output, else hidden
    activation: str = "relu"
    bias: float = 0.0


@dataclass
class ConnGene:
    innov: int       # global innovation number (tracks historical origin)
    src: int
    dst: int
    weight: float
    enabled: bool = True


@dataclass
class Genome:
    gid: int
    nodes: List[NodeGene] = field(default_factory=list)
    conns: List[ConnGene] = field(default_factory=list)
    fitness: float = 0.0
    adjusted_fitness: float = 0.0
    species_id: int = 0
    age: int = 0

    def node_ids(self) -> List[int]:
        return [n.node_id for n in self.nodes]

    def active_conns(self) -> List[ConnGene]:
        return [c for c in self.conns if c.enabled]

    def param_count(self) -> int:
        return len(self.conns) + len(self.nodes)

    def weight_array(self) -> np.ndarray:
        return np.array([c.weight for c in self.conns], dtype=np.float32)


@dataclass
class Species:
    sid: int
    representative: Genome
    members: List[Genome] = field(default_factory=list)
    best_fitness: float = 0.0
    stagnation: int = 0
    age: int = 0


# ─── Neuro-Evolution Engine ───────────────────────────────────────────────────

class NeuroEvolution:
    """
    Manages a population of Genomes and evolves them each generation.
    After each generation the best genome is pushed to the live
    CyberneticNN and VirtualNPU.
    """

    # Hyper-parameters (NEAT-style)
    POP_SIZE           = 20
    COMPAT_THRESHOLD   = 3.0
    C1, C2, C3         = 1.0, 1.0, 0.4   # excess, disjoint, weight distance coefficients
    WEIGHT_MUT_PROB    = 0.80
    ADD_NODE_PROB      = 0.03
    ADD_CONN_PROB      = 0.05
    DISABLE_PROB       = 0.10
    ELITISM            = 2                # champions carried forward per species
    INTERSPECIES_PROB  = 0.001
    STAGNATION_LIMIT   = 15

    def __init__(
        self,
        bus: SystemBus,
        cnn: CyberneticNN,
        npu: VirtualNPU,
        input_size: int = 16,
        output_size: int = 8,
    ) -> None:
        self.bus = bus
        self.cnn = cnn
        self.npu = npu
        self._lock = threading.RLock()
        self._rng = np.random.default_rng()

        self._innov_counter = 0
        self._genome_counter = 0
        self._innov_cache: Dict[Tuple[int, int], int] = {}

        self.generation = 0
        self.species: List[Species] = []
        self.population: List[Genome] = []
        self.best_fitness_history: deque[float] = deque(maxlen=200)
        self.avg_fitness_history: deque[float] = deque(maxlen=200)
        self.species_count_history: deque[int] = deque(maxlen=200)
        self.mutation_log: deque[dict] = deque(maxlen=100)
        self.best_genome: Optional[Genome] = None

        # Seed population
        input_size = max(input_size, 4)
        output_size = max(output_size, 2)
        self._input_size = input_size
        self._output_size = output_size
        self._seed_population()
        bus.publish("evo.init", {"pop": self.POP_SIZE, "input": input_size, "output": output_size})

    # ── Innovation tracking ───────────────────────────────────────────────────

    def _get_innov(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self._innov_cache:
            self._innov_counter += 1
            self._innov_cache[key] = self._innov_counter
        return self._innov_cache[key]

    def _new_genome(self) -> Genome:
        gid = self._genome_counter
        self._genome_counter += 1
        return Genome(gid=gid)

    # ── Seed population ───────────────────────────────────────────────────────

    def _seed_population(self) -> None:
        for _ in range(self.POP_SIZE):
            g = self._minimal_genome()
            self.population.append(g)
        self._speciate()

    def _minimal_genome(self) -> Genome:
        """Create a minimal fully-connected 2-layer genome."""
        g = self._new_genome()
        for i in range(self._input_size):
            g.nodes.append(NodeGene(node_id=i, layer=0, activation="linear"))
        out_start = self._input_size
        for j in range(self._output_size):
            g.nodes.append(NodeGene(node_id=out_start + j, layer=-1, activation="linear"))
        for i in range(self._input_size):
            for j in range(self._output_size):
                w = float(self._rng.standard_normal() * 0.5)
                g.conns.append(ConnGene(
                    innov=self._get_innov(i, out_start + j),
                    src=i, dst=out_start + j, weight=w,
                ))
        return g

    # ── Compatibility distance ────────────────────────────────────────────────

    def _compat(self, g1: Genome, g2: Genome) -> float:
        max_innov1 = max((c.innov for c in g1.conns), default=0)
        max_innov2 = max((c.innov for c in g2.conns), default=0)
        max_innov = max(max_innov1, max_innov2)

        g1_dict = {c.innov: c for c in g1.conns}
        g2_dict = {c.innov: c for c in g2.conns}

        excess = disjoint = matching = 0
        weight_diff_sum = 0.0
        threshold = min(max_innov1, max_innov2)

        for innov in range(1, max_innov + 1):
            in1 = innov in g1_dict
            in2 = innov in g2_dict
            if in1 and in2:
                matching += 1
                weight_diff_sum += abs(g1_dict[innov].weight - g2_dict[innov].weight)
            elif in1 or in2:
                if innov > threshold:
                    excess += 1
                else:
                    disjoint += 1

        n = max(len(g1.conns), len(g2.conns), 1)
        avg_w = weight_diff_sum / max(matching, 1)
        return self.C1 * excess / n + self.C2 * disjoint / n + self.C3 * avg_w

    # ── Speciation ────────────────────────────────────────────────────────────

    def _speciate(self) -> None:
        # Reset membership
        for sp in self.species:
            sp.members = []

        for g in self.population:
            placed = False
            for sp in self.species:
                if self._compat(g, sp.representative) < self.COMPAT_THRESHOLD:
                    sp.members.append(g)
                    g.species_id = sp.sid
                    placed = True
                    break
            if not placed:
                sid = len(self.species)
                new_sp = Species(sid=sid, representative=g, members=[g])
                g.species_id = sid
                self.species.append(new_sp)

        # Remove empty species; update representatives
        self.species = [sp for sp in self.species if sp.members]
        for sp in self.species:
            sp.representative = self._rng.choice(sp.members)
            best = max(sp.members, key=lambda g: g.fitness)
            if best.fitness > sp.best_fitness:
                sp.best_fitness = best.fitness
                sp.stagnation = 0
            else:
                sp.stagnation += 1
            sp.age += 1

    # ── Mutation ──────────────────────────────────────────────────────────────

    def _mutate(self, g: Genome) -> Genome:
        """In-place mutation; returns the genome."""
        rng = self._rng

        # Weight perturbation
        if rng.random() < self.WEIGHT_MUT_PROB:
            for conn in g.conns:
                if rng.random() < 0.9:
                    conn.weight += float(rng.standard_normal() * 0.2)
                else:
                    conn.weight = float(rng.standard_normal() * 0.5)
            self.mutation_log.append({"op": "weight_perturb", "gid": g.gid, "ts": time.time()})

        # Add connection
        if rng.random() < self.ADD_CONN_PROB and g.nodes:
            src_node = int(rng.choice([n.node_id for n in g.nodes if n.layer != -1]))
            dst_node = int(rng.choice([n.node_id for n in g.nodes if n.layer != 0]))
            if not any(c.src == src_node and c.dst == dst_node for c in g.conns):
                innov = self._get_innov(src_node, dst_node)
                g.conns.append(ConnGene(innov=innov, src=src_node, dst=dst_node,
                                        weight=float(rng.standard_normal() * 0.5)))
                self.mutation_log.append({"op": "add_conn", "gid": g.gid,
                                          "src": src_node, "dst": dst_node, "ts": time.time()})

        # Add node (split an existing connection)
        if rng.random() < self.ADD_NODE_PROB and g.conns:
            conn = g.conns[int(rng.integers(0, len(g.conns)))]
            conn.enabled = False
            new_id = max(n.node_id for n in g.nodes) + 1
            g.nodes.append(NodeGene(node_id=new_id, layer=1, activation="relu"))
            g.conns.append(ConnGene(innov=self._get_innov(conn.src, new_id),
                                    src=conn.src, dst=new_id, weight=1.0))
            g.conns.append(ConnGene(innov=self._get_innov(new_id, conn.dst),
                                    src=new_id, dst=conn.dst, weight=conn.weight))
            self.mutation_log.append({"op": "add_node", "gid": g.gid,
                                      "nid": new_id, "ts": time.time()})

        # Toggle disable
        if rng.random() < self.DISABLE_PROB and g.conns:
            idx = int(rng.integers(0, len(g.conns)))
            g.conns[idx].enabled = not g.conns[idx].enabled

        return g

    # ── Crossover ─────────────────────────────────────────────────────────────

    def _crossover(self, p1: Genome, p2: Genome) -> Genome:
        """Produce a child from two parents (NEAT-style gene alignment)."""
        child = self._new_genome()
        # Nodes from the fitter parent
        fitter = p1 if p1.fitness >= p2.fitness else p2
        child.nodes = [NodeGene(**vars(n)) for n in fitter.nodes]

        p2_dict = {c.innov: c for c in p2.conns}
        for c in p1.conns:
            if c.innov in p2_dict:
                # Matching gene: inherit randomly
                source = c if self._rng.random() < 0.5 else p2_dict[c.innov]
                enabled = c.enabled and p2_dict[c.innov].enabled
                child.conns.append(ConnGene(innov=c.innov, src=source.src, dst=source.dst,
                                             weight=source.weight, enabled=enabled))
            else:
                # Disjoint/excess from fitter parent
                if p1.fitness >= p2.fitness:
                    child.conns.append(ConnGene(**vars(c)))
        return child

    # ── Fitness evaluation ────────────────────────────────────────────────────

    def evaluate_population(self, response_text: str) -> None:
        """
        Assign fitness to each genome based on the latest LLM response.
        Fitness = content quality × efficiency (fewer params = bonus).
        """
        tokens = len(response_text.split())
        code_density = response_text.count("```") * 2
        base_quality = min(1.0, (tokens + code_density) / 300.0)

        for g in self.population:
            # Efficiency bonus for compact genomes
            size_penalty = g.param_count() / 1000.0
            g.fitness = max(0.0, base_quality - size_penalty * 0.1 + self._rng.random() * 0.05)

        # Adjusted fitness (share fitness within species)
        for sp in self.species:
            for g in sp.members:
                g.adjusted_fitness = g.fitness / max(len(sp.members), 1)

    # ── Selection & reproduction ──────────────────────────────────────────────

    def _next_generation(self) -> List[Genome]:
        new_pop: List[Genome] = []

        # Elitism: carry champions from each species
        for sp in self.species:
            if sp.stagnation > self.STAGNATION_LIMIT:
                continue
            sorted_members = sorted(sp.members, key=lambda g: g.fitness, reverse=True)
            new_pop.extend(sorted_members[:self.ELITISM])

        # Fill remainder via crossover + mutation
        total_adj = sum(g.adjusted_fitness for g in self.population)
        if total_adj <= 0:
            # Random restart
            return [self._minimal_genome() for _ in range(self.POP_SIZE)]

        for sp in self.species:
            if sp.stagnation > self.STAGNATION_LIMIT:
                continue
            sp_share = sum(g.adjusted_fitness for g in sp.members) / total_adj
            n_offspring = max(0, int(sp_share * self.POP_SIZE) - self.ELITISM)
            for _ in range(n_offspring):
                if len(sp.members) > 1 and self._rng.random() > self.INTERSPECIES_PROB:
                    p1, p2 = self._rng.choice(sp.members, size=2, replace=False)
                else:
                    p1 = self._rng.choice(sp.members)
                    p2 = self._rng.choice(self.population)
                child = self._crossover(p1, p2)
                self._mutate(child)
                new_pop.append(child)

        # Pad to POP_SIZE if needed
        while len(new_pop) < self.POP_SIZE:
            new_pop.append(self._mutate(self._minimal_genome()))

        return new_pop[:self.POP_SIZE]

    # ── Main generation step ──────────────────────────────────────────────────

    def evolve(self, response_text: str = "") -> dict:
        """
        Run one generation:
          1. Evaluate fitness
          2. Speciate
          3. Reproduce
          4. Apply best genome to live CNN + NPU
        """
        with self._lock:
            self.evaluate_population(response_text)
            best = max(self.population, key=lambda g: g.fitness)
            self.best_genome = best
            avg_fit = float(np.mean([g.fitness for g in self.population]))

            self.best_fitness_history.append(best.fitness)
            self.avg_fitness_history.append(avg_fit)
            self.species_count_history.append(len(self.species))

            self.population = self._next_generation()
            self._speciate()
            self.generation += 1

        # Push best genome topology into live CNN
        self._apply_best_to_cnn(best)
        # Push best genome weight perturbation into NPU
        self._apply_best_to_npu(best)

        result = {
            "generation": self.generation,
            "best_fitness": best.fitness,
            "avg_fitness": avg_fit,
            "species": len(self.species),
            "population": len(self.population),
            "best_topology": f"{len(best.nodes)}N / {len(best.active_conns())}C",
        }
        self.bus.publish("evo.generation", result)
        return result

    # ── Apply best genome to live systems ────────────────────────────────────

    def _apply_best_to_cnn(self, genome: Genome) -> None:
        """
        Nudge the CNN towards the best genome:
        - Add nodes if genome has hidden nodes the CNN lacks
        - Prune dead neurons that have no corresponding genome node
        """
        hidden_in_genome = sum(1 for n in genome.nodes if n.layer not in (0, -1))
        with self.cnn._lock:
            cnn_hidden_layers = list(range(1, len(self.cnn._layers) - 1))
            current_hidden = sum(len(self.cnn._layers[li]) for li in cnn_hidden_layers)

        if hidden_in_genome > current_hidden and cnn_hidden_layers:
            # Add one neuron to the smallest hidden layer
            target = min(cnn_hidden_layers, key=lambda li: len(self.cnn._layers[li]))
            self.cnn.add_neuron(target)
        elif hidden_in_genome < current_hidden - 1:
            # Prune a dead neuron
            with self.cnn._lock:
                dead = [n.nid for n in self.cnn._neurons.values()
                        if n.is_dead and n.layer_idx not in (0, len(self.cnn._layers) - 1)]
            if dead:
                self.cnn.prune_neuron(dead[0])

    def _apply_best_to_npu(self, genome: Genome) -> None:
        """Apply weight perturbation from best genome to NPU's first layer."""
        weights = genome.weight_array()
        if not weights.size:
            return
        layers = self.npu.get_layers()
        if not layers:
            return
        first = layers[0]
        # Reshape genome weights to first layer weight shape
        total = first.weights.size
        if weights.size >= total:
            delta = weights[:total].reshape(first.weights.shape) * 0.01
        else:
            delta = np.resize(weights, total).reshape(first.weights.shape) * 0.01
        self.npu.set_weights(first.name, first.weights + delta)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        pop_fitness = [g.fitness for g in self.population]
        return {
            "generation": self.generation,
            "population": len(self.population),
            "species": len(self.species),
            "best_fitness": float(max(pop_fitness)) if pop_fitness else 0.0,
            "avg_fitness": float(np.mean(pop_fitness)) if pop_fitness else 0.0,
            "mutations": len(self.mutation_log),
            "innovation_count": self._innov_counter,
            "best_topology": (
                f"{len(self.best_genome.nodes)}N/{len(self.best_genome.active_conns())}C"
                if self.best_genome else "–"
            ),
            "fitness_trend": list(self.best_fitness_history)[-10:],
        }
