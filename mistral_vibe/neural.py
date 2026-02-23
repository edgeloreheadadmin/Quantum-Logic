"""
neural.py — Cybernetic Neural Network + Deep Learning Shaper

CyberneticNN  : dynamic-topology network with self-monitoring
DeepLearningShaper : learns to reshape the network (weights *and* structure)
  - tracks gradient magnitudes per layer
  - proposes add-node / prune-node / rewire based on gradient signal
  - folds its decisions back into the CyberneticNN live
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import SharedVirtualStorage, SystemBus, VirtualGPU


# ═══════════════════════════════════════════════════════════════════════════════
# Cybernetic Neural Network
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Neuron:
    nid: int
    layer_idx: int
    bias: float = 0.0
    activation_sum: float = 0.0   # running EMA of |activations|
    gradient_mag: float = 0.0     # running EMA of |gradients|
    alive: bool = True

    @property
    def is_dead(self) -> bool:
        """Dying ReLU heuristic: very low activation over time."""
        return self.activation_sum < 1e-6 and self.alive


@dataclass
class Synapse:
    src: int          # neuron id
    dst: int          # neuron id
    weight: float
    grad_ema: float = 0.0   # exponential moving avg of |grad|
    enabled: bool = True


class CyberneticNN:
    """
    Dynamic-topology neural network.

    Topology is stored as a list of layers (lists of Neuron ids) and a
    flat dict of Synapses keyed by (src, dst).  The network can grow /
    shrink at runtime and integrates with SharedVirtualStorage for
    weight persistence across the shared memory bus.
    """

    EMA_ALPHA = 0.05    # smoothing factor for running statistics

    def __init__(
        self,
        bus: SystemBus,
        gpu: VirtualGPU,
        storage: SharedVirtualStorage,
        layer_sizes: List[int] = None,
    ) -> None:
        self.bus = bus
        self.gpu = gpu
        self.storage = storage
        self._lock = threading.RLock()

        self._neurons: Dict[int, Neuron] = {}
        self._synapses: Dict[Tuple[int, int], Synapse] = {}
        self._layers: List[List[int]] = []   # list of [nid, ...] per layer
        self._nid_counter = 0

        # Performance tracking
        self.loss_history: deque[float] = deque(maxlen=200)
        self.accuracy_history: deque[float] = deque(maxlen=200)
        self.reshape_events: deque[dict] = deque(maxlen=50)
        self.generation = 0

        # Build initial topology
        sizes = layer_sizes or [512, 256, 128, 64, 32]
        self._build_fully_connected(sizes)

        # Persist initial weights to shared storage
        self._persist_weights()
        bus.publish("cnn.init", {"layers": len(self._layers), "synapses": len(self._synapses)})

    # ── Construction ──────────────────────────────────────────────────────────

    def _new_neuron(self, layer_idx: int) -> int:
        nid = self._nid_counter
        self._nid_counter += 1
        self._neurons[nid] = Neuron(nid=nid, layer_idx=layer_idx)
        return nid

    def _build_fully_connected(self, sizes: List[int]) -> None:
        self._layers = []
        rng = np.random.default_rng(42)
        for li, size in enumerate(sizes):
            layer_nids = [self._new_neuron(li) for _ in range(size)]
            self._layers.append(layer_nids)

        # Connect consecutive layers
        for li in range(len(self._layers) - 1):
            for src in self._layers[li]:
                for dst in self._layers[li + 1]:
                    w = float(rng.standard_normal() * np.sqrt(2.0 / len(self._layers[li])))
                    self._synapses[(src, dst)] = Synapse(src=src, dst=dst, weight=w)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Layer-by-layer forward pass.
        x shape: (batch, in_features) or (in_features,)
        Returns (output, metrics).
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        with self._lock:
            layers = [list(l) for l in self._layers]
            synapses = dict(self._synapses)
            neurons = dict(self._neurons)

        # Map each neuron in layer 0 → input feature (wrap-around if needed)
        in_size = x.shape[-1]
        activations: Dict[int, np.ndarray] = {}
        for i, nid in enumerate(layers[0]):
            feat_idx = i % in_size
            val = x[:, feat_idx] + neurons[nid].bias
            activations[nid] = np.maximum(0.0, val)   # ReLU

        for li in range(1, len(layers)):
            for dst_nid in layers[li]:
                agg = np.zeros(x.shape[0], dtype=np.float32)
                for src_nid in layers[li - 1]:
                    syn = synapses.get((src_nid, dst_nid))
                    if syn and syn.enabled:
                        src_act = activations.get(src_nid, np.zeros(x.shape[0]))
                        agg += src_act * syn.weight
                agg += neurons[dst_nid].bias
                # Activation: last layer linear, others ReLU
                if li == len(layers) - 1:
                    activations[dst_nid] = agg
                else:
                    activations[dst_nid] = np.maximum(0.0, agg)

                # Update EMA stats
                with self._lock:
                    n = self._neurons.get(dst_nid)
                    if n:
                        mag = float(np.mean(np.abs(agg)))
                        n.activation_sum = (1 - self.EMA_ALPHA) * n.activation_sum + self.EMA_ALPHA * mag

        # Collect output layer
        out_nids = layers[-1]
        out = np.stack([activations.get(nid, np.zeros(x.shape[0])) for nid in out_nids], axis=-1)

        metrics = {
            "active_neurons": sum(1 for n in self._neurons.values() if n.alive),
            "synapses": sum(1 for s in self._synapses.values() if s.enabled),
            "layers": len(self._layers),
        }
        return out, metrics

    # ── Backward (gradient tracking) ─────────────────────────────────────────

    def backward_signal(self, loss: float) -> None:
        """
        Simplified gradient signal: distribute loss backwards as EMA.
        Real gradient is approximated by perturbing weights slightly.
        """
        self.loss_history.append(loss)
        with self._lock:
            for syn in self._synapses.values():
                if not syn.enabled:
                    continue
                # Pseudo-gradient: weight magnitude scaled by loss
                pseudo_grad = abs(syn.weight) * loss * 0.01
                syn.grad_ema = (1 - self.EMA_ALPHA) * syn.grad_ema + self.EMA_ALPHA * pseudo_grad

        self.bus.publish("cnn.backward", {"loss": loss})

    # ── Dynamic topology ──────────────────────────────────────────────────────

    def add_neuron(self, layer_idx: int) -> int:
        """Grow the network by adding one neuron to a hidden layer."""
        with self._lock:
            if layer_idx <= 0 or layer_idx >= len(self._layers) - 1:
                return -1
            nid = self._new_neuron(layer_idx)
            self._layers[layer_idx].append(nid)
            rng = np.random.default_rng()
            # Connect to previous and next layer
            for src in self._layers[layer_idx - 1]:
                w = float(rng.standard_normal() * 0.1)
                self._synapses[(src, nid)] = Synapse(src=src, dst=nid, weight=w)
            for dst in self._layers[layer_idx + 1]:
                w = float(rng.standard_normal() * 0.1)
                self._synapses[(nid, dst)] = Synapse(src=nid, dst=dst, weight=w)

        self.reshape_events.append({
            "op": "add_neuron", "layer": layer_idx, "nid": nid, "ts": time.time()
        })
        self.bus.publish("cnn.grow", {"layer": layer_idx, "nid": nid})
        self._persist_weights()
        return nid

    def prune_neuron(self, nid: int) -> bool:
        """Remove a neuron and all its synapses."""
        with self._lock:
            n = self._neurons.get(nid)
            if not n or n.layer_idx in (0, len(self._layers) - 1):
                return False
            n.alive = False
            self._layers[n.layer_idx] = [x for x in self._layers[n.layer_idx] if x != nid]
            to_del = [(s, d) for s, d in self._synapses if s == nid or d == nid]
            for k in to_del:
                del self._synapses[k]
            del self._neurons[nid]

        self.reshape_events.append({
            "op": "prune_neuron", "nid": nid, "ts": time.time()
        })
        self.bus.publish("cnn.prune", {"nid": nid})
        self._persist_weights()
        return True

    def rewire_synapse(self, old_src: int, old_dst: int, new_src: int, new_dst: int) -> bool:
        """Move a synapse connection."""
        with self._lock:
            key = (old_src, old_dst)
            if key not in self._synapses:
                return False
            syn = self._synapses.pop(key)
            syn.src, syn.dst = new_src, new_dst
            self._synapses[(new_src, new_dst)] = syn
        self.bus.publish("cnn.rewire", {"from": key, "to": (new_src, new_dst)})
        return True

    # ── Weight persistence ────────────────────────────────────────────────────

    def _persist_weights(self) -> None:
        """Flatten all synapse weights into shared storage."""
        with self._lock:
            w = np.array([s.weight for s in self._synapses.values()], dtype=np.float32)
            g = np.array([s.grad_ema for s in self._synapses.values()], dtype=np.float32)
        if self.storage.get("cnn.weights") is None:
            self.storage.alloc("cnn.weights", (len(w),), owner="cnn", init=w)
            self.storage.alloc("cnn.grads", (len(g),), owner="cnn", init=g)
        else:
            self.storage.put("cnn.weights", w)
            self.storage.put("cnn.grads", g)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def topology_str(self) -> str:
        return "→".join(str(len(l)) for l in self._layers)

    def stats(self) -> dict:
        with self._lock:
            active_syns = sum(1 for s in self._synapses.values() if s.enabled)
            total_syns = len(self._synapses)
            dead = sum(1 for n in self._neurons.values() if n.is_dead)
        return {
            "topology": self.topology_str(),
            "neurons": len(self._neurons),
            "synapses": total_syns,
            "active_synapses": active_syns,
            "dead_neurons": dead,
            "layers": len(self._layers),
            "reshape_events": len(self.reshape_events),
            "generation": self.generation,
            "avg_loss": float(np.mean(list(self.loss_history))) if self.loss_history else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Deep Learning Shaper
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShapeDecision:
    action: str          # "add_neuron" | "prune_neuron" | "scale_weights" | "rewire"
    target: dict         # action-specific parameters
    confidence: float    # 0–1
    reason: str


class DeepLearningShaper:
    """
    A meta-learning model that observes the CyberneticNN's gradient/activation
    statistics and decides when and how to reshape it.

    The shaper itself is a small 2-layer MLP trained via online stochastic
    gradient descent (SGD).  Its input features are:
      - per-layer mean activation magnitude
      - per-layer mean gradient EMA
      - current loss
      - network size (neuron count, synapse count)

    Its output is a soft distribution over reshape actions.
    """

    ACTIONS = ["noop", "add_neuron", "prune_dead", "scale_up", "scale_down", "rewire"]
    FEATURE_DIM = 16
    HIDDEN_DIM = 32

    def __init__(self, bus: SystemBus, storage: SharedVirtualStorage) -> None:
        self.bus = bus
        self.storage = storage
        self._lock = threading.RLock()

        # Shaper MLP: features → hidden → action_logits
        rng = np.random.default_rng(0)
        self.W1 = rng.standard_normal((self.FEATURE_DIM, self.HIDDEN_DIM)).astype(np.float32) * 0.1
        self.b1 = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
        self.W2 = rng.standard_normal((self.HIDDEN_DIM, len(self.ACTIONS))).astype(np.float32) * 0.1
        self.b2 = np.zeros(len(self.ACTIONS), dtype=np.float32)

        # Training state
        self.lr = 0.001
        self.loss_history: deque[float] = deque(maxlen=100)
        self.decision_history: deque[ShapeDecision] = deque(maxlen=50)
        self.weight_change_history: deque[float] = deque(maxlen=100)
        self._step = 0

        # Persist own weights
        storage.alloc("shaper.W1", self.W1.shape, owner="shaper", init=self.W1)
        storage.alloc("shaper.W2", self.W2.shape, owner="shaper", init=self.W2)
        storage.pin("shaper.W1")
        storage.pin("shaper.W2")

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_features(self, cnn: CyberneticNN) -> np.ndarray:
        feat = np.zeros(self.FEATURE_DIM, dtype=np.float32)
        with cnn._lock:
            # Per-layer mean activation (first 5 layers)
            for li, layer in enumerate(cnn._layers[:5]):
                acts = [cnn._neurons[n].activation_sum for n in layer if n in cnn._neurons]
                feat[li] = float(np.mean(acts)) if acts else 0.0
            # Mean gradient EMA
            grads = [s.grad_ema for s in cnn._synapses.values()]
            feat[5] = float(np.mean(grads)) if grads else 0.0
            feat[6] = float(np.std(grads)) if grads else 0.0
            # Network size normalised
            feat[7] = len(cnn._neurons) / 2000.0
            feat[8] = len(cnn._synapses) / 50000.0
            # Dead neuron fraction
            dead = sum(1 for n in cnn._neurons.values() if n.is_dead)
            feat[9] = dead / max(len(cnn._neurons), 1)
        # Recent loss
        if cnn.loss_history:
            feat[10] = float(cnn.loss_history[-1])
            feat[11] = float(np.mean(list(cnn.loss_history)[-20:]))
        # Reshape event rate
        feat[12] = len(cnn.reshape_events) / 50.0
        feat[13] = self._step / 10000.0   # training progress
        return feat

    # ── Forward pass of shaper ────────────────────────────────────────────────

    def _forward(self, feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.maximum(0.0, feat @ self.W1 + self.b1)   # ReLU hidden
        logits = h @ self.W2 + self.b2
        exp_l = np.exp(logits - logits.max())
        probs = exp_l / exp_l.sum()
        return probs, h

    # ── Update shaper weights (online SGD) ───────────────────────────────────

    def _update(self, feat: np.ndarray, action_idx: int, reward: float) -> float:
        """
        Policy-gradient-style update: increase probability of action if reward > 0.
        Also minimizes weight magnitude for L2 regularisation.
        """
        probs, h = self._forward(feat)
        # Cross-entropy loss weighted by -reward
        target = np.zeros(len(self.ACTIONS), dtype=np.float32)
        target[action_idx] = 1.0
        loss = float(-reward * np.sum(target * np.log(probs + 1e-8)))

        # Gradients (chain rule, simplified)
        dlogits = probs - target
        dlogits *= -reward
        dW2 = np.outer(h, dlogits)
        db2 = dlogits
        dh = dlogits @ self.W2.T
        dh *= (h > 0).astype(np.float32)  # ReLU gradient
        dW1 = np.outer(feat, dh)
        db1 = dh

        # SGD with L2
        reg = 1e-4
        with self._lock:
            old_W1 = self.W1.copy()
            self.W1 -= self.lr * (dW1 + reg * self.W1)
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * (dW2 + reg * self.W2)
            self.b2 -= self.lr * db2
            w_change = float(np.mean(np.abs(self.W1 - old_W1)))
            self.weight_change_history.append(w_change)

        # Persist
        self.storage.put("shaper.W1", self.W1)
        self.storage.put("shaper.W2", self.W2)
        return loss

    # ── Main step: observe CNN → decide → act ────────────────────────────────

    def step(self, cnn: CyberneticNN) -> Optional[ShapeDecision]:
        """
        Called after each LLM exchange.  Returns the ShapeDecision taken
        (or None if noop).
        """
        self._step += 1
        feat = self._extract_features(cnn)
        probs, _ = self._forward(feat)
        action_idx = int(np.argmax(probs))
        action = self.ACTIONS[action_idx]
        confidence = float(probs[action_idx])

        decision = self._execute_action(action, confidence, cnn)

        # Reward: improvement in loss (negative Δloss = good)
        if len(cnn.loss_history) >= 2:
            reward = float(cnn.loss_history[-2]) - float(cnn.loss_history[-1])
        else:
            reward = 0.0

        shaper_loss = self._update(feat, action_idx, reward)
        self.loss_history.append(shaper_loss)

        if decision:
            self.decision_history.append(decision)
            self.bus.publish("shaper.decision", {
                "action": decision.action,
                "confidence": decision.confidence,
                "reason": decision.reason,
            })

        cnn.generation += 1
        return decision

    def _execute_action(self, action: str, conf: float, cnn: CyberneticNN) -> Optional[ShapeDecision]:
        if action == "noop":
            return None

        if action == "add_neuron":
            # Pick the hidden layer with fewest neurons
            with cnn._lock:
                hi_layers = list(range(1, len(cnn._layers) - 1))
            if not hi_layers:
                return None
            target_layer = min(hi_layers, key=lambda li: len(cnn._layers[li]))
            nid = cnn.add_neuron(target_layer)
            if nid >= 0:
                return ShapeDecision("add_neuron", {"layer": target_layer, "nid": nid},
                                     conf, f"Layer {target_layer} under-capacity")

        elif action == "prune_dead":
            with cnn._lock:
                dead = [n.nid for n in cnn._neurons.values() if n.is_dead
                        and n.layer_idx not in (0, len(cnn._layers) - 1)]
            if dead:
                nid = dead[0]
                cnn.prune_neuron(nid)
                return ShapeDecision("prune_neuron", {"nid": nid}, conf,
                                     f"Neuron {nid} dead (activation≈0)")

        elif action in ("scale_up", "scale_down"):
            scale = 1.05 if action == "scale_up" else 0.95
            with cnn._lock:
                for syn in cnn._synapses.values():
                    syn.weight *= scale
            return ShapeDecision(action, {"scale": scale}, conf,
                                 f"{'Amplify' if scale > 1 else 'Dampen'} all weights ×{scale:.2f}")

        elif action == "rewire":
            with cnn._lock:
                # Pick weakest synapse and rewire within same layers
                if not cnn._synapses:
                    return None
                weakest = min(cnn._synapses.values(), key=lambda s: abs(s.weight))
                rng = np.random.default_rng()
                li = cnn._neurons[weakest.src].layer_idx
                next_li = li + 1
                if next_li >= len(cnn._layers):
                    return None
                new_src = rng.choice(cnn._layers[li])
                new_dst = rng.choice(cnn._layers[next_li])
            cnn.rewire_synapse(weakest.src, weakest.dst, int(new_src), int(new_dst))
            return ShapeDecision("rewire", {"from": (weakest.src, weakest.dst),
                                             "to": (int(new_src), int(new_dst))},
                                  conf, "Redirect weak synapse")
        return None

    # ── Inject LLM knowledge ─────────────────────────────────────────────────

    def inject_llm_signal(self, response_text: str, cnn: CyberneticNN) -> None:
        """
        Parse an LLM response to extract a rough quality signal and use it
        to guide the shaper's next learning step.
        Code-heavy responses → reward structure-building actions.
        Short responses → penalise, network may need more capacity.
        """
        tokens = len(response_text.split())
        code_blocks = response_text.count("```")
        # Simple heuristic quality: more content = better
        quality = min(1.0, (tokens / 200.0)) + (code_blocks * 0.1)
        pseudo_loss = max(0.0, 1.0 - quality)
        cnn.backward_signal(pseudo_loss)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "step": self._step,
            "shaper_loss": float(np.mean(list(self.loss_history))) if self.loss_history else 0.0,
            "avg_w_change": float(np.mean(list(self.weight_change_history))) if self.weight_change_history else 0.0,
            "decisions": len(self.decision_history),
            "last_decisions": [d.action for d in list(self.decision_history)[-5:]],
            "lr": self.lr,
            "W1_norm": float(np.linalg.norm(self.W1)),
            "W2_norm": float(np.linalg.norm(self.W2)),
        }
