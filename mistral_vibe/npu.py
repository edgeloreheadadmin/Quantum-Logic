"""
npu.py — Virtual Neural Processing Unit
Configurable pipeline of compute layers backed by the VirtualGPU.
Processes text embeddings and tracks TFLOPS, latency, utilisation.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import SystemBus, VirtualGPU


# ─── Layer definitions ────────────────────────────────────────────────────────

ACTIVATIONS = {
    "relu":    lambda x: np.maximum(0.0, x),
    "gelu":    lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))),
    "tanh":    np.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0))),
    "linear":  lambda x: x,
}


@dataclass
class NPULayer:
    name: str
    in_dim: int
    out_dim: int
    activation: str = "relu"
    weights: np.ndarray = field(default_factory=lambda: np.empty(0))
    bias: np.ndarray = field(default_factory=lambda: np.empty(0))
    flops_last: float = 0.0
    layer_id: int = 0

    def __post_init__(self) -> None:
        if self.weights.size == 0:
            scale = np.sqrt(2.0 / self.in_dim)
            self.weights = np.random.randn(self.in_dim, self.out_dim).astype(np.float32) * scale
        if self.bias.size == 0:
            self.bias = np.zeros(self.out_dim, dtype=np.float32)

    @property
    def param_count(self) -> int:
        return self.weights.size + self.bias.size

    @property
    def size_bytes(self) -> int:
        return self.weights.nbytes + self.bias.nbytes


# ─── Virtual NPU ──────────────────────────────────────────────────────────────

class VirtualNPU:
    """
    Simulated NPU pipeline.

    Architecture (default):
      embed(512→512) → attn_q(512→64) → attn_k(512→64) →
      attn_v(512→64) → proj(64→512) → ffn1(512→2048) → ffn2(2048→512)

    Integrates with VirtualGPU for matmul and VRAM accounting.
    Can dynamically add / remove / mutate layers.
    """

    TFLOPS_NOMINAL = 100.0   # Simulated NPU peak
    VRAM_CAP = 4 * 1024 * 1024 * 1024  # 4 GB dedicated

    _DEFAULT_ARCH: List[Tuple[str, int, int, str]] = [
        ("embed",  512,  512,  "linear"),
        ("attn_q", 512,   64,  "linear"),
        ("attn_k", 512,   64,  "linear"),
        ("attn_v", 512,   64,  "linear"),
        ("proj",    64,  512,  "relu"),
        ("ffn1",   512, 2048,  "gelu"),
        ("ffn2",  2048,  512,  "linear"),
    ]

    def __init__(self, bus: SystemBus, gpu: VirtualGPU) -> None:
        self.bus = bus
        self.gpu = gpu
        self._layers: List[NPULayer] = []
        self._lock = threading.RLock()
        self._layer_counter = 0

        # Metrics
        self.total_flops: float = 0.0
        self.total_ops: int = 0
        self.last_latency_ms: float = 0.0
        self.utilization: float = 0.0
        self._util_history: deque[float] = deque(maxlen=60)
        self._latency_history: deque[float] = deque(maxlen=60)
        self._flop_history: deque[float] = deque(maxlen=60)

        # Temperature simulation
        self._temperature: float = 42.0

        # Build default architecture
        for name, i, o, act in self._DEFAULT_ARCH:
            self._add_layer_internal(name, i, o, act)

        bus.publish("npu.init", {"layers": len(self._layers)})

    # ── Layer management ──────────────────────────────────────────────────────

    def _add_layer_internal(self, name: str, in_d: int, out_d: int, act: str) -> NPULayer:
        layer = NPULayer(name=name, in_dim=in_d, out_dim=out_d, activation=act,
                         layer_id=self._layer_counter)
        self._layer_counter += 1
        # Register weights in GPU VRAM
        self.gpu.alloc(f"npu.{name}.w", (in_d, out_d), owner="npu")
        self.gpu.alloc(f"npu.{name}.b", (out_d,), owner="npu")
        # Also try synapse pre-alloc for weights
        self.gpu.alloc_synapse(f"npu.syn.{name}", layer.weights.size)
        self._layers.append(layer)
        return layer

    def add_layer(self, name: str, in_dim: int, out_dim: int, activation: str = "relu",
                  position: Optional[int] = None) -> NPULayer:
        with self._lock:
            layer = NPULayer(name=name, in_dim=in_dim, out_dim=out_dim,
                             activation=activation, layer_id=self._layer_counter)
            self._layer_counter += 1
            self.gpu.alloc(f"npu.{name}.w", (in_dim, out_dim), owner="npu")
            self.gpu.alloc_synapse(f"npu.syn.{name}", layer.weights.size)
            if position is None:
                self._layers.append(layer)
            else:
                self._layers.insert(position, layer)
        self.bus.publish("npu.layer_add", {"name": name, "in": in_dim, "out": out_dim})
        return layer

    def remove_layer(self, name: str) -> bool:
        with self._lock:
            before = len(self._layers)
            self._layers = [l for l in self._layers if l.name != name]
            if len(self._layers) < before:
                self.gpu.free(f"npu.{name}.w")
                self.gpu.free(f"npu.{name}.b")
                self.bus.publish("npu.layer_remove", {"name": name})
                return True
        return False

    def mutate_layer(self, name: str, noise_scale: float = 0.01) -> bool:
        """Perturb a layer's weights (neuro-evolution step)."""
        with self._lock:
            for layer in self._layers:
                if layer.name == name:
                    layer.weights += np.random.randn(*layer.weights.shape).astype(np.float32) * noise_scale
                    self.bus.publish("npu.mutate", {"name": name, "scale": noise_scale})
                    return True
        return False

    def set_weights(self, name: str, weights: np.ndarray, bias: Optional[np.ndarray] = None) -> bool:
        with self._lock:
            for layer in self._layers:
                if layer.name == name:
                    if weights.shape == layer.weights.shape:
                        layer.weights = weights.astype(np.float32)
                    if bias is not None and bias.shape == layer.bias.shape:
                        layer.bias = bias.astype(np.float32)
                    return True
        return False

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Run x through all NPU layers; return (output, metrics)."""
        t0 = time.perf_counter()
        total_flops = 0.0

        with self._lock:
            layers_snapshot = list(self._layers)

        for layer in layers_snapshot:
            # Reshape if dimensions mismatch
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.shape[-1] != layer.in_dim:
                pad = layer.in_dim - x.shape[-1]
                if pad > 0:
                    x = np.pad(x, ((0, 0), (0, pad)))
                else:
                    x = x[..., :layer.in_dim]

            cache_key = f"npu.{layer.name}.{hashlib.md5(x.tobytes()).hexdigest()[:8]}"
            out = self.gpu.matmul(x, layer.weights, cache_key=cache_key)
            out = out + layer.bias

            act_fn = ACTIVATIONS.get(layer.activation, ACTIVATIONS["relu"])
            out = act_fn(out)

            flops = 2.0 * x.shape[0] * x.shape[-1] * layer.out_dim
            layer.flops_last = flops
            total_flops += flops
            x = out

        dt = time.perf_counter() - t0
        self.last_latency_ms = dt * 1000
        self.total_flops += total_flops
        self.total_ops += 1

        theoretical = total_flops / (self.TFLOPS_NOMINAL * 1e12)
        util = min(100.0, theoretical / max(dt, 1e-12) * 100)
        self.utilization = util
        self._util_history.append(util)
        self._latency_history.append(self.last_latency_ms)
        self._flop_history.append(total_flops / 1e9)

        self._temperature = 42.0 + util * 0.35

        metrics = {
            "flops": total_flops,
            "latency_ms": self.last_latency_ms,
            "utilization": util,
            "temperature": self._temperature,
        }
        self.bus.publish("npu.forward", metrics)
        return x, metrics

    def process_text(self, text: str) -> Tuple[np.ndarray, dict]:
        """Hash-embed text tokens and run through NPU."""
        words = text.lower().split()[:64]
        seq_len = max(1, len(words))
        embed = np.zeros((seq_len, 512), dtype=np.float32)
        for i, w in enumerate(words):
            seed = int(hashlib.md5(w.encode()).hexdigest()[:8], 16) % (2**31)
            rng = np.random.default_rng(seed)
            embed[i] = rng.standard_normal(512).astype(np.float32) * 0.1
        return self.forward(embed)

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def layer_count(self) -> int:
        return len(self._layers)

    @property
    def param_count(self) -> int:
        return sum(l.param_count for l in self._layers)

    @property
    def vram_bytes(self) -> int:
        return sum(l.size_bytes for l in self._layers)

    def get_layers(self) -> List[NPULayer]:
        with self._lock:
            return list(self._layers)

    def stats(self) -> dict:
        return {
            "layers": self.layer_count,
            "params": self.param_count,
            "vram_mb": self.vram_bytes / 1_048_576,
            "total_gflops": self.total_flops / 1e9,
            "total_ops": self.total_ops,
            "utilization": self.utilization,
            "avg_util": float(np.mean(self._util_history)) if self._util_history else 0.0,
            "last_latency_ms": self.last_latency_ms,
            "avg_latency_ms": float(np.mean(self._latency_history)) if self._latency_history else 0.0,
            "temperature": self._temperature,
            "layer_names": [l.name for l in self._layers],
        }
