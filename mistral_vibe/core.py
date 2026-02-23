"""
core.py — Virtual hardware infrastructure:
  SystemBus · VirtualRAMDisk · SharedVirtualStorage · VirtualGPU
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ─── System Event Bus ─────────────────────────────────────────────────────────

class SystemBus:
    """Central publish/subscribe event bus wiring all virtual components."""

    def __init__(self) -> None:
        self._subs: Dict[str, List[Callable[[dict], None]]] = {}
        self._lock = threading.Lock()
        self.history: deque[dict] = deque(maxlen=500)

    def subscribe(self, event: str, handler: Callable[[dict], None]) -> None:
        with self._lock:
            self._subs.setdefault(event, []).append(handler)

    def publish(self, event: str, data: Any = None) -> None:
        ev = {"type": event, "data": data, "ts": time.time()}
        self.history.append(ev)
        with self._lock:
            handlers = list(self._subs.get(event, []))
        for h in handlers:
            try:
                h(ev)
            except Exception:
                pass

    def recent(self, n: int = 30) -> List[dict]:
        return list(self.history)[-n:]


# ─── Virtual RAM Disk ─────────────────────────────────────────────────────────

@dataclass
class RamFile:
    path: str
    data: bytes
    tags: List[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    modified: float = field(default_factory=time.time)
    reads: int = 0
    writes: int = 1

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def hot(self) -> bool:
        age = time.time() - self.modified
        return self.reads > 4 and age < 300


class VirtualRAMDisk:
    """512 MB in-process filesystem (LRU eviction, bandwidth sampling)."""

    CAPACITY = 512 * 1024 * 1024

    def __init__(self, bus: SystemBus) -> None:
        self.bus = bus
        self._fs: OrderedDict[str, RamFile] = OrderedDict()
        self._lock = threading.RLock()
        self.total_reads = 0
        self.total_writes = 0
        self._bw_ops: deque[Tuple[float, int]] = deque(maxlen=120)

    # ── properties ──

    @property
    def used(self) -> int:
        return sum(f.size for f in self._fs.values())

    @property
    def free(self) -> int:
        return self.CAPACITY - self.used

    @property
    def usage_pct(self) -> float:
        return self.used / self.CAPACITY * 100

    # ── operations ──

    def write(self, path: str, data: bytes | str, tags: List[str] | None = None) -> bool:
        if isinstance(data, str):
            data = data.encode()
        with self._lock:
            old = self._fs[path].size if path in self._fs else 0
            if self.used - old + len(data) > self.CAPACITY:
                self._evict(len(data) - old)
            if path in self._fs:
                f = self._fs[path]
                f.data, f.modified, f.writes = data, time.time(), f.writes + 1
                self._fs.move_to_end(path)
            else:
                self._fs[path] = RamFile(path=path, data=data, tags=tags or [])
            self.total_writes += 1
            self._bw_ops.append((time.time(), len(data)))
        self.bus.publish("ramdisk.write", {"path": path, "bytes": len(data)})
        return True

    def read(self, path: str) -> Optional[bytes]:
        with self._lock:
            f = self._fs.get(path)
            if not f:
                return None
            f.reads += 1
            self.total_reads += 1
            self._bw_ops.append((time.time(), f.size))
            return f.data

    def delete(self, path: str) -> bool:
        with self._lock:
            if path in self._fs:
                del self._fs[path]
                self.bus.publish("ramdisk.delete", {"path": path})
                return True
        return False

    def ls(self, prefix: str = "") -> List[RamFile]:
        with self._lock:
            return sorted(
                [f for f in self._fs.values() if f.path.startswith(prefix)],
                key=lambda f: f.modified,
                reverse=True,
            )

    def _evict(self, needed: int) -> None:
        freed = 0
        victims = [p for p, f in self._fs.items() if not f.hot]
        for p in victims:
            if freed >= needed:
                break
            freed += self._fs[p].size
            del self._fs[p]

    def bandwidth_mbps(self) -> float:
        now = time.time()
        window = [b for t, b in self._bw_ops if now - t < 1.0]
        return sum(window) / 1_048_576

    def stats(self) -> dict:
        return {
            "used": self.used,
            "free": self.free,
            "usage_pct": self.usage_pct,
            "files": len(self._fs),
            "reads": self.total_reads,
            "writes": self.total_writes,
            "bw_mbps": self.bandwidth_mbps(),
        }


# ─── Shared Virtual Storage ───────────────────────────────────────────────────

@dataclass
class Segment:
    name: str
    owner: str
    data: np.ndarray
    pinned: bool = False
    created: float = field(default_factory=time.time)
    reads: int = 0
    writes: int = 0

    @property
    def size_mb(self) -> float:
        return self.data.nbytes / 1_048_576


class SharedVirtualStorage:
    """2 GB shared numpy segment store (cross-component shared memory)."""

    CAPACITY_MB = 2048

    def __init__(self, bus: SystemBus) -> None:
        self.bus = bus
        self._segs: Dict[str, Segment] = {}
        self._lock = threading.RLock()
        self.access_log: deque[Tuple[str, str, float]] = deque(maxlen=200)

    @property
    def used_mb(self) -> float:
        return sum(s.data.nbytes for s in self._segs.values()) / 1_048_576

    @property
    def usage_pct(self) -> float:
        return self.used_mb / self.CAPACITY_MB * 100

    def alloc(
        self,
        name: str,
        shape: tuple,
        dtype=np.float32,
        owner: str = "system",
        init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        arr = init.copy() if init is not None else np.zeros(shape, dtype=dtype)
        with self._lock:
            self._segs[name] = Segment(name=name, owner=owner, data=arr)
        self.bus.publish("storage.alloc", {"name": name, "shape": shape, "owner": owner})
        return arr

    def get(self, name: str) -> Optional[np.ndarray]:
        with self._lock:
            seg = self._segs.get(name)
            if not seg:
                return None
            seg.reads += 1
            self.access_log.append(("read", name, time.time()))
            return seg.data

    def put(self, name: str, data: np.ndarray) -> bool:
        with self._lock:
            seg = self._segs.get(name)
            if not seg:
                return False
            seg.data = data
            seg.writes += 1
            self.access_log.append(("write", name, time.time()))
        return True

    def free(self, name: str) -> bool:
        with self._lock:
            seg = self._segs.get(name)
            if seg and not seg.pinned:
                del self._segs[name]
                return True
        return False

    def pin(self, name: str) -> None:
        with self._lock:
            if name in self._segs:
                self._segs[name].pinned = True

    def list_segments(self) -> List[Segment]:
        with self._lock:
            return sorted(self._segs.values(), key=lambda s: s.created, reverse=True)

    def stats(self) -> dict:
        segs = self.list_segments()
        return {
            "count": len(segs),
            "used_mb": self.used_mb,
            "usage_pct": self.usage_pct,
            "pinned": sum(1 for s in segs if s.pinned),
            "total_reads": sum(s.reads for s in segs),
            "total_writes": sum(s.writes for s in segs),
        }


# ─── Virtual GPU ──────────────────────────────────────────────────────────────

@dataclass
class GPUBuffer:
    name: str
    owner: str
    data: np.ndarray
    pinned: bool = False
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    cache_hits: int = 0

    @property
    def size(self) -> int:
        return self.data.nbytes


class VirtualGPU:
    """Simulated 8 GB GPU: VRAM allocator, L2 cache, synapse prealloc, matmul."""

    VRAM_TOTAL = 8 * 1024 * 1024 * 1024   # 8 GB
    CACHE_CAPACITY = 256 * 1024 * 1024     # 256 MB L2
    SYNAPSE_POOL_MB = 512                   # 512 MB pre-allocated synapse pool
    TFLOPS = 19.5
    BANDWIDTH_GBPS = 900.0

    def __init__(self, bus: SystemBus) -> None:
        self.bus = bus
        self._bufs: Dict[str, GPUBuffer] = {}
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self._lock = threading.RLock()

        # Metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.compute_ops = 0
        self.total_flops: float = 0.0
        self.utilization = 0.0
        self._util_history: deque[float] = deque(maxlen=60)

        # Pre-allocate synapse pool
        pool_elements = (self.SYNAPSE_POOL_MB * 1024 * 1024) // 4
        self._synapse_pool = np.zeros(pool_elements, dtype=np.float32)
        self._synapse_ptr = 0
        self._synapse_blocks: Dict[str, np.ndarray] = {}
        self.bus.publish("gpu.prealloc", {"pool_mb": self.SYNAPSE_POOL_MB})

    # ── VRAM ──

    @property
    def vram_used(self) -> int:
        buf_mem = sum(b.size for b in self._bufs.values())
        return buf_mem + self._synapse_pool.nbytes

    @property
    def vram_pct(self) -> float:
        return self.vram_used / self.VRAM_TOTAL * 100

    @property
    def cache_pct(self) -> float:
        return self._cache_bytes / self.CACHE_CAPACITY * 100

    def alloc(self, name: str, shape: tuple, dtype=np.float32, owner: str = "system") -> np.ndarray:
        arr = np.zeros(shape, dtype=dtype)
        with self._lock:
            if self.vram_used + arr.nbytes > self.VRAM_TOTAL:
                self._evict(arr.nbytes)
            self._bufs[name] = GPUBuffer(name=name, owner=owner, data=arr)
        self.bus.publish("gpu.alloc", {"name": name, "mb": arr.nbytes / 1_048_576})
        return arr

    def free(self, name: str) -> bool:
        with self._lock:
            b = self._bufs.get(name)
            if b and not b.pinned:
                del self._bufs[name]
                return True
        return False

    def pin(self, name: str) -> None:
        with self._lock:
            if name in self._bufs:
                self._bufs[name].pinned = True

    # ── Synapse pre-allocation ──

    def alloc_synapse(self, name: str, n: int) -> Optional[np.ndarray]:
        """Carve a slice of n floats from the pre-allocated synapse pool."""
        if self._synapse_ptr + n > len(self._synapse_pool):
            return None
        view = self._synapse_pool[self._synapse_ptr : self._synapse_ptr + n]
        self._synapse_blocks[name] = view
        self._synapse_ptr += n
        self.bus.publish("gpu.synapse_alloc", {"name": name, "elements": n})
        return view

    def get_synapse(self, name: str) -> Optional[np.ndarray]:
        return self._synapse_blocks.get(name)

    # ── Cache ──

    def cache_get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.cache_hits += 1
                return self._cache[key]
            self.cache_misses += 1
        return None

    def cache_put(self, key: str, data: np.ndarray) -> None:
        with self._lock:
            if data.nbytes > self.CACHE_CAPACITY:
                return
            while self._cache_bytes + data.nbytes > self.CACHE_CAPACITY and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._cache_bytes -= evicted.nbytes
            self._cache[key] = data.copy()
            self._cache_bytes += data.nbytes

    # ── Compute ──

    def matmul(self, a: np.ndarray, b: np.ndarray, cache_key: str = "") -> np.ndarray:
        if cache_key:
            hit = self.cache_get(cache_key)
            if hit is not None:
                return hit

        t0 = time.perf_counter()
        result = np.dot(a, b)
        dt = time.perf_counter() - t0

        rows_a = a.shape[0] if a.ndim > 1 else 1
        cols_a = a.shape[-1]
        cols_b = b.shape[-1] if b.ndim > 1 else 1
        flops = 2.0 * rows_a * cols_a * cols_b
        self.total_flops += flops
        self.compute_ops += 1

        theoretical = flops / (self.TFLOPS * 1e12)
        util = min(100.0, theoretical / max(dt, 1e-12) * 100)
        self.utilization = util
        self._util_history.append(util)

        if cache_key:
            self.cache_put(cache_key, result)
        return result

    def _evict(self, needed: int) -> None:
        victims = sorted(
            [b for b in self._bufs.values() if not b.pinned],
            key=lambda b: b.last_used,
        )
        freed = 0
        for b in victims:
            if freed >= needed:
                break
            freed += b.size
            del self._bufs[b.name]

    # ── Stats ──

    def stats(self) -> dict:
        total_cache_ops = self.cache_hits + self.cache_misses
        return {
            "vram_used": self.vram_used,
            "vram_total": self.VRAM_TOTAL,
            "vram_pct": self.vram_pct,
            "cache_pct": self.cache_pct,
            "cache_hit_rate": self.cache_hits / max(total_cache_ops, 1) * 100,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "buf_count": len(self._bufs),
            "compute_ops": self.compute_ops,
            "total_gflops": self.total_flops / 1e9,
            "utilization": self.utilization,
            "avg_util": float(np.mean(self._util_history)) if self._util_history else 0.0,
            "synapse_used_mb": self._synapse_ptr * 4 / 1_048_576,
            "synapse_total_mb": self.SYNAPSE_POOL_MB,
            "synapse_blocks": len(self._synapse_blocks),
        }
