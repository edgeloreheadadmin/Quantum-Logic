"""
app.py — Mistral Vibe
Codex-style multi-tab TUI with integrated virtual hardware.

Tabs
────
  Chat        — LLM conversation (streaming)
  NPU         — Virtual NPU pipeline monitor
  Neural Net  — Cybernetic NN + DL Shaper live view
  Evolution   — NEAT neuro-evolution dashboard
  Memory      — RAM disk + shared storage + GPU
  Cognitive   — Cognitive CPU modules + knowledge graph
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Footer,
    Input,
    Label,
    LoadingIndicator,
    Markdown,
    Static,
    TabbedContent,
    TabPane,
)

from .client import AVAILABLE_MODELS, Message, MistralClient
from .cognitive import CognitiveCPU
from .core import SharedVirtualStorage, SystemBus, VirtualGPU, VirtualRAMDisk
from .evolution import NeuroEvolution
from .neural import CyberneticNN, DeepLearningShaper
from .npu import VirtualNPU


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _bar(pct: float, width: int = 20, filled: str = "█", empty: str = "░") -> str:
    n = max(0, min(width, int(pct / 100 * width)))
    return filled * n + empty * (width - n)

def _fmt_bytes(b: int) -> str:
    if b >= 1_073_741_824:
        return f"{b / 1_073_741_824:.1f} GB"
    if b >= 1_048_576:
        return f"{b / 1_048_576:.1f} MB"
    if b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"

def _sparkline(values: list, width: int = 20) -> str:
    SPARKS = " ▁▂▃▄▅▆▇█"
    if not values:
        return " " * width
    mn, mx = min(values), max(values)
    rng = mx - mn or 1
    chars = []
    step = max(1, len(values) // width)
    sample = values[-width * step :: step][-width:]
    for v in sample:
        idx = int((v - mn) / rng * (len(SPARKS) - 1))
        chars.append(SPARKS[idx])
    return "".join(chars).ljust(width)


# ═══════════════════════════════════════════════════════════════════════════════
# Chat widgets
# ═══════════════════════════════════════════════════════════════════════════════

class WelcomeBanner(Static):
    def render(self) -> str:
        return (
            "[bold cyan]✦  Mistral Vibe[/bold cyan]\n"
            "[dim]────────────────────────────────[/dim]\n"
            "[dim]Chat with Mistral. All systems online.[/dim]"
        )


class UserTurn(Static):
    def __init__(self, content: str, **kw):
        super().__init__(**kw)
        self._content = content

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]You[/bold cyan]", classes="turn-label")
        yield Static(self._content, classes="turn-content")


class AssistantTurn(Static):
    content: reactive[str] = reactive("", recompose=True)

    def compose(self) -> ComposeResult:
        yield Label("[bold magenta]✦ Mistral[/bold magenta]", classes="turn-label")
        if self.content:
            yield Markdown(self.content)
        else:
            yield LoadingIndicator()

    def append(self, chunk: str) -> None:
        self.content = self.content + chunk


# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════

class HeaderBar(Horizontal):
    def __init__(self, model: str, cwd: str, **kw):
        super().__init__(**kw)
        self._model = model
        self._cwd = cwd

    def compose(self) -> ComposeResult:
        yield Label("✦ [bold cyan]Mistral Vibe[/bold cyan]", id="hl-logo")
        yield Label("│", classes="h-sep")
        yield Label(self._model, id="hl-model")
        yield Label("│", classes="h-sep")
        yield Label(f"~/{self._cwd}", id="hl-dir")
        yield Label("", id="hl-status")

    def set_model(self, m: str) -> None:
        self.query_one("#hl-model", Label).update(m)

    def set_status(self, s: str) -> None:
        self.query_one("#hl-status", Label).update(s)


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard panels (live-updating Static widgets)
# ═══════════════════════════════════════════════════════════════════════════════

class NPUPanel(Static):
    """Real-time NPU statistics."""

    def __init__(self, npu: VirtualNPU, **kw):
        super().__init__(**kw)
        self._npu = npu

    def render(self) -> str:
        s = self._npu.stats()
        util = s["utilization"]
        bar = _bar(util)
        temp = s["temperature"]
        spark = _sparkline(list(self._npu._util_history))
        lat_spark = _sparkline(list(self._npu._latency_history))
        layers = self._npu.get_layers()
        layer_lines = []
        for l in layers[:8]:
            fp = _bar(l.flops_last / max(1, sum(x.flops_last for x in layers)) * 100, 10)
            layer_lines.append(
                f"  [dim]{l.name:<10}[/dim] {l.in_dim:>5}→{l.out_dim:<5} [{l.activation:<8}] {fp}"
            )
        layer_str = "\n".join(layer_lines) or "  (none)"

        return (
            "[bold cyan]◈ Virtual NPU[/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"  Utilisation  {bar} [bold]{util:5.1f}%[/bold]\n"
            f"  Temperature  [{'red' if temp > 75 else 'yellow' if temp > 60 else 'green'}]{temp:.1f}°C[/]\n"
            f"  Latency ms   {s['last_latency_ms']:.3f}  avg {s['avg_latency_ms']:.3f}\n"
            f"  Total GFLOPS {s['total_gflops']:.2f}\n"
            f"  Ops          {s['total_ops']}\n"
            f"  Params       {s['params']:,}\n"
            f"\n  [dim]Utilisation history[/dim] {spark}\n"
            f"  [dim]Latency history    [/dim] {lat_spark}\n"
            f"\n[bold cyan]◈ Pipeline Layers ({s['layers']})[/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"{layer_str}"
        )


class NeuralPanel(Static):
    """CyberneticNN + DeepLearningShaper live view."""

    def __init__(self, cnn: CyberneticNN, shaper: DeepLearningShaper, **kw):
        super().__init__(**kw)
        self._cnn = cnn
        self._shaper = shaper

    def render(self) -> str:
        cs = self._cnn.stats()
        ss = self._shaper.stats()

        topology = cs["topology"]
        # Reshape events
        evts = list(self._cnn.reshape_events)[-5:]
        evt_lines = []
        for e in evts:
            age = time.time() - e["ts"]
            evt_lines.append(f"  [dim]{age:.0f}s ago[/dim] {e['op']}")
        evt_str = "\n".join(evt_lines) or "  [dim](none yet)[/dim]"

        # Shaper decisions
        decisions = ss["last_decisions"]
        dec_str = "  " + "  →  ".join(decisions) if decisions else "  [dim](none yet)[/dim]"

        # Loss sparkline
        loss_vals = list(self._cnn.loss_history)
        loss_spark = _sparkline(loss_vals) if loss_vals else " " * 20

        return (
            "[bold cyan]◈ Cybernetic Neural Network[/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"  Topology     [bold]{topology}[/bold]\n"
            f"  Neurons      {cs['neurons']}  [dim](dead: {cs['dead_neurons']})[/dim]\n"
            f"  Synapses     {cs['active_synapses']} / {cs['synapses']}\n"
            f"  Generation   {cs['generation']}\n"
            f"  Avg loss     {cs['avg_loss']:.4f}\n"
            f"  Loss trend   {loss_spark}\n"
            f"\n[bold cyan]◈ Deep Learning Shaper[/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"  Step         {ss['step']}\n"
            f"  Shaper loss  {ss['shaper_loss']:.4f}\n"
            f"  Avg Δweight  {ss['avg_w_change']:.6f}\n"
            f"  Decisions    {ss['decisions']}\n"
            f"  ||W1||       {ss['W1_norm']:.3f}  ||W2|| {ss['W2_norm']:.3f}\n"
            f"\n  Recent decisions:\n{dec_str}\n"
            f"\n  Recent reshape events:\n{evt_str}"
        )


class EvolutionPanel(Static):
    """NEAT evolution dashboard."""

    def __init__(self, evo: NeuroEvolution, **kw):
        super().__init__(**kw)
        self._evo = evo

    def render(self) -> str:
        s = self._evo.stats()
        bf_hist = s["fitness_trend"]
        spark = _sparkline(bf_hist)
        # Mutation log
        muts = list(self._evo.mutation_log)[-8:]
        mut_lines = []
        for m in muts:
            age = time.time() - m["ts"]
            mut_lines.append(f"  [dim]{age:.0f}s[/dim]  {m['op']:20} gid={m.get('gid','?')}")
        mut_str = "\n".join(mut_lines) or "  [dim](none yet)[/dim]"

        best_fit = s["best_fitness"]
        avg_fit = s["avg_fitness"]
        fit_bar = _bar(best_fit * 100)

        # Species summary
        sp_lines = []
        for sp in self._evo.species[:5]:
            members = len(sp.members)
            stag = sp.stagnation
            sp_lines.append(
                f"  sp{sp.sid:<2} [{_bar(sp.best_fitness * 100, 8)}] "
                f"fit={sp.best_fitness:.3f}  n={members}  stag={stag}"
            )
        sp_str = "\n".join(sp_lines) or "  [dim](none)[/dim]"

        return (
            "[bold cyan]◈ Neuro-Evolution Engine[/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"  Generation   [bold]{s['generation']}[/bold]\n"
            f"  Population   {s['population']}  species {s['species']}\n"
            f"  Best fitness {fit_bar} {best_fit:.4f}\n"
            f"  Avg fitness  {avg_fit:.4f}\n"
            f"  Innovations  {s['innovation_count']}\n"
            f"  Best genome  {s['best_topology']}\n"
            f"\n  Fitness trend  {spark}\n"
            f"\n[bold cyan]◈ Species[/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"{sp_str}\n"
            f"\n[bold cyan]◈ Recent Mutations[/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"{mut_str}"
        )


class MemoryPanel(Static):
    """RAM disk + shared storage + GPU memory panel."""

    def __init__(self, ramdisk: VirtualRAMDisk, storage: SharedVirtualStorage,
                 gpu: VirtualGPU, **kw):
        super().__init__(**kw)
        self._rd = ramdisk
        self._st = storage
        self._gpu = gpu

    def render(self) -> str:
        rd = self._rd.stats()
        st = self._st.stats()
        gp = self._gpu.stats()

        rd_bar = _bar(rd["usage_pct"])
        st_bar = _bar(st["usage_pct"])
        gp_bar = _bar(gp["vram_pct"])
        gc_bar = _bar(gp["cache_pct"])

        # Top RAM files
        files = self._rd.ls()[:6]
        file_lines = [
            f"  [dim]{f.path[:32]:<34}[/dim] {_fmt_bytes(f.size):>8}  "
            f"{'[green]HOT[/green]' if f.hot else '[dim]cold[/dim]'}  r={f.reads}"
            for f in files
        ]
        file_str = "\n".join(file_lines) or "  [dim](empty)[/dim]"

        # Top storage segments
        segs = self._st.list_segments()[:5]
        seg_lines = [
            f"  [dim]{s.name[:28]:<30}[/dim] {s.size_mb:.2f} MB  "
            f"{'📌' if s.pinned else '  '}  {s.owner}"
            for s in segs
        ]
        seg_str = "\n".join(seg_lines) or "  [dim](empty)[/dim]"

        hit_rate = gp["cache_hit_rate"]

        return (
            "[bold cyan]◈ Virtual RAM Disk  (512 MB)[/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"  Used  {rd_bar} {rd['usage_pct']:.1f}%  ({_fmt_bytes(rd['used'])})\n"
            f"  Files {rd['files']}   R={rd['reads']}  W={rd['writes']}  "
            f"BW={rd['bw_mbps']:.2f} MB/s\n"
            f"\n  Recent files:\n{file_str}\n"
            f"\n[bold cyan]◈ Shared Virtual Storage  (2 GB)[/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"  Used  {st_bar} {st['usage_pct']:.1f}%  ({st['used_mb']:.1f} MB)\n"
            f"  Segs  {st['count']}   pinned={st['pinned']}  "
            f"R={st['total_reads']}  W={st['total_writes']}\n"
            f"\n  Segments:\n{seg_str}\n"
            f"\n[bold cyan]◈ Virtual GPU  (8 GB VRAM)[/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"  VRAM   {gp_bar} {gp['vram_pct']:.1f}%  ({_fmt_bytes(gp['vram_used'])})\n"
            f"  Cache  {gc_bar} {gp['cache_pct']:.1f}%  "
            f"hit rate [bold]{hit_rate:.1f}%[/bold]\n"
            f"  Bufs   {gp['buf_count']}  ops={gp['compute_ops']}  "
            f"GFLOPS={gp['total_gflops']:.2f}\n"
            f"  Synapse pool  {gp['synapse_used_mb']:.1f}/{gp['synapse_total_mb']} MB  "
            f"blocks={gp['synapse_blocks']}\n"
            f"  Utilisation   [bold]{gp['utilization']:.1f}%[/bold]  "
            f"avg={gp['avg_util']:.1f}%"
        )


class CognitivePanel(Static):
    """Cognitive CPU module view."""

    def __init__(self, cpu: CognitiveCPU, **kw):
        super().__init__(**kw)
        self._cpu = cpu

    def render(self) -> str:
        s = self._cpu.stats()
        mode = s["mode"]
        cl = s["cognitive_load"]
        cl_bar = _bar(cl * 100)
        load_spark = _sparkline(list(self._cpu.load_history))

        # Modules
        mod_lines = []
        for name, info in s["modules"].items():
            cap = float(info["cap"])
            load = float(info["load"])
            mod_lines.append(
                f"  [dim]{name:<22}[/dim] "
                f"cap=[bold]{cap:.2f}[/bold]  "
                f"load={_bar(load * 100, 8)} v{info['v']}"
            )
        mod_str = "\n".join(mod_lines)

        # Attention
        att = s["top_attention"]
        att_lines = [
            f"  [cyan]{c:<20}[/cyan] {_bar(w * 100, 10)} {w:.3f}"
            for c, w in att
        ]
        att_str = "\n".join(att_lines) or "  [dim](none)[/dim]"

        # Knowledge graph
        top_c = self._cpu.knowledge.top_concepts(8)
        kg_lines = [
            f"  [dim]{n.concept:<20}[/dim] w={n.weight:.2f}  r={n.access_count}"
            for n in top_c
        ]
        kg_str = "\n".join(kg_lines) or "  [dim](empty)[/dim]"

        last_thought = s["last_thought"] or "[dim](none)[/dim]"
        ug_events = s["upgrade_events"]

        return (
            f"[bold cyan]◈ Cognitive CPU  [{mode}][/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"  Load     {cl_bar} {cl:.2f}\n"
            f"  History  {load_spark}\n"
            f"  Upgrades triggered: {ug_events}\n"
            f"\n  [dim]Last thought:[/dim]\n"
            f"  [italic dim]{last_thought}[/italic dim]\n"
            f"\n[bold cyan]◈ Modules[/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"{mod_str}\n"
            f"\n[bold cyan]◈ Attention  ({s['attention_topics']} topics)[/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"{att_str}\n"
            f"\n[bold cyan]◈ Knowledge Graph  "
            f"({s['knowledge_nodes']} nodes / {s['knowledge_edges']} edges)[/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"{kg_str}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════════

class MistralVibeApp(App):
    CSS_PATH = Path(__file__).parent / "styles.tcss"
    TITLE = "Mistral Vibe"

    BINDINGS = [
        Binding("ctrl+c", "quit",         "Exit",    priority=True, show=True),
        Binding("ctrl+l", "clear_chat",   "Clear",   show=True),
        Binding("ctrl+n", "new_session",  "New",     show=True),
        Binding("ctrl+r", "cycle_model",  "Model",   show=True),
        Binding("ctrl+h", "toggle_help",  "Help",    show=True),
        Binding("ctrl+u", "refresh_dash", "Refresh", show=True),
    ]

    model: reactive[str] = reactive("mistral-large-latest")
    is_loading: reactive[bool] = reactive(False)

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "mistral-large-latest") -> None:
        super().__init__()
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        self.model = model
        self.client = MistralClient(api_key=self._api_key, model=self.model)
        self._conversation: List[Message] = []
        self._cwd = Path.cwd().name or "~"

        # ── Virtual hardware stack ──
        self.bus    = SystemBus()
        self.gpu    = VirtualGPU(self.bus)
        self.ramdisk = VirtualRAMDisk(self.bus)
        self.storage = SharedVirtualStorage(self.bus)
        self.npu    = VirtualNPU(self.bus, self.gpu)
        self.cnn    = CyberneticNN(self.bus, self.gpu, self.storage)
        self.shaper = DeepLearningShaper(self.bus, self.storage)
        self.evo    = NeuroEvolution(self.bus, self.cnn, self.npu)
        self.cpu    = CognitiveCPU(self.bus, self.storage)

        self._total_tokens = 0

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield HeaderBar(model=self.model, cwd=self._cwd, id="header-bar")

        with TabbedContent(id="main-tabs"):

            # ── Chat ──
            with TabPane("Chat", id="tab-chat"):
                with VerticalScroll(id="chat-scroll"):
                    yield Container(id="chat-messages")
                with Horizontal(id="input-row"):
                    yield Label("[bold cyan]>[/bold cyan]", id="prompt-icon")
                    yield Input(placeholder="Ask Mistral anything…", id="message-input")

            # ── NPU ──
            with TabPane("NPU", id="tab-npu"):
                with VerticalScroll(id="npu-scroll"):
                    yield NPUPanel(self.npu, id="npu-panel")

            # ── Neural Net ──
            with TabPane("Neural Net", id="tab-nn"):
                with VerticalScroll(id="nn-scroll"):
                    yield NeuralPanel(self.cnn, self.shaper, id="nn-panel")

            # ── Evolution ──
            with TabPane("Evolution", id="tab-evo"):
                with VerticalScroll(id="evo-scroll"):
                    yield EvolutionPanel(self.evo, id="evo-panel")

            # ── Memory ──
            with TabPane("Memory", id="tab-mem"):
                with VerticalScroll(id="mem-scroll"):
                    yield MemoryPanel(self.ramdisk, self.storage, self.gpu, id="mem-panel")

            # ── Cognitive ──
            with TabPane("Cognitive", id="tab-cog"):
                with VerticalScroll(id="cog-scroll"):
                    yield CognitivePanel(self.cpu, id="cog-panel")

        yield Footer()

    def on_mount(self) -> None:
        # Show welcome in chat
        chat = self.query_one("#chat-messages", Container)
        chat.mount(WelcomeBanner())
        self.query_one("#message-input", Input).focus()

        # Start auto-refresh timer (every 2 s)
        self.set_interval(2.0, self._refresh_panels)

    # ── Auto-refresh dashboards ───────────────────────────────────────────────

    def _refresh_panels(self) -> None:
        for panel_id in ("#npu-panel", "#nn-panel", "#evo-panel", "#mem-panel", "#cog-panel"):
            try:
                self.query_one(panel_id).refresh()
            except Exception:
                pass

    def action_refresh_dash(self) -> None:
        self._refresh_panels()

    # ── Chat input ────────────────────────────────────────────────────────────

    @on(Input.Submitted, "#message-input")
    def on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self.is_loading:
            return
        self.query_one("#message-input", Input).value = ""
        self._dispatch(text)

    @work(exclusive=True, thread=False)
    async def _dispatch(self, text: str) -> None:
        self.is_loading = True
        chat = self.query_one("#chat-messages", Container)
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        header = self.query_one("#header-bar", HeaderBar)

        # ── Render user message ──
        user_turn = UserTurn(text)
        await chat.mount(user_turn)
        scroll.scroll_end(animate=False)
        self._conversation.append({"role": "user", "content": text})

        # ── Run NPU on user input (background) ──
        try:
            npu_out, npu_metrics = self.npu.process_text(text)
            # Store embedding in RAM disk
            self.ramdisk.write(
                f"/conv/embed_{int(time.time())}.bin",
                npu_out.tobytes(),
                tags=["embedding"],
            )
        except Exception:
            npu_metrics = {}

        # ── Assistant streaming turn ──
        asst_turn = AssistantTurn()
        await chat.mount(asst_turn)
        scroll.scroll_end(animate=False)
        header.set_status("[dim]● thinking…[/dim]")

        full_response = ""
        try:
            async for chunk in self.client.stream_chat(self._conversation):
                full_response += chunk
                asst_turn.append(chunk)
                scroll.scroll_end(animate=False)
            self._conversation.append({"role": "assistant", "content": full_response})
            self._total_tokens += len(full_response.split())
            header.set_status(
                f"[dim]~{self._total_tokens} tokens  gen={self.evo.generation}[/dim]"
            )
        except RuntimeError as exc:
            asst_turn.append(f"\n**Error:** {exc}")
            header.set_status("[red]● error[/red]")
        finally:
            self.is_loading = False
            self.query_one("#message-input", Input).focus()

        if not full_response:
            return

        # ── Post-exchange virtual hardware updates ──
        self._post_exchange(text, full_response, npu_metrics)

    def _post_exchange(self, user: str, assistant: str, npu_metrics: dict) -> None:
        """Update all virtual systems after each LLM round-trip."""
        # 1. Store conversation in RAM disk
        self.ramdisk.write(
            f"/conv/turn_{len(self._conversation):04d}.txt",
            f"U: {user}\nA: {assistant[:500]}",
            tags=["conversation"],
        )

        # 2. Shaper injects LLM quality signal → CNN backward pass
        try:
            self.shaper.inject_llm_signal(assistant, self.cnn)
            self.shaper.step(self.cnn)
        except Exception:
            pass

        # 3. Neuro-evolution step
        try:
            self.evo.evolve(response_text=assistant)
        except Exception:
            pass

        # 4. Cognitive CPU update
        try:
            self.cpu.process_exchange(user, assistant, npu_metrics=npu_metrics)
        except Exception:
            pass

        # 5. Refresh dashboards immediately after exchange
        self._refresh_panels()

    # ── Reactive watchers ─────────────────────────────────────────────────────

    def watch_is_loading(self, loading: bool) -> None:
        inp = self.query_one("#message-input", Input)
        if loading:
            inp.placeholder = "Mistral is thinking…"
            inp.disabled = True
        else:
            inp.placeholder = "Ask Mistral anything…"
            inp.disabled = False

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_clear_chat(self) -> None:
        self._conversation = []
        self._total_tokens = 0
        chat = self.query_one("#chat-messages", Container)
        chat.remove_children()
        chat.mount(WelcomeBanner())
        self.query_one("#header-bar", HeaderBar).set_status("")

    def action_new_session(self) -> None:
        self.action_clear_chat()

    def action_cycle_model(self) -> None:
        idx = AVAILABLE_MODELS.index(self.model) if self.model in AVAILABLE_MODELS else 0
        self.model = AVAILABLE_MODELS[(idx + 1) % len(AVAILABLE_MODELS)]
        self.client.model = self.model
        self.query_one("#header-bar", HeaderBar).set_model(self.model)
        self.notify(f"Model → {self.model}", severity="information", timeout=3)

    def action_toggle_help(self) -> None:
        self.notify(
            "Ctrl+L clear · Ctrl+N new · Ctrl+R cycle model · "
            "Ctrl+U refresh · Ctrl+C exit",
            title="Shortcuts",
            severity="information",
            timeout=5,
        )


# ─── Entry point ──────────────────────────────────────────────────────────────

def run(api_key: Optional[str] = None, model: str = "mistral-large-latest") -> None:
    MistralVibeApp(api_key=api_key, model=model).run()


if __name__ == "__main__":
    run()
