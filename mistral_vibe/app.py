"""
app.py — Mistral Vibe
Codex-style multi-tab TUI with integrated virtual hardware,
artifact builder, and project / folder-context system.

Tabs
────
  Chat        — LLM conversation (streaming) + command interpreter
  Artifacts   — React / HTML / Plugin artifacts with live preview
  Projects    — Project workspaces, folder ingestion, system instructions
  NPU         — Virtual NPU pipeline monitor
  Neural Net  — Cybernetic NN + DL Shaper live view
  Evolution   — NEAT neuro-evolution dashboard
  Memory      — RAM disk + shared storage + GPU
  Cognitive   — Cognitive CPU modules + knowledge graph

Commands (type in chat input)
──────────────────────────────
  /load <path>                  ingest folder into active project
  /project new <name>           create project
  /project load <name>          activate a saved project
  /project list                 list all projects
  /instructions <text>          set system instructions for active project
  /unload                       deactivate active project
  /artifact react               scaffold a React artifact template hint
  /artifact html                scaffold an HTML artifact template hint
  /launch <aid>                 open artifact in browser
  /stop <aid>                   stop artifact server
  /help                         show command reference
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
from textual.containers import Container, Horizontal, VerticalScroll
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

from .artifact import ArtifactEngine, NPM_AVAILABLE
from .client import AVAILABLE_MODELS, SYSTEM_PROMPT, Message, MistralClient
from .cognitive import CognitiveCPU
from .core import SharedVirtualStorage, SystemBus, VirtualGPU, VirtualRAMDisk
from .evolution import NeuroEvolution
from .neural import CyberneticNN, DeepLearningShaper
from .npu import VirtualNPU
from .project import ProjectManager


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _bar(pct: float, width: int = 20, filled: str = "█", empty: str = "░") -> str:
    n = max(0, min(width, int(pct / 100 * width)))
    return filled * n + empty * (width - n)

def _fmt_bytes(b: int) -> str:
    if b >= 1_073_741_824: return f"{b/1_073_741_824:.1f} GB"
    if b >= 1_048_576:     return f"{b/1_048_576:.1f} MB"
    if b >= 1024:          return f"{b/1024:.1f} KB"
    return f"{b} B"

def _spark(values: list, width: int = 20) -> str:
    SPARKS = " ▁▂▃▄▅▆▇█"
    if not values:
        return " " * width
    mn, mx = min(values), max(values)
    rng = mx - mn or 1
    step = max(1, len(values) // width)
    sample = values[-width * step :: step][-width:]
    return "".join(SPARKS[int((v - mn) / rng * (len(SPARKS) - 1))] for v in sample).ljust(width)


# ═══════════════════════════════════════════════════════════════════════════════
# Chat widgets
# ═══════════════════════════════════════════════════════════════════════════════

class WelcomeBanner(Static):
    def render(self) -> str:
        npm_info = "[green]npm ✓[/green]" if NPM_AVAILABLE else "[yellow]npm ✗ → CDN fallback[/yellow]"
        return (
            "[bold cyan]✦  Mistral Vibe[/bold cyan]\n"
            "[dim]─────────────────────────────────────────[/dim]\n"
            f"  Artifact engine: {npm_info}\n"
            "[dim]Type a message or a [bold]/command[/bold]. "
            "Type [bold]/help[/bold] for the full command list.[/dim]"
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


class SystemMessage(Static):
    """In-chat notification for commands, artifacts, errors, etc."""

    def __init__(self, text: str, kind: str = "info", **kw):
        super().__init__(**kw)
        self._text = text
        self._kind = kind

    def compose(self) -> ComposeResult:
        colours = {"info": "cyan", "success": "green", "warn": "yellow", "error": "red"}
        c = colours.get(self._kind, "cyan")
        icon = {"info": "ℹ", "success": "✓", "warn": "⚠", "error": "✗"}.get(self._kind, "ℹ")
        yield Label(f"[{c}]{icon}[/{c}]", classes="sys-icon")
        yield Markdown(self._text)


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
# Artifacts panel
# ═══════════════════════════════════════════════════════════════════════════════

class ArtifactsPanel(Static):
    def __init__(self, engine: ArtifactEngine, **kw):
        super().__init__(**kw)
        self._engine = engine

    def render(self) -> str:
        arts = self._engine.list_artifacts()
        s = self._engine.stats()

        header = (
            "[bold cyan]◈ Artifacts[/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"  Total={s['total']}  "
            f"Running=[green]{s['running']}[/green]  "
            f"React={s['react']}  HTML={s['html']}  Plugin={s['plugin']}\n"
            f"  npm: {'[green]available[/green]' if NPM_AVAILABLE else '[yellow]not found → CDN fallback[/yellow]'}\n\n"
        )

        if not arts:
            return (
                header
                + "  [dim]No artifacts yet.[/dim]\n\n"
                + "  Ask Mistral to build something, e.g.:\n"
                + "  [dim]> Build a React counter component with styled buttons[/dim]\n"
                + "  [dim]> Create an HTML dashboard with dark theme[/dim]\n"
            )

        rows = []
        for art in arts[:15]:
            st_col = {
                "running": "green", "building": "yellow",
                "error": "red", "saved": "cyan", "stopped": "dim",
            }.get(art.status, "white")
            url_part = f"  {art.url}" if art.url else ""
            rows.append(
                f"  [bold]{art.aid}[/bold]  [{st_col}]{art.status:<9}[/{st_col}]  "
                f"[dim]{art.kind:<7}[/dim]  {art.age_str}{url_part}"
            )

        # Code preview of the most recent artifact
        preview_block = ""
        if arts:
            a = arts[0]
            lang = {"react": "jsx", "html": "html", "plugin": "python"}.get(a.kind, "text")
            preview_block = (
                f"\n[bold cyan]◈ Code Preview — {a.aid}[/bold cyan]\n"
                f"[dim]─────────────────────────────────────────────[/dim]\n"
                f"```{lang}\n{a.short_code}\n```\n"
            )

        commands_block = (
            "\n[bold cyan]◈ Commands[/bold cyan]\n"
            "[dim]─────────────────────────────────────────────[/dim]\n"
            "  [bold]/launch <aid>[/bold]    open in browser\n"
            "  [bold]/stop <aid>[/bold]      stop server\n"
            "  [bold]/artifact react[/bold]  get React build hint\n"
            "  [bold]/artifact html[/bold]   get HTML build hint\n"
        )

        return header + "\n".join(rows) + preview_block + commands_block


# ═══════════════════════════════════════════════════════════════════════════════
# Projects panel
# ═══════════════════════════════════════════════════════════════════════════════

class ProjectsPanel(Static):
    def __init__(self, pm: ProjectManager, **kw):
        super().__init__(**kw)
        self._pm = pm

    def render(self) -> str:
        s = self._pm.stats()

        # Active project
        if self._pm.active:
            p = self._pm.active
            inst_preview = (p.instructions.strip()[:160].replace("\n", " ") + "…") \
                            if len(p.instructions) > 160 else p.instructions.strip()
            inst_str = f"  [italic dim]{inst_preview}[/italic dim]" if inst_preview \
                       else "  [dim](no instructions set)[/dim]"

            tree_lines = p.tree.splitlines()
            tree_sample = "\n".join(f"  [dim]{l}[/dim]" for l in tree_lines[:12])
            if len(tree_lines) > 12:
                tree_sample += f"\n  [dim]… +{len(tree_lines)-12} lines[/dim]"

            file_lines = [
                f"  [dim]{f.rel_path:<42}[/dim] {f.size_kb:.1f} KB  [cyan]{f.language}[/cyan]"
                for f in p.files[:10]
            ]
            if len(p.files) > 10:
                file_lines.append(f"  [dim]… +{len(p.files)-10} more[/dim]")

            active_block = (
                f"[bold cyan]◈ Active — {p.name}[/bold cyan]\n"
                f"[dim]─────────────────────────────────────────────[/dim]\n"
                f"  Files={p.file_count}  Size={p.total_kb:.1f} KB\n"
                f"\n  [bold]Instructions:[/bold]\n{inst_str}\n"
                + (f"\n  [bold]Tree:[/bold]\n{tree_sample}\n" if tree_sample.strip() else "")
                + (f"\n  [bold]Files in context:[/bold]\n" + "\n".join(file_lines) + "\n"
                   if file_lines else "")
            )
        else:
            active_block = (
                "[bold cyan]◈ Active Project[/bold cyan]\n"
                "[dim]─────────────────────────────────────────────[/dim]\n"
                "  [dim]No project active.[/dim]\n"
                "  [dim]Use [bold]/load <folder>[/bold] to ingest a folder, or[/dim]\n"
                "  [dim][bold]/project new <name>[/bold] to create a project.[/dim]\n"
            )

        # All projects list
        all_projs = self._pm.list_all()
        proj_lines = [
            f"  [bold]{p.name}[/bold]"
            + (" [bold cyan]← active[/bold cyan]" if p == self._pm.active else "")
            + f"  [dim]{p.file_count} files · {p.total_kb:.0f} KB"
            + (" · instructions ✓" if p.instructions.strip() else "")
            + "[/dim]"
            for p in all_projs[:12]
        ]

        commands_block = (
            "\n[bold cyan]◈ Commands[/bold cyan]\n"
            "[dim]─────────────────────────────────────────────[/dim]\n"
            "  [bold]/load <folder-path>[/bold]       ingest folder into context\n"
            "  [bold]/project new <name>[/bold]       create a new project\n"
            "  [bold]/project load <name>[/bold]      activate saved project\n"
            "  [bold]/project list[/bold]             list all projects\n"
            "  [bold]/instructions <text>[/bold]      set system instructions\n"
            "  [bold]/unload[/bold]                   deactivate current project\n"
        )

        return (
            active_block
            + f"\n[bold cyan]◈ All Projects ({s['count']})[/bold cyan]\n"
            + "[dim]─────────────────────────────────────────────[/dim]\n"
            + ("\n".join(proj_lines) if proj_lines else "  [dim](none)[/dim]")
            + commands_block
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Hardware dashboard panels
# ═══════════════════════════════════════════════════════════════════════════════

class NPUPanel(Static):
    def __init__(self, npu: VirtualNPU, **kw):
        super().__init__(**kw)
        self._npu = npu

    def render(self) -> str:
        s = self._npu.stats()
        util = s["utilization"]
        layers = self._npu.get_layers()
        total_fl = max(1.0, sum(l.flops_last for l in layers))
        layer_lines = [
            f"  [dim]{l.name:<10}[/dim] {l.in_dim:>5}→{l.out_dim:<5} "
            f"[{l.activation:<8}] {_bar(l.flops_last/total_fl*100, 10)}"
            for l in layers[:8]
        ]
        return (
            "[bold cyan]◈ Virtual NPU[/bold cyan]\n"
            f"[dim]─────────────────────────────────────────────[/dim]\n"
            f"  Utilisation  {_bar(util)} [bold]{util:5.1f}%[/bold]\n"
            f"  Temperature  [{'red' if s['temperature']>75 else 'yellow' if s['temperature']>60 else 'green'}]"
            f"{s['temperature']:.1f}°C[/]\n"
            f"  Latency ms   {s['last_latency_ms']:.3f}  avg={s['avg_latency_ms']:.3f}\n"
            f"  GFLOPS       {s['total_gflops']:.2f}  ops={s['total_ops']}\n"
            f"  Params       {s['params']:,}\n"
            f"\n  Util history   {_spark(list(self._npu._util_history))}\n"
            f"  Latency hist   {_spark(list(self._npu._latency_history))}\n"
            f"\n[bold cyan]◈ Pipeline ({s['layers']} layers)[/bold cyan]\n"
            "[dim]─────────────────────────────────────────────[/dim]\n"
            + "\n".join(layer_lines)
        )


class NeuralPanel(Static):
    def __init__(self, cnn: CyberneticNN, shaper: DeepLearningShaper, **kw):
        super().__init__(**kw)
        self._cnn = cnn
        self._shaper = shaper

    def render(self) -> str:
        cs = self._cnn.stats()
        ss = self._shaper.stats()
        evts = [
            f"  [dim]{time.time()-e['ts']:.0f}s ago[/dim] {e['op']}"
            for e in list(self._cnn.reshape_events)[-5:]
        ]
        decs = ("  " + "  →  ".join(ss["last_decisions"])) if ss["last_decisions"] \
               else "  [dim](none yet)[/dim]"
        return (
            "[bold cyan]◈ Cybernetic Neural Network[/bold cyan]\n"
            "[dim]─────────────────────────────────────────────[/dim]\n"
            f"  Topology   [bold]{cs['topology']}[/bold]\n"
            f"  Neurons    {cs['neurons']}  dead={cs['dead_neurons']}\n"
            f"  Synapses   {cs['active_synapses']} / {cs['synapses']}\n"
            f"  Generation {cs['generation']}  avg_loss={cs['avg_loss']:.4f}\n"
            f"  Loss       {_spark(list(self._cnn.loss_history))}\n"
            f"\n[bold cyan]◈ Deep Learning Shaper[/bold cyan]\n"
            "[dim]─────────────────────────────────────────────[/dim]\n"
            f"  Step={ss['step']}  loss={ss['shaper_loss']:.4f}  "
            f"ΔW={ss['avg_w_change']:.6f}  decisions={ss['decisions']}\n"
            f"  Recent: {decs}\n"
            f"\n  Reshape events:\n"
            + ("\n".join(evts) if evts else "  [dim](none)[/dim]")
        )


class EvolutionPanel(Static):
    def __init__(self, evo: NeuroEvolution, **kw):
        super().__init__(**kw)
        self._evo = evo

    def render(self) -> str:
        s = self._evo.stats()
        sp_lines = [
            f"  sp{sp.sid:<2} {_bar(sp.best_fitness*100, 8)} "
            f"fit={sp.best_fitness:.3f} n={len(sp.members)} stag={sp.stagnation}"
            for sp in self._evo.species[:5]
        ]
        mut_lines = [
            f"  [dim]{time.time()-m['ts']:.0f}s[/dim]  {m['op']:20} gid={m.get('gid','?')}"
            for m in list(self._evo.mutation_log)[-6:]
        ]
        return (
            "[bold cyan]◈ Neuro-Evolution[/bold cyan]\n"
            "[dim]─────────────────────────────────────────────[/dim]\n"
            f"  Gen={s['generation']}  pop={s['population']}  species={s['species']}\n"
            f"  Best   {_bar(s['best_fitness']*100)} {s['best_fitness']:.4f}\n"
            f"  Avg    {s['avg_fitness']:.4f}  innovations={s['innovation_count']}\n"
            f"  Genome {s['best_topology']}\n"
            f"  Trend  {_spark(s['fitness_trend'])}\n"
            f"\n  Species:\n"
            + ("\n".join(sp_lines) if sp_lines else "  [dim](none)[/dim]")
            + "\n\n  Mutations:\n"
            + ("\n".join(mut_lines) if mut_lines else "  [dim](none)[/dim]")
        )


class MemoryPanel(Static):
    def __init__(self, rd: VirtualRAMDisk, st: SharedVirtualStorage,
                 gpu: VirtualGPU, **kw):
        super().__init__(**kw)
        self._rd, self._st, self._gpu = rd, st, gpu

    def render(self) -> str:
        rd = self._rd.stats()
        st = self._st.stats()
        gp = self._gpu.stats()
        files = [
            f"  [dim]{f.path[:36]:<38}[/dim] {_fmt_bytes(f.size):>8}  "
            f"{'[green]HOT[/green]' if f.hot else '[dim]cold[/dim]'}  r={f.reads}"
            for f in self._rd.ls()[:6]
        ]
        segs = [
            f"  [dim]{s.name[:30]:<32}[/dim] {s.size_mb:.2f} MB  "
            f"{'📌' if s.pinned else '  '}  {s.owner}"
            for s in self._st.list_segments()[:5]
        ]
        return (
            "[bold cyan]◈ RAM Disk (512 MB)[/bold cyan]\n"
            "[dim]─────────────────────────────────────────────[/dim]\n"
            f"  {_bar(rd['usage_pct'])} {rd['usage_pct']:.1f}%  {_fmt_bytes(rd['used'])}\n"
            f"  Files={rd['files']}  R={rd['reads']}  W={rd['writes']}  "
            f"BW={rd['bw_mbps']:.2f} MB/s\n"
            + ("\n".join(files) if files else "  [dim](empty)[/dim]")
            + "\n\n[bold cyan]◈ Shared Storage (2 GB)[/bold cyan]\n"
            "[dim]─────────────────────────────────────────────[/dim]\n"
            f"  {_bar(st['usage_pct'])} {st['usage_pct']:.1f}%  {st['used_mb']:.1f} MB\n"
            f"  Segs={st['count']}  pinned={st['pinned']}  "
            f"R={st['total_reads']}  W={st['total_writes']}\n"
            + ("\n".join(segs) if segs else "  [dim](empty)[/dim]")
            + "\n\n[bold cyan]◈ GPU (8 GB VRAM)[/bold cyan]\n"
            "[dim]─────────────────────────────────────────────[/dim]\n"
            f"  VRAM  {_bar(gp['vram_pct'])} {gp['vram_pct']:.1f}%\n"
            f"  Cache {_bar(gp['cache_pct'])} {gp['cache_pct']:.1f}%  "
            f"hit={gp['cache_hit_rate']:.1f}%\n"
            f"  Util  {gp['utilization']:.1f}%  GFLOPS={gp['total_gflops']:.2f}\n"
            f"  Synapse {gp['synapse_used_mb']:.1f}/{gp['synapse_total_mb']} MB  "
            f"blocks={gp['synapse_blocks']}"
        )


class CognitivePanel(Static):
    def __init__(self, cpu: CognitiveCPU, **kw):
        super().__init__(**kw)
        self._cpu = cpu

    def render(self) -> str:
        s = self._cpu.stats()
        mods = [
            f"  [dim]{n:<22}[/dim] cap=[bold]{float(i['cap']):.2f}[/bold]  "
            f"load={_bar(float(i['load'])*100, 8)} v{i['v']}"
            for n, i in s["modules"].items()
        ]
        att = [
            f"  [cyan]{c:<20}[/cyan] {_bar(w*100, 10)} {w:.3f}"
            for c, w in s["top_attention"]
        ]
        kg = [
            f"  [dim]{n.concept:<20}[/dim] w={n.weight:.2f}  r={n.access_count}"
            for n in self._cpu.knowledge.top_concepts(8)
        ]
        return (
            f"[bold cyan]◈ Cognitive CPU  [{s['mode']}][/bold cyan]\n"
            "[dim]─────────────────────────────────────────────[/dim]\n"
            f"  Load  {_bar(s['cognitive_load']*100)} {s['cognitive_load']:.2f}\n"
            f"  Hist  {_spark(list(self._cpu.load_history))}\n"
            f"  Upgrades triggered: {s['upgrade_events']}\n"
            f"\n  [italic dim]{s['last_thought']}[/italic dim]\n"
            f"\n[bold cyan]◈ Modules[/bold cyan]\n"
            "[dim]─────────────────────────────────────────────[/dim]\n"
            + "\n".join(mods)
            + f"\n\n[bold cyan]◈ Attention ({s['attention_topics']} topics)[/bold cyan]\n"
            "[dim]─────────────────────────────────────────────[/dim]\n"
            + ("\n".join(att) if att else "  [dim](none)[/dim]")
            + f"\n\n[bold cyan]◈ Knowledge ({s['knowledge_nodes']}N/{s['knowledge_edges']}E)[/bold cyan]\n"
            "[dim]─────────────────────────────────────────────[/dim]\n"
            + ("\n".join(kg) if kg else "  [dim](empty)[/dim]")
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Command parser
# ═══════════════════════════════════════════════════════════════════════════════

_HELP_MD = """\
**Mistral Vibe — Commands**

**Projects & context**
| Command | Description |
|---|---|
| `/load <path>` | Ingest folder tree into active project context |
| `/project new <name>` | Create & activate a new project |
| `/project load <name>` | Activate a saved project |
| `/project list` | List all projects |
| `/instructions <text>` | Set system instructions for active project |
| `/unload` | Deactivate current project |

**Artifacts**
| Command | Description |
|---|---|
| `/artifact react` | Get a React build prompt hint |
| `/artifact html` | Get an HTML build prompt hint |
| `/launch <aid>` | Open artifact preview in browser |
| `/stop <aid>` | Stop preview server |

**Navigation**
`Ctrl+L` clear · `Ctrl+N` new session · `Ctrl+R` cycle model ·
`Ctrl+U` refresh dashboards · `Ctrl+H` this help · `Ctrl+C` exit
"""


class _Cmd:
    def __init__(self, handled: bool, msg: str = "",
                 kind: str = "info") -> None:
        self.handled = handled
        self.msg = msg
        self.kind = kind


class CommandParser:
    def __init__(self, pm: ProjectManager, engine: ArtifactEngine) -> None:
        self.pm = pm
        self.engine = engine

    def parse(self, text: str) -> _Cmd:
        if not text.startswith("/"):
            return _Cmd(False)

        parts = text.strip().split(None, 2)
        cmd = parts[0].lower()

        if cmd == "/help":
            return _Cmd(True, _HELP_MD, "info")

        if cmd == "/load":
            if len(parts) < 2:
                return _Cmd(True, "Usage: `/load <folder-path>`", "warn")
            try:
                _, summary = self.pm.load_folder(parts[1])
                return _Cmd(True, summary, "success")
            except Exception as exc:
                return _Cmd(True, f"**Error:** {exc}", "error")

        if cmd == "/project":
            if len(parts) < 2:
                return _Cmd(True, "Usage: `/project new|load|list <name>`", "warn")
            sub = parts[1].lower()
            if sub == "new":
                if len(parts) < 3:
                    return _Cmd(True, "Usage: `/project new <name>`", "warn")
                p = self.pm.new(parts[2])
                return _Cmd(True, f"Project **{p.name}** created and activated.", "success")
            if sub == "load":
                if len(parts) < 3:
                    return _Cmd(True, "Usage: `/project load <name>`", "warn")
                p = self.pm.activate(parts[2])
                if p:
                    return _Cmd(True, f"Project **{p.name}** activated ({p.file_count} files).", "success")
                return _Cmd(True, f"Project `{parts[2]}` not found.", "error")
            if sub == "list":
                projs = self.pm.list_all()
                if not projs:
                    return _Cmd(True, "No projects yet.", "info")
                lines = [
                    f"- **{p.name}** — {p.file_count} files, {p.total_kb:.0f} KB"
                    + (" ← active" if p == self.pm.active else "")
                    for p in projs
                ]
                return _Cmd(True, "\n".join(lines), "info")

        if cmd == "/instructions":
            instructions = text[len("/instructions"):].strip()
            if not instructions:
                return _Cmd(True, "Usage: `/instructions <text>`", "warn")
            if self.pm.set_instructions(instructions):
                return _Cmd(True, f"Instructions set ({len(instructions)} chars).", "success")
            return _Cmd(True, "No active project. Use `/project new <name>` first.", "warn")

        if cmd == "/unload":
            name = self.pm.active.name if self.pm.active else None
            self.pm.deactivate()
            return _Cmd(True, f"Project **{name}** deactivated." if name else "No active project.", "info")

        if cmd == "/artifact":
            if len(parts) < 2:
                return _Cmd(True, "Usage: `/artifact react|html`", "warn")
            kind = parts[1].lower()
            if kind == "react":
                return _Cmd(True,
                    "Tell Mistral what to build, e.g.:\n\n"
                    "> *Build a React dashboard with live charts and dark theme*\n\n"
                    "Mistral will generate `jsx` code which will be auto-detected "
                    "and launched as a preview.", "info")
            if kind == "html":
                return _Cmd(True,
                    "Tell Mistral what to build, e.g.:\n\n"
                    "> *Create a dark-themed HTML todo list app*\n\n"
                    "Mistral will generate `html` code which will be auto-detected "
                    "and served locally.", "info")

        if cmd == "/launch":
            if len(parts) < 2:
                return _Cmd(True, "Usage: `/launch <artifact-id>`", "warn")
            if self.engine.open_browser(parts[1]):
                return _Cmd(True, f"Opening **{parts[1]}** in browser.", "success")
            return _Cmd(True, f"Artifact `{parts[1]}` not found or has no URL.", "error")

        if cmd == "/stop":
            if len(parts) < 2:
                return _Cmd(True, "Usage: `/stop <artifact-id>`", "warn")
            if self.engine.stop(parts[1]):
                return _Cmd(True, f"Artifact **{parts[1]}** stopped.", "info")
            return _Cmd(True, f"Artifact `{parts[1]}` not found.", "error")

        return _Cmd(True, f"Unknown command `{cmd}`. Type `/help`.", "warn")


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
        Binding("ctrl+u", "refresh_dash", "Refresh", show=True),
        Binding("ctrl+h", "show_help",    "Help",    show=True),
    ]

    model: reactive[str] = reactive("mistral-large-latest")
    is_loading: reactive[bool] = reactive(False)

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "mistral-large-latest") -> None:
        super().__init__()
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        self.model = model
        self.client = MistralClient(api_key=self._api_key, model=self.model)
        self._conversation: List[Message] = []
        self._cwd = Path.cwd().name or "~"

        # Virtual hardware stack
        self.bus     = SystemBus()
        self.gpu     = VirtualGPU(self.bus)
        self.ramdisk = VirtualRAMDisk(self.bus)
        self.storage = SharedVirtualStorage(self.bus)
        self.npu     = VirtualNPU(self.bus, self.gpu)
        self.cnn     = CyberneticNN(self.bus, self.gpu, self.storage)
        self.shaper  = DeepLearningShaper(self.bus, self.storage)
        self.evo     = NeuroEvolution(self.bus, self.cnn, self.npu)
        self.cpu     = CognitiveCPU(self.bus, self.storage)

        # Artifact + project systems
        self.artifacts = ArtifactEngine(self.bus, self.ramdisk)
        self.projects  = ProjectManager(self.bus, self.ramdisk)
        self.commands  = CommandParser(self.projects, self.artifacts)

        self._total_tokens = 0

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield HeaderBar(model=self.model, cwd=self._cwd, id="header-bar")

        with TabbedContent(id="main-tabs"):
            with TabPane("Chat",      id="tab-chat"):
                with VerticalScroll(id="chat-scroll"):
                    yield Container(id="chat-messages")
                with Horizontal(id="input-row"):
                    yield Label("[bold cyan]>[/bold cyan]", id="prompt-icon")
                    yield Input(placeholder="Ask Mistral… or /help", id="message-input")

            with TabPane("Artifacts", id="tab-artifacts"):
                with VerticalScroll(id="art-scroll"):
                    yield ArtifactsPanel(self.artifacts, id="art-panel")

            with TabPane("Projects",  id="tab-projects"):
                with VerticalScroll(id="proj-scroll"):
                    yield ProjectsPanel(self.projects, id="proj-panel")

            with TabPane("NPU",       id="tab-npu"):
                with VerticalScroll(id="npu-scroll"):
                    yield NPUPanel(self.npu, id="npu-panel")

            with TabPane("Neural Net",id="tab-nn"):
                with VerticalScroll(id="nn-scroll"):
                    yield NeuralPanel(self.cnn, self.shaper, id="nn-panel")

            with TabPane("Evolution", id="tab-evo"):
                with VerticalScroll(id="evo-scroll"):
                    yield EvolutionPanel(self.evo, id="evo-panel")

            with TabPane("Memory",    id="tab-mem"):
                with VerticalScroll(id="mem-scroll"):
                    yield MemoryPanel(self.ramdisk, self.storage, self.gpu, id="mem-panel")

            with TabPane("Cognitive", id="tab-cog"):
                with VerticalScroll(id="cog-scroll"):
                    yield CognitivePanel(self.cpu, id="cog-panel")

        yield Footer()

    def on_mount(self) -> None:
        chat = self.query_one("#chat-messages", Container)
        chat.mount(WelcomeBanner())
        self.query_one("#message-input", Input).focus()
        self.set_interval(2.0, self._refresh_panels)

    # ── Panel refresh ─────────────────────────────────────────────────────────

    def _refresh_panels(self) -> None:
        for pid in ("#npu-panel", "#nn-panel", "#evo-panel", "#mem-panel",
                    "#cog-panel", "#art-panel", "#proj-panel"):
            try:
                self.query_one(pid).refresh()
            except Exception:
                pass

    def action_refresh_dash(self) -> None:
        self._refresh_panels()

    # ── Input handling ────────────────────────────────────────────────────────

    @on(Input.Submitted, "#message-input")
    def on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self.is_loading:
            return
        self.query_one("#message-input", Input).value = ""
        self._handle_input(text)

    def _handle_input(self, text: str) -> None:
        result = self.commands.parse(text)
        if result.handled:
            chat = self.query_one("#chat-messages", Container)
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            chat.mount(UserTurn(text))
            chat.mount(SystemMessage(result.msg, result.kind))
            scroll.scroll_end(animate=False)
            self._refresh_panels()
        else:
            self._dispatch(text)

    # ── LLM dispatch ─────────────────────────────────────────────────────────

    @work(exclusive=True, thread=False)
    async def _dispatch(self, text: str) -> None:
        self.is_loading = True
        chat   = self.query_one("#chat-messages", Container)
        scroll = self.query_one("#chat-scroll",   VerticalScroll)
        header = self.query_one("#header-bar",    HeaderBar)

        await chat.mount(UserTurn(text))
        scroll.scroll_end(animate=False)
        self._conversation.append({"role": "user", "content": text})

        # NPU embed pass
        npu_metrics: dict = {}
        try:
            _, npu_metrics = self.npu.process_text(text)
            self.ramdisk.write(
                f"/conv/embed_{int(time.time())}.bin",
                text.encode(),
                tags=["embedding"],
            )
        except Exception:
            pass

        # Assistant streaming turn
        asst = AssistantTurn()
        await chat.mount(asst)
        scroll.scroll_end(animate=False)
        header.set_status("[dim]● thinking…[/dim]")

        full_response = ""
        try:
            system_prompt = self.projects.get_system_prompt(SYSTEM_PROMPT)
            async for chunk in self.client.stream_chat(
                self._conversation,
                system_prompt=system_prompt,
            ):
                full_response += chunk
                asst.append(chunk)
                scroll.scroll_end(animate=False)

            self._conversation.append({"role": "assistant", "content": full_response})
            self._total_tokens += len(full_response.split())
            proj_tag = f"  [{self.projects.active.name}]" if self.projects.active else ""
            header.set_status(
                f"[dim]~{self._total_tokens} tok  gen={self.evo.generation}{proj_tag}[/dim]"
            )
        except RuntimeError as exc:
            asst.append(f"\n**Error:** {exc}")
            header.set_status("[red]● error[/red]")
        finally:
            self.is_loading = False
            self.query_one("#message-input", Input).focus()

        if not full_response:
            return

        self._post_exchange(text, full_response, npu_metrics, chat, scroll)

    def _post_exchange(
        self,
        user: str,
        assistant: str,
        npu_metrics: dict,
        chat: Container,
        scroll: VerticalScroll,
    ) -> None:
        # Store conversation in RAM disk
        self.ramdisk.write(
            f"/conv/turn_{len(self._conversation):04d}.txt",
            f"U: {user}\nA: {assistant[:500]}",
            tags=["conversation"],
        )

        # Detect & launch artifacts
        new_arts = self.artifacts.detect_and_create(assistant)
        for art in new_arts:
            npm_note = "" if NPM_AVAILABLE else " (CDN fallback — no npm)"
            msg = (
                f"**Artifact** `{art.aid}` detected ({art.kind}{npm_note})\n"
                + (f"Preview server starting on port **{art.port}**…\n" if art.port else "")
                + f"Use `/launch {art.aid}` to open · Check the **Artifacts** tab."
            )
            self.call_after_refresh(
                lambda m=msg: (
                    chat.mount(SystemMessage(m, "success")),
                    scroll.scroll_end(animate=False),
                )
            )

        # Virtual hardware updates
        try:
            self.shaper.inject_llm_signal(assistant, self.cnn)
            self.shaper.step(self.cnn)
        except Exception:
            pass
        try:
            self.evo.evolve(response_text=assistant)
        except Exception:
            pass
        try:
            self.cpu.process_exchange(user, assistant, npu_metrics=npu_metrics)
        except Exception:
            pass

        self._refresh_panels()

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

    def action_show_help(self) -> None:
        chat   = self.query_one("#chat-messages", Container)
        scroll = self.query_one("#chat-scroll",   VerticalScroll)
        chat.mount(SystemMessage(_HELP_MD, "info"))
        scroll.scroll_end(animate=False)

    def watch_is_loading(self, loading: bool) -> None:
        inp = self.query_one("#message-input", Input)
        if loading:
            inp.placeholder = "Mistral is thinking…"
            inp.disabled = True
        else:
            inp.placeholder = "Ask Mistral… or /help"
            inp.disabled = False


# ─── Entry point ──────────────────────────────────────────────────────────────

def run(api_key: Optional[str] = None, model: str = "mistral-large-latest") -> None:
    MistralVibeApp(api_key=api_key, model=model).run()


if __name__ == "__main__":
    run()
