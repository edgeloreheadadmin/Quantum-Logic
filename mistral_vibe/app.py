"""Mistral Vibe - Codex-style TUI for Mistral AI.

A terminal-based chat interface inspired by OpenAI Codex CLI,
powered by Mistral AI models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

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
)

from .client import AVAILABLE_MODELS, Message, MistralClient

# ─── Message Widgets ──────────────────────────────────────────────────────────


class WelcomeBanner(Static):
    """Startup welcome banner shown before first message."""

    DEFAULT_CSS = """
    WelcomeBanner {
        padding: 1 2;
        color: $text-muted;
        text-align: center;
    }
    """

    def render(self) -> str:
        return (
            "[bold cyan]✦  Mistral Vibe[/bold cyan]\n"
            "[dim]────────────────────────────────────────[/dim]\n"
            "[dim]Type a message and press [bold]Enter[/bold] to chat.[/dim]\n"
            "[dim]Press [bold]Ctrl+H[/bold] for help & shortcuts.[/dim]"
        )


class UserTurn(Static):
    """A user message bubble."""

    DEFAULT_CSS = """
    UserTurn {
        background: $surface;
        border-left: thick $accent;
        padding: 0 1;
        margin-bottom: 1;
    }
    .user-label {
        color: $accent;
        text-style: bold;
    }
    .user-content {
        color: $text;
        padding-top: 0;
    }
    """

    def __init__(self, content: str, **kwargs):
        super().__init__(**kwargs)
        self._content = content

    def compose(self) -> ComposeResult:
        yield Label("[bold]You[/bold]", classes="user-label")
        yield Static(self._content, classes="user-content")


class AssistantTurn(Static):
    """An assistant message bubble with streaming Markdown support."""

    DEFAULT_CSS = """
    AssistantTurn {
        padding: 0 1;
        margin-bottom: 1;
    }
    .assistant-label {
        color: $secondary;
        text-style: bold;
    }
    AssistantTurn LoadingIndicator {
        background: transparent;
        color: $secondary;
        height: 1;
    }
    """

    content: reactive[str] = reactive("", recompose=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Label("[bold]✦ Mistral[/bold]", classes="assistant-label")
        if self.content:
            yield Markdown(self.content)
        else:
            yield LoadingIndicator()

    def append(self, chunk: str) -> None:
        self.content = self.content + chunk


class HelpOverlay(Static):
    """Keyboard shortcut help panel."""

    DEFAULT_CSS = """
    HelpOverlay {
        background: $surface;
        border: round $accent;
        padding: 1 2;
        margin: 1 4;
    }
    """

    def render(self) -> str:
        lines = [
            "[bold cyan]Keyboard Shortcuts[/bold cyan]",
            "[dim]──────────────────────────────────[/dim]",
            "  [bold]Enter[/bold]        Send message",
            "  [bold]Ctrl+L[/bold]       Clear conversation",
            "  [bold]Ctrl+N[/bold]       New session",
            "  [bold]Ctrl+R[/bold]       Cycle Mistral models",
            "  [bold]Ctrl+H[/bold]       Toggle this help",
            "  [bold]Ctrl+C[/bold]       Exit",
            "",
            "[bold cyan]Models[/bold cyan]",
            "[dim]──────────────────────────────────[/dim]",
        ]
        for m in AVAILABLE_MODELS:
            lines.append(f"  • {m}")
        return "\n".join(lines)


# ─── Header Bar ───────────────────────────────────────────────────────────────


class HeaderBar(Horizontal):
    """Custom header bar showing app name, model, and working directory."""

    DEFAULT_CSS = """
    HeaderBar {
        height: 3;
        background: $surface;
        border-bottom: tall $panel;
        padding: 0 2;
        align: left middle;
    }
    .header-logo {
        color: $accent;
        text-style: bold;
        width: auto;
    }
    .header-sep {
        color: $text-muted;
        width: auto;
        margin: 0 1;
    }
    .header-model {
        color: $secondary;
        width: auto;
    }
    .header-dir {
        color: $success;
        width: auto;
        margin-left: 2;
    }
    .header-status {
        color: $text-muted;
        width: 1fr;
        text-align: right;
    }
    """

    def __init__(self, model: str, cwd: str, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._cwd = cwd

    def compose(self) -> ComposeResult:
        yield Label("✦ Mistral Vibe", classes="header-logo", id="logo")
        yield Label("│", classes="header-sep")
        yield Label(self._model, classes="header-model", id="header-model")
        yield Label("│", classes="header-sep")
        yield Label(f"~/{self._cwd}", classes="header-dir", id="header-dir")
        yield Label("", classes="header-status", id="header-status")

    def set_model(self, model: str) -> None:
        self.query_one("#header-model", Label).update(model)

    def set_status(self, text: str) -> None:
        self.query_one("#header-status", Label).update(text)


# ─── Main Application ─────────────────────────────────────────────────────────


class MistralVibeApp(App):
    """Codex-style TUI for Mistral AI."""

    CSS_PATH = Path(__file__).parent / "styles.tcss"
    TITLE = "Mistral Vibe"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Exit", priority=True, show=True),
        Binding("ctrl+l", "clear_chat", "Clear", show=True),
        Binding("ctrl+n", "new_session", "New", show=True),
        Binding("ctrl+r", "cycle_model", "Model", show=True),
        Binding("ctrl+h", "toggle_help", "Help", show=True),
    ]

    model: reactive[str] = reactive("mistral-large-latest")
    is_loading: reactive[bool] = reactive(False)
    show_help: reactive[bool] = reactive(False)

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-large-latest",
    ):
        super().__init__()
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        self.model = model
        self.client = MistralClient(api_key=self._api_key, model=self.model)
        self._conversation: list[Message] = []
        self._cwd = Path.cwd().name or "~"
        self._total_tokens = 0

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield HeaderBar(model=self.model, cwd=self._cwd, id="header-bar")
        with VerticalScroll(id="chat-scroll"):
            yield Container(id="chat-messages")
        with Horizontal(id="input-row"):
            yield Label("[dim bold]>[/dim bold]", id="prompt-icon")
            yield Input(
                placeholder="Ask Mistral anything...",
                id="message-input",
            )
        yield Footer()

    def on_mount(self) -> None:
        self._show_welcome()
        self.query_one("#message-input", Input).focus()

    # ── Welcome ───────────────────────────────────────────────────────────────

    def _show_welcome(self) -> None:
        chat = self.query_one("#chat-messages", Container)
        chat.mount(WelcomeBanner())

    # ── Input Handling ────────────────────────────────────────────────────────

    @on(Input.Submitted, "#message-input")
    def on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self.is_loading:
            return
        self.query_one("#message-input", Input).value = ""
        self._dispatch_message(text)

    @work(exclusive=True, thread=False)
    async def _dispatch_message(self, text: str) -> None:
        self.is_loading = True
        chat = self.query_one("#chat-messages", Container)
        scroll = self.query_one("#chat-scroll", VerticalScroll)

        # Mount user turn
        user_turn = UserTurn(text)
        await chat.mount(user_turn)
        scroll.scroll_end(animate=False)

        self._conversation.append({"role": "user", "content": text})

        # Mount assistant turn (shows loading spinner until content arrives)
        asst_turn = AssistantTurn()
        await chat.mount(asst_turn)
        scroll.scroll_end(animate=False)

        header = self.query_one("#header-bar", HeaderBar)
        header.set_status("[dim]● thinking…[/dim]")

        try:
            full_response = ""
            async for chunk in self.client.stream_chat(self._conversation):
                full_response += chunk
                asst_turn.append(chunk)
                scroll.scroll_end(animate=False)
            self._conversation.append(
                {"role": "assistant", "content": full_response}
            )
            self._total_tokens += len(full_response.split()) * 1  # rough est.
            header.set_status(
                f"[dim]~{self._total_tokens} tokens  {len(self._conversation) // 2} turns[/dim]"
            )
        except RuntimeError as exc:
            asst_turn.append(f"\n**Error:** {exc}")
            header.set_status("[red]● error[/red]")
        finally:
            self.is_loading = False
            self.query_one("#message-input", Input).focus()

    # ── Reactive Watchers ─────────────────────────────────────────────────────

    def watch_is_loading(self, loading: bool) -> None:
        inp = self.query_one("#message-input", Input)
        if loading:
            inp.placeholder = "Mistral is thinking…"
            inp.disabled = True
        else:
            inp.placeholder = "Ask Mistral anything…"
            inp.disabled = False

    def watch_show_help(self, show: bool) -> None:
        chat = self.query_one("#chat-messages", Container)
        existing = chat.query("HelpOverlay")
        if show:
            if not existing:
                chat.mount(HelpOverlay())
                self.query_one("#chat-scroll", VerticalScroll).scroll_end(
                    animate=False
                )
        else:
            for node in existing:
                node.remove()

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_clear_chat(self) -> None:
        self._conversation = []
        self._total_tokens = 0
        chat = self.query_one("#chat-messages", Container)
        chat.remove_children()
        self._show_welcome()
        header = self.query_one("#header-bar", HeaderBar)
        header.set_status("")

    def action_new_session(self) -> None:
        self.action_clear_chat()

    def action_cycle_model(self) -> None:
        idx = AVAILABLE_MODELS.index(self.model) if self.model in AVAILABLE_MODELS else 0
        self.model = AVAILABLE_MODELS[(idx + 1) % len(AVAILABLE_MODELS)]
        self.client.model = self.model
        self.query_one("#header-bar", HeaderBar).set_model(self.model)
        self.notify(f"Model → {self.model}", severity="information", timeout=3)

    def action_toggle_help(self) -> None:
        self.show_help = not self.show_help


# ─── Entry Point ──────────────────────────────────────────────────────────────


def run(api_key: Optional[str] = None, model: str = "mistral-large-latest") -> None:
    """Launch the Mistral Vibe TUI."""
    app = MistralVibeApp(api_key=api_key, model=model)
    app.run()


if __name__ == "__main__":
    run()
