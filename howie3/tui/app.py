"""Howie v3 TUI — a thin Textual shell over the shared command registry.

Design: one scrollback log, one input. Commands run in worker threads so the
UI never blocks; every command's output comes from views.py, the same code
the CLI prints. No console monkey-patching, no duplicated dispatch.
"""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Input, RichLog

from ..commands import REGISTRY, dispatch
from ..config import Settings

BANNER = """[bold green]HOWIE[/bold green] [dim]v3 — marginal value engine[/dim]
Type a command ([bold]help[/bold] to list them). Up/Down for history, Tab to complete, Ctrl+Q to quit."""


class HowieApp(App):
    TITLE = "Howie"
    CSS = """
    Screen { layout: vertical; }
    #log { height: 1fr; border: round $surface-lighten-2; padding: 0 1; }
    #cmd { dock: bottom; margin: 0 0 1 0; }
    """
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.settings = Settings()
        self.history: list = []
        self.history_pos = 0

    def compose(self) -> ComposeResult:
        with Vertical():
            yield RichLog(id="log", markup=True, wrap=True, highlight=False)
            yield Input(id="cmd", placeholder="board 3 · pick 4 have=\"CMC, ARSB\" · player Puka Nacua · help")
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#log", RichLog)
        log.write(BANNER)
        if not self.settings.db_path.exists():
            log.write("[red]No howie.db found — run `refresh` to build it.[/red]")
        self.query_one("#cmd", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        event.input.value = ""
        if not line:
            return
        if line in ("quit", "exit", "q"):
            self.exit()
            return
        self.history.append(line)
        self.history_pos = len(self.history)
        log = self.query_one("#log", RichLog)
        log.write(f"[bold cyan]›[/bold cyan] [bold]{line}[/bold]")
        self.run_worker(lambda: self._execute(line), thread=True, exclusive=False)

    def _execute(self, line: str) -> None:
        renderables = dispatch(self.settings, line)
        if renderables is None:
            return
        self.call_from_thread(self._write_results, renderables)

    def _write_results(self, renderables) -> None:
        log = self.query_one("#log", RichLog)
        for r in renderables:
            log.write(r)
        log.write("")

    def on_key(self, event) -> None:
        cmd = self.query_one("#cmd", Input)
        if not cmd.has_focus:
            return
        if event.key == "up" and self.history:
            self.history_pos = max(0, self.history_pos - 1)
            cmd.value = self.history[self.history_pos]
            cmd.cursor_position = len(cmd.value)
            event.stop()
        elif event.key == "down" and self.history:
            self.history_pos = min(len(self.history), self.history_pos + 1)
            cmd.value = self.history[self.history_pos] if self.history_pos < len(self.history) else ""
            cmd.cursor_position = len(cmd.value)
            event.stop()
        elif event.key == "tab" and cmd.value:
            prefix = cmd.value.split()[0].lower()
            matches = [n for n in REGISTRY if n.startswith(prefix)]
            if len(matches) == 1 and " " not in cmd.value:
                cmd.value = matches[0] + " "
                cmd.cursor_position = len(cmd.value)
                event.stop()

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()
