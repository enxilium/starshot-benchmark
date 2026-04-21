from __future__ import annotations

from rich.console import Console
from rich.text import Text

from app.core.events import (
    Event,
    EventLog,
    MeshGenerated,
    PhaseStarted,
    RunCompleted,
    RunFailed,
    RunStarted,
    StateRepoWrite,
    StepCompleted,
    StepRetried,
    StepStarted,
)


class TerminalLogger:
    """
    Subscribes to an EventLog and renders events to stdout with color coding.

    Runs as a background task; terminates when the EventLog closes. Color
    conventions: phase = bold cyan, step start = blue, step completed = green,
    retry = yellow (with conflict diff), failure = bold red, mesh = magenta,
    run completed = bold green.
    """

    def __init__(
        self,
        event_log: EventLog,
        console: Console | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        self.event_log = event_log
        self.console = console or Console()
        self.verbose = verbose

    async def run(self) -> None:
        async for event in self.event_log.subscribe():
            self._render(event)

    def _prefix(self, event: Event) -> Text:
        prefix = Text()
        prefix.append(f"[{event.run_id[:8]}] ", style="dim")
        step_or_phase = getattr(event, "step_id", None) or getattr(event, "phase", None)
        if step_or_phase:
            prefix.append(f"[{step_or_phase}] ", style="bold blue")
        return prefix

    def _render(self, event: Event) -> None:
        method = getattr(self, f"_render_{event.type}", self._render_default)
        method(event)

    def _render_run_started(self, event: RunStarted) -> None:
        msg = Text("▶ run started", style="bold cyan")
        msg.append(f" model={event.model_id} ", style="dim")
        excerpt = event.prompt if len(event.prompt) <= 60 else event.prompt[:60] + "…"
        msg.append(f"prompt={excerpt!r}", style="white")
        self.console.print(self._prefix(event), msg)

    def _render_phase_started(self, event: PhaseStarted) -> None:
        msg = Text(f"▸ {event.phase} started ", style="bold cyan")
        msg.append(f"scope={event.scope_id}", style="dim")
        self.console.print(self._prefix(event), msg)

    def _render_step_started(self, event: StepStarted) -> None:
        msg = Text("· step started", style="blue")
        if self.verbose and event.inputs_summary:
            msg.append(f" inputs={event.inputs_summary}", style="dim")
        self.console.print(self._prefix(event), msg)

    def _render_step_completed(self, event: StepCompleted) -> None:
        msg = Text("✓ step completed", style="green")
        msg.append(f" ({event.duration_ms:.0f} ms)", style="dim")
        if self.verbose and event.output_summary:
            msg.append(f" output={event.output_summary}", style="dim")
        self.console.print(self._prefix(event), msg)

    def _render_step_retried(self, event: StepRetried) -> None:
        msg = Text(f"↻ retry #{event.attempt}", style="yellow bold")
        msg.append(f" — {event.conflict.validator}: {event.conflict.detail}", style="yellow")
        self.console.print(self._prefix(event), msg)

    def _render_state_repo_write(self, event: StateRepoWrite) -> None:
        msg = Text(f"≡ state.{event.entry_type} ← {event.scope_id}", style="dim cyan")
        self.console.print(self._prefix(event), msg)

    def _render_mesh_generated(self, event: MeshGenerated) -> None:
        msg = Text(f"▣ mesh {event.object_id}", style="magenta")
        msg.append(f" ({event.duration_ms:.0f} ms, {event.backend})", style="dim")
        self.console.print(self._prefix(event), msg)

    def _render_run_completed(self, event: RunCompleted) -> None:
        msg = Text("■ run completed", style="bold green")
        msg.append(f" ({event.total_duration_ms:.0f} ms, glb={event.glb_url})", style="dim")
        self.console.print(self._prefix(event), msg)
        if event.retry_summary:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(event.retry_summary.items()))
            self.console.print(Text(f"  retries: {parts}", style="yellow dim"))

    def _render_run_failed(self, event: RunFailed) -> None:
        msg = Text("✗ run failed", style="bold red")
        msg.append(f" step={event.step_id} error={event.error}", style="red")
        self.console.print(self._prefix(event), msg)

    def _render_default(self, event: Event) -> None:
        self.console.print(self._prefix(event), Text(f"event {event.type}", style="dim"))
