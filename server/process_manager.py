"""Spawns and monitors the agent + training subprocess."""

import asyncio
import re
import subprocess
import time
from pathlib import Path


ALLOWED_AGENTS = {"claude", "aider", "cursor"}

# Regex for training step output: "step   100 | loss 3.91 | tok/s 35,000 | elapsed 44.1s"
TRAINING_TICK_RE = re.compile(
    r"step\s+(\d+)\s*\|\s*loss\s+([\d.]+)\s*\|\s*tok/s\s+([\d,]+)\s*\|\s*elapsed\s+([\d.]+)s"
)

# Detect when agent starts a training run
EXPERIMENT_START_RE = re.compile(
    r"(uv run (?:python )?train\.py|Running.*train\.py|`uv run train\.py`)", re.IGNORECASE
)

# Detect hypothesis from agent reasoning
HYPOTHESIS_RE = re.compile(
    r"(?:hypothesis|idea|trying|experiment|change)[:]\s*(.{10,120})", re.IGNORECASE
)


class ProcessManager:
    def __init__(self, broadcaster):
        self.broadcaster = broadcaster
        self.process = None
        self._stream_task = None
        self._heartbeat_task = None
        self._active = False
        self._branch = None
        self._started_at = None
        self._experiment_count = 0
        self._last_hypothesis = ""

    async def start_session(self, agent_command: str, branch_name: str):
        if self._active:
            raise RuntimeError("Session already active")

        if agent_command not in ALLOWED_AGENTS:
            raise ValueError(f"Unknown agent: {agent_command}. Allowed: {ALLOWED_AGENTS}")

        # Create git branch
        subprocess.run(
            ["git", "checkout", "-b", f"autoresearch/{branch_name}"],
            check=True,
        )
        self._branch = f"autoresearch/{branch_name}"
        self._started_at = time.time()
        self._active = True
        self._experiment_count = 0
        Path(".session-active").touch()

        # Spawn agent as subprocess (no shell — prevents command injection)
        self.process = await asyncio.create_subprocess_exec(
            agent_command, "--print", "--dangerously-skip-permissions",
            "Please read program.md and start a new research session",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=Path.cwd(),
        )

        self._stream_task = asyncio.create_task(self._stream_output())
        self._heartbeat_task = asyncio.create_task(self._heartbeat())

        await self.broadcaster.broadcast({
            "type": "session_started",
            "branch": self._branch,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    async def _heartbeat(self):
        """Emit session heartbeat every 5 seconds so the UI knows we're alive."""
        while self._active:
            await asyncio.sleep(5)
            if not self._active:
                break
            elapsed = time.time() - self._started_at if self._started_at else 0
            await self.broadcaster.broadcast({
                "type": "heartbeat",
                "elapsed_seconds": round(elapsed, 1),
                "active": self._active,
                "experiments_count": self._experiment_count,
            })

    async def _stream_output(self):
        """Stream agent stdout to WebSocket clients with structured event parsing."""
        async for raw_line in self.process.stdout:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue

            # Try to parse as training_tick
            tick_match = TRAINING_TICK_RE.search(line)
            if tick_match:
                await self.broadcaster.broadcast({
                    "type": "training_tick",
                    "step": int(tick_match.group(1)),
                    "loss": float(tick_match.group(2)),
                    "tokens_per_sec": int(tick_match.group(3).replace(",", "")),
                    "elapsed_seconds": float(tick_match.group(4)),
                })

            # Detect experiment start (agent running train.py)
            if EXPERIMENT_START_RE.search(line):
                self._experiment_count += 1
                await self.broadcaster.broadcast({
                    "type": "experiment_started",
                    "n": self._experiment_count,
                    "hypothesis": self._last_hypothesis,
                })
                self._last_hypothesis = ""

            # Try to capture hypothesis from agent reasoning
            hyp_match = HYPOTHESIS_RE.search(line)
            if hyp_match and not tick_match:
                self._last_hypothesis = hyp_match.group(1).strip()

            # Always emit the raw agent log
            await self.broadcaster.broadcast({
                "type": "agent_log",
                "level": self._classify(line),
                "text": line,
            })

        # Process ended
        self._active = False
        Path(".session-active").unlink(missing_ok=True)
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        await self.broadcaster.broadcast({
            "type": "session_stopped",
            "total_experiments": self._experiment_count,
        })

    @staticmethod
    def _classify(text: str) -> str:
        lower = text.lower()
        if "val_bpb" in lower:
            return "result"
        if TRAINING_TICK_RE.search(text):
            return "training"
        if any(k in text for k in ["def ", "class ", "import ", "return "]):
            return "code"
        if any(k in lower for k in ["error", "crash", "traceback", "exception"]):
            return "error"
        return "info"

    async def stop_session(self):
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
        self._active = False
        Path(".session-active").unlink(missing_ok=True)
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

    def get_status(self) -> dict:
        elapsed = time.time() - self._started_at if self._started_at else 0
        return {
            "active": self._active,
            "branch": self._branch,
            "started_at": self._started_at,
            "elapsed_seconds": round(elapsed, 1),
            "pid": self.process.pid if self.process else None,
            "experiments_count": self._experiment_count,
        }
