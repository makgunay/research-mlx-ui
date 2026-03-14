"""Spawns and monitors the agent + training subprocess."""

import asyncio
import subprocess
import time
from pathlib import Path


ALLOWED_AGENTS = {"claude", "aider", "cursor"}


class ProcessManager:
    def __init__(self, broadcaster):
        self.broadcaster = broadcaster
        self.process = None
        self._stream_task = None
        self._active = False
        self._branch = None
        self._started_at = None

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

        # Spawn agent as subprocess (no shell — prevents command injection)
        # --print: non-interactive mode (single prompt, streamed output)
        # --dangerously-skip-permissions: allow file writes and bash without
        #   interactive approval (required since --print has no TTY)
        self.process = await asyncio.create_subprocess_exec(
            agent_command, "--print", "--dangerously-skip-permissions",
            "Please read program.md and start a new research session",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=Path.cwd(),
        )

        self._stream_task = asyncio.create_task(self._stream_output())

        await self.broadcaster.broadcast({
            "type": "session_started",
            "branch": self._branch,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    async def _stream_output(self):
        """Stream agent stdout to WebSocket clients."""
        async for raw_line in self.process.stdout:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue
            await self.broadcaster.broadcast({
                "type": "agent_log",
                "level": self._classify(line),
                "text": line,
            })
        # Process ended
        self._active = False
        await self.broadcaster.broadcast({
            "type": "session_stopped",
            "total_experiments": 0,
            "best_val_bpb": 0,
        })

    @staticmethod
    def _classify(text: str) -> str:
        lower = text.lower()
        if "val_bpb" in lower:
            return "result"
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

    def get_status(self) -> dict:
        return {
            "active": self._active,
            "branch": self._branch,
            "started_at": self._started_at,
            "pid": self.process.pid if self.process else None,
        }
