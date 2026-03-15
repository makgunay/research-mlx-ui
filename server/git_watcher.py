"""Watches results.tsv and git log for new experiments."""

import asyncio
import csv
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Experiment:
    commit: str
    val_bpb: float
    memory_gb: float
    status: str          # "keep" | "discard" | "crash"
    description: str
    diff: str = field(default="", repr=False)


class GitWatcher:
    def __init__(self, broadcaster, poll_interval: float = 2.0):
        self.broadcaster = broadcaster
        self.poll_interval = poll_interval
        self._seen_count = 0
        self._experiments: list[Experiment] = []
        # Load existing experiments from results.tsv on startup
        self._load_existing()

    def _load_existing(self):
        """Load experiments already in results.tsv (e.g. after server restart)."""
        existing = self._read_results_tsv()
        self._experiments = existing
        self._seen_count = len(existing)

    def reload(self):
        """Reload experiments from results.tsv (called on project switch)."""
        self._load_existing()

    async def watch(self):
        while True:
            await asyncio.sleep(self.poll_interval)
            fresh = self._read_results_tsv()
            if len(fresh) > self._seen_count:
                new_experiments = fresh[self._seen_count:]
                self._seen_count = len(fresh)
                for exp in new_experiments:
                    self._experiments.append(exp)
                    await self.broadcaster.broadcast({
                        "type": "experiment_done",
                        "commit": exp.commit,
                        "val_bpb": exp.val_bpb,
                        "memory_gb": exp.memory_gb,
                        "status": exp.status,
                        "description": exp.description,
                    })

    def get_all_experiments(self) -> list[dict]:
        return [
            {"commit": e.commit, "val_bpb": e.val_bpb, "memory_gb": e.memory_gb,
             "status": e.status, "description": e.description}
            for e in self._experiments
        ]

    def get_experiment_with_diff(self, commit: str) -> dict | None:
        for exp in self._experiments:
            if exp.commit == commit:
                if not exp.diff:
                    exp.diff = self._get_diff(commit)
                return {
                    "commit": exp.commit, "val_bpb": exp.val_bpb,
                    "memory_gb": exp.memory_gb, "status": exp.status,
                    "description": exp.description, "diff": exp.diff,
                }
        return None

    @staticmethod
    def _read_results_tsv() -> list[Experiment]:
        path = Path("results.tsv")
        if not path.exists():
            return []
        experiments = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    experiments.append(Experiment(
                        commit=row.get("commit", "").strip(),
                        val_bpb=float(row.get("val_bpb", 0)),
                        memory_gb=float(row.get("memory_gb", 0)),
                        status=row.get("status", "").strip(),
                        description=row.get("description", "").strip(),
                    ))
                except (ValueError, KeyError):
                    continue
        return experiments

    @staticmethod
    def _get_diff(commit: str) -> str:
        result = subprocess.run(
            ["git", "show", "--unified=5", commit, "--", "train.py"],
            capture_output=True, text=True,
        )
        return result.stdout
