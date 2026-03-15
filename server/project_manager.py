"""Manages research projects — CRUD, switching, forking."""

import csv
import re
import shutil
import subprocess
from pathlib import Path

PROJECT_FILE = Path(".active-project")
RESULTS_FILE = Path("results.tsv")
TRAIN_FILE = Path("train.py")
RESULTS_HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"


def _valid_name(name: str) -> bool:
    return bool(re.match(r"^[a-z0-9]([a-z0-9-]{0,48}[a-z0-9])?$", name))


def _results_path(name: str) -> Path:
    return Path(f"results-{name}.tsv")


def _trainpy_path(name: str) -> Path:
    return Path(f"trainpy-{name}.py")


def _read_project_stats(name: str) -> dict:
    """Read experiment count and best BPB from a project's results file."""
    path = _results_path(name)
    if not path.exists():
        return {"name": name, "experiments": 0, "kept": 0, "best_bpb": None}

    experiments = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                experiments.append({
                    "val_bpb": float(row.get("val_bpb", 0)),
                    "status": row.get("status", "").strip(),
                })
            except (ValueError, KeyError):
                continue

    kept = [e for e in experiments if e["status"] == "keep"]
    best = min(kept, key=lambda e: e["val_bpb"])["val_bpb"] if kept else None

    return {
        "name": name,
        "experiments": len(experiments),
        "kept": len(kept),
        "best_bpb": best,
    }


def get_active_project() -> str | None:
    if PROJECT_FILE.exists():
        return PROJECT_FILE.read_text().strip() or None
    return None


def list_projects() -> list[dict]:
    """List all projects with stats."""
    projects = []
    active = get_active_project()

    for f in sorted(Path(".").glob("results-*.tsv")):
        name = f.stem.removeprefix("results-")
        stats = _read_project_stats(name)
        stats["active"] = (name == active)
        projects.append(stats)

    # If no projects exist but results.tsv does, create a "default" project
    if not projects and RESULTS_FILE.exists():
        _migrate_legacy()
        return list_projects()

    return projects


def create_project(name: str, fork_from: str | None = None) -> dict:
    """Create a new project, optionally forking train.py from another."""
    if not _valid_name(name):
        raise ValueError(f"Invalid project name: '{name}'. Use lowercase alphanumeric + hyphens.")

    if _results_path(name).exists():
        raise ValueError(f"Project '{name}' already exists.")

    # Create empty results file
    _results_path(name).write_text(RESULTS_HEADER)

    # Get train.py: fork from another project, or baseline from git
    if fork_from:
        if not _trainpy_path(fork_from).exists():
            raise ValueError(f"Fork source '{fork_from}' has no train.py snapshot.")
        shutil.copy(_trainpy_path(fork_from), _trainpy_path(name))
    else:
        # Get baseline train.py from first commit
        result = subprocess.run(
            ["git", "log", "--reverse", "--format=%H", "--", "train.py"],
            capture_output=True, text=True,
        )
        commits = result.stdout.strip().split("\n")
        if commits and commits[0]:
            baseline = subprocess.run(
                ["git", "show", f"{commits[0]}:train.py"],
                capture_output=True, text=True,
            )
            _trainpy_path(name).write_text(baseline.stdout)
        elif TRAIN_FILE.exists():
            shutil.copy(TRAIN_FILE, _trainpy_path(name))

    # Activate the new project — clean up files if activation fails
    try:
        activate_project(name)
    except Exception:
        _results_path(name).unlink(missing_ok=True)
        _trainpy_path(name).unlink(missing_ok=True)
        raise

    return _read_project_stats(name)


def is_session_active() -> bool:
    """Check if an agent session is currently running (process_manager sets this)."""
    return Path(".session-active").exists()


def activate_project(name: str) -> dict:
    """Switch to a project: save current state, restore target state."""
    if not _valid_name(name):
        raise ValueError(f"Invalid project name: '{name}'.")
    if is_session_active():
        raise RuntimeError("Cannot switch projects while a session is running. Stop the session first.")
    if not _results_path(name).exists():
        raise ValueError(f"Project '{name}' does not exist.")

    current = get_active_project()

    # Save current project state
    if current and current != name:
        if RESULTS_FILE.exists():
            shutil.copy(RESULTS_FILE, _results_path(current))
        if TRAIN_FILE.exists():
            shutil.copy(TRAIN_FILE, _trainpy_path(current))

    # Restore target project state
    shutil.copy(_results_path(name), RESULTS_FILE)
    if _trainpy_path(name).exists():
        shutil.copy(_trainpy_path(name), TRAIN_FILE)

    # Update active marker
    PROJECT_FILE.write_text(name)

    return _read_project_stats(name)


def delete_project(name: str, prune_branches: bool = False) -> dict:
    """Delete a project and optionally its git branches."""
    if not _valid_name(name):
        raise ValueError(f"Invalid project name: '{name}'.")
    active = get_active_project()
    if name == active:
        raise ValueError("Cannot delete the active project. Switch to another first.")

    removed = {"name": name, "files_removed": [], "branches_removed": []}

    rp = _results_path(name)
    if rp.exists():
        rp.unlink()
        removed["files_removed"].append(str(rp))

    tp = _trainpy_path(name)
    if tp.exists():
        tp.unlink()
        removed["files_removed"].append(str(tp))

    if prune_branches:
        result = subprocess.run(
            ["git", "branch", "--list", f"autoresearch/{name}/*"],
            capture_output=True, text=True,
        )
        for branch in result.stdout.strip().split("\n"):
            branch = branch.strip()
            if branch:
                subprocess.run(["git", "branch", "-D", branch], capture_output=True)
                removed["branches_removed"].append(branch)

    return removed



def save_active_state():
    """Save the active project's current state (call before server shutdown or session end)."""
    current = get_active_project()
    if current:
        if RESULTS_FILE.exists():
            shutil.copy(RESULTS_FILE, _results_path(current))
        if TRAIN_FILE.exists():
            shutil.copy(TRAIN_FILE, _trainpy_path(current))


def _migrate_legacy():
    """Migrate a legacy setup (bare results.tsv) to the project system."""
    name = "default"
    if RESULTS_FILE.exists():
        shutil.copy(RESULTS_FILE, _results_path(name))
    if TRAIN_FILE.exists():
        shutil.copy(TRAIN_FILE, _trainpy_path(name))
    PROJECT_FILE.write_text(name)
