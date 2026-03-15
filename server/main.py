"""FastAPI server — REST + WebSocket for autoresearch UI."""

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.process_manager import ProcessManager
from server.git_watcher import GitWatcher
from server.hardware import detect_hardware
from server.program_generator import generate_program_md
from server import project_manager


# ─── WebSocket Connection Manager ───────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                pass


manager = ConnectionManager()
process_mgr = ProcessManager(manager)
git_watcher = GitWatcher(manager)


# ─── App Lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    watcher_task = asyncio.create_task(git_watcher.watch())
    yield
    project_manager.save_active_state()
    watcher_task.cancel()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── WebSocket ───────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ─── REST: Health & Hardware ─────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/hardware")
async def get_hardware():
    return detect_hardware()


# ─── REST: Projects ─────────────────────────────────────────────────────────

@app.get("/api/projects")
async def list_projects():
    return project_manager.list_projects()


class CreateProjectConfig(BaseModel):
    name: str
    forkFrom: str | None = None


@app.post("/api/projects")
async def create_project(config: CreateProjectConfig):
    return project_manager.create_project(config.name, config.forkFrom)


@app.post("/api/projects/{name}/activate")
async def activate_project(name: str):
    result = project_manager.activate_project(name)
    # Reload git_watcher with new results.tsv
    git_watcher.reload()
    return result


class DeleteProjectConfig(BaseModel):
    pruneBranches: bool = False


@app.delete("/api/projects/{name}")
async def delete_project(name: str, config: DeleteProjectConfig = DeleteProjectConfig()):
    return project_manager.delete_project(name, config.pruneBranches)


# ─── REST: Sessions ─────────────────────────────────────────────────────────

class SessionConfig(BaseModel):
    focusAreas: list[str] = []
    hints: str = ""
    agentCommand: str = "claude"
    branchName: str = "session"
    maxExperiments: int = 15


@app.post("/api/session/start")
async def start_session(config: SessionConfig):
    hardware = detect_hardware()
    active_project = project_manager.get_active_project() or "default"

    program_md = generate_program_md(
        focus_areas=config.focusAreas,
        hints=config.hints,
        hardware=hardware,
        max_experiments=config.maxExperiments,
    )
    with open("program.md", "w") as f:
        f.write(program_md)

    branch = f"{active_project}/{config.branchName}-{int(time.time())}"
    await process_mgr.start_session(
        agent_command=config.agentCommand,
        branch_name=branch,
    )
    return {"status": "started", "branch": branch, "project": active_project}


@app.post("/api/session/stop")
async def stop_session():
    await process_mgr.stop_session()
    project_manager.save_active_state()
    return {"status": "stopped"}


@app.get("/api/session/status")
async def session_status():
    status = process_mgr.get_status()
    status["project"] = project_manager.get_active_project()
    return status


# ─── REST: Experiments ───────────────────────────────────────────────────────

@app.get("/api/experiments")
async def get_experiments():
    return git_watcher.get_all_experiments()


@app.get("/api/experiments/{commit}")
async def get_experiment(commit: str):
    return git_watcher.get_experiment_with_diff(commit)


@app.get("/api/results/summary")
async def get_summary():
    from server.summarizer import generate_summary
    experiments = git_watcher.get_all_experiments()
    return await generate_summary(experiments)
