# server_api.py
import sys
from pathlib import Path
# 确保能正确导入同级目录的 ServerManager
sys.path.insert(0, str(Path(__file__).parent))

import os
import logging
import threading
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from server_manager import ServerManager  # 确保类名与文件名一致

app = FastAPI(title="Tool Server Manager API")
logger = logging.getLogger(__name__)

# ================= 状态管理 =================
class ServiceState:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"  # idle, starting, running, stopping, failed
        self.manager = None
        self.error = ""

state = ServiceState()

def _run_start(config_path: str):
    with state.lock:
        if state.status not in ("idle", "failed"): return
        state.status, state.error = "starting", ""
    try:
        with open(config_path, "r") as f: config = yaml.safe_load(f)
        state.manager = ServerManager(config)
        state.manager.start_controller()
        state.manager.start_all_workers()
        with state.lock: state.status = "running"
    except Exception as e:
        with state.lock: state.status, state.error = "failed", str(e)

def _run_stop():
    with state.lock:
        if state.status != "running": return
        state.status = "stopping"
    try:
        if state.manager: state.manager.shutdown_services()
        state.manager = None
        with state.lock: state.status = "idle"
    except Exception as e:
        with state.lock: state.status, state.error = "failed", str(e)

@app.post("/start")
async def start(bg: BackgroundTasks):
    bg.add_task(_run_start, "./config/all_service_example_local.yaml")
    return {"status": "starting", "message": "启动任务已提交"}

@app.post("/stop")
async def stop(bg: BackgroundTasks):
    bg.add_task(_run_stop)
    return {"status": "stopping", "message": "停止任务已提交"}

@app.get("/status")
async def status(): return {"status": state.status, "error": state.error}

@app.on_event("shutdown")
async def shutdown():
    if state.manager and state.status == "running":
        try: state.manager.shutdown_services()
        except: pass
