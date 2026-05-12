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

# server_api.py 末尾追加
import requests
from fastapi import Request, HTTPException

@app.post("/invoke/{tool_name}")
async def invoke_tool(tool_name: str, request: Request):
    """统一调用入口：自动查询控制器地址并转发请求"""
    if state.status != "running" or not state.manager:
        raise HTTPException(status_code=400, detail="Services not running")
        
    if not state.manager.controller_addr:
        raise HTTPException(status_code=503, detail="Controller address not available")
        
    # 1. 向控制器查询该 worker 的实际地址
    try:
        resp = requests.post(
            f"{state.manager.controller_addr}/get_worker_address",
            json={"model": tool_name},
            timeout=5
        )
        resp.raise_for_status()
        worker_addr = resp.json().get("address")
        if not worker_addr:
            raise HTTPException(status_code=404, detail=f"Worker '{tool_name}' not registered")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Controller query failed: {e}")

    # 2. 转发客户端请求到对应 worker
    try:
        body = await request.body()
        headers = dict(request.headers)
        # 移除代理不应转发的 hop-by-hop 头
        for h in ["host", "content-length", "transfer-encoding", "connection"]:
            headers.pop(h, None)
            
        worker_resp = requests.post(
            worker_addr,
            data=body,
            headers=headers,
            timeout=60  # 模型推理通常耗时较长
        )
        return worker_resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Worker '{tool_name}' call failed: {e}")
