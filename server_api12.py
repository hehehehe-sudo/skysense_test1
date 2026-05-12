# server_api.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import logging
import threading
import json
import asyncio
from typing import Optional
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from server_manager import ServerManager

app = FastAPI(title="Tool Server Manager API")
logger = logging.getLogger(__name__)

# ================= 状态管理 =================
class ServiceState:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"  # idle, starting, running, stopping, failed
        self.manager: Optional[ServerManager] = None
        self.error = ""

state = ServiceState()

def _run_start(config_path: str):
    with state.lock:
        if state.status not in ("idle", "failed"): return
        state.status, state.error = "starting", ""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        state.manager = ServerManager(config)
        state.manager.start_controller()
        state.manager.start_all_workers()
        with state.lock: state.status = "running"
        logger.info("All services started successfully.")
    except Exception as e:
        with state.lock: state.status, state.error = "failed", str(e)
        logger.error(f"Service start failed: {e}")

def _run_stop():
    with state.lock:
        if state.status != "running": return
        state.status = "stopping"
    try:
        if state.manager:
            state.manager.shutdown_services()
        state.manager = None
        with state.lock: state.status = "idle"
        logger.info("All services stopped successfully.")
    except Exception as e:
        with state.lock: state.status, state.error = "failed", str(e)
        logger.error(f"Service stop failed: {e}")

# ================= 生命周期路由 =================
@app.post("/start")
async def start(bg: BackgroundTasks):
    config_path = "./config/all_service_example_local.yaml"
    if not Path(config_path).exists():
        raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")
    bg.add_task(_run_start, config_path)
    return {"status": "starting", "message": "启动任务已提交至后台"}

@app.post("/stop")
async def stop(bg: BackgroundTasks):
    bg.add_task(_run_stop)
    return {"status": "stopping", "message": "停止任务已提交至后台"}

@app.get("/status")
async def get_status(): 
    return {"status": state.status, "error": state.error}

@app.on_event("shutdown")
async def shutdown_event():
    if state.manager and state.status == "running":
        logger.info("Shutting down services due to server termination...")
        try: state.manager.shutdown_services()
        except Exception as e: logger.error(f"Shutdown error: {e}")

# ================= 统一工具调用入口 =================
@app.post("/invoke/{tool_name}")
async def invoke_tool(tool_name: str, request: Request):
    """
    统一调用入口：复用 ServerManager.call_tool 逻辑
    自动路由在线/离线工具，线程安全超时控制，统一错误格式
    """
    if state.status != "running" or not state.manager:
        raise HTTPException(status_code=400, detail="Services not running. Please call /start first.")
        
    # 1. 解析请求体为字典
    try:
        body_bytes = await request.body()
        params = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    # 2. 调用统一的同步方法（放入线程池避免阻塞 FastAPI 事件循环）
    try:
        ret_message = await asyncio.to_thread(state.manager.call_tool, tool_name, params)
    except Exception as e:
        logger.error(f"Tool execution thread failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {e}")

    # 3. 根据 call_tool 返回的 {text, error_code} 构造 HTTP 响应
    if ret_message.get("error_code") == 0:
        return JSONResponse(content=ret_message, status_code=200)
    else:
        error_text = ret_message.get("text", "").lower()
        status_code = 400 if "not found" in error_text or "not ready" in error_text else 500
        return JSONResponse(content=ret_message, status_code=status_code)

# ================= 新增：工具指令查询接口 =================
@app.get("/tools")
async def list_tools():
    """返回所有已注册工具的名称列表"""
    if state.status != "running" or not state.manager:
        raise HTTPException(status_code=400, detail="Services not running. Please call /start first.")
    
    tool_names = list(state.manager.tool_instructions.keys())
    return {"tools": tool_names, "error_code": 0}

@app.get("/tools/{tool_name}")
async def get_tool_instruction(tool_name: str):
    """
    返回指定工具的 instruction（OpenAI function 格式）。
    如果工具未找到，返回 404；如果服务未运行，返回 400。
    """
    if state.status != "running" or not state.manager:
        raise HTTPException(status_code=400, detail="Services not running. Please call /start first.")
    
    result = state.manager.get_tool_instruction(tool_name)
    if result.get("error_code") != 0:
        raise HTTPException(status_code=404, detail=result.get("text", f"Tool instruction for '{tool_name}' not found."))
    return result["instruction"]   # 直接返回 instruction 字典（符合 OpenAI function 格式）
