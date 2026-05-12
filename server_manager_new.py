# server_manager.py
import os
import time
import subprocess
import logging
import copy
from pathlib import Path
import requests
import signal
from typing import Optional, List, Dict, Callable
from box import Box
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# 保持与你原路径一致
try:
    from tool_server.utils.utils import load_json_file, write_json_file
except ImportError:
    def load_json_file(path): 
        import json
        with open(path, 'r') as f: return json.load(f)
    def write_json_file(data, path): 
        import json
        with open(path, 'w') as f: json.dump(data, f, indent=2)

class ServerManager:
    """统一服务管理器：整合进程生命周期管理 + 工具调用路由"""
    def __init__(self, config: Optional[Dict] = None):
        self.config = Box(config)
        self.logger = self._setup_logger()
        self.log_folder = Path(self.config.log_folder)
        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.tools_output_dir = Path(self.config.tools_output_dir)
        self.tools_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.controller_addr = None
        self._clean_environment()
        
        self.base_dir = Path(self.config.base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.controller_config = self.config.controller_config
        self.model_worker_config = self.config.model_worker_config if "model_worker_config" in self.config else []
        self.tool_worker_config = self.config.tool_worker_config if "tool_worker_config" in self.config else []
        self.processes = []

        # ================= 工具管理注册表 (原 base_manager 功能融合) =================
        self.available_offline_tools = set()
        self.available_online_tools = set()
        self.offline_tool_fns: Dict[str, Callable] = {}  # {tool_name: generate_fn}
        self.headers = {"Content-Type": "application/json"}
        # ========================================================================

    def _setup_logger(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _clean_environment(self) -> None:
        os.environ["OMP_NUM_THREADS"] = "1"

    def run_local_command(self, job_name: str, command: List[str], log_file: str, 
                          conda_env: str = None, cuda_visible_devices: str = None) -> subprocess.Popen:
        env = os.environ.copy()
        if cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        
        cmd = []
        if conda_env:
            cmd = ["conda", "run", "--no-capture-output", "-n", conda_env]
        cmd.extend(command)
        
        self.logger.info(f"Starting process: {job_name} with command: {' '.join(cmd)} in cwd: {self.base_dir}")
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=f, env=env, cwd=str(self.base_dir))
            return process

    def wait_for_process(self, process, job_name: str) -> dict:
        self.logger.info(f"Waiting for process to start: {job_name}")
        time.sleep(2)
        if process.poll() is not None:
            self.logger.error(f"Process {job_name} failed to start. Exit code: {process.returncode}")
            raise Exception(f"Process {job_name} failed to start")
        self.logger.info(f"Process {job_name} is running with PID: {process.pid}")
        return {"process": process, "pid": process.pid}

    def wait_for_worker_addr(self, worker_name: str) -> str:
        self.logger.info(f"Waiting for {worker_name} worker...")
        attempt = 0
        while True:
            try:
                attempt += 1
                response = requests.post(
                    f"{self.controller_addr}/get_worker_address",
                    json={"model": worker_name},
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()
                address = response.json().get("address", "")
                if address.strip():
                    self.logger.info(f"Worker {worker_name} is ready at: {address}")
                    return address
                self.logger.warning(f"Attempt {attempt}: worker not ready")
            except Exception as e:
                self.logger.error(f"Attempt {attempt} failed: {e}")
            time.sleep(self.config.retry_interval)
            
    def wait_for_controller_ready(self):
        self.logger.info("Waiting for controller to become ready...")
        for i in range(10):
            try:
                response = requests.post(f"{self.controller_addr}/list_models", timeout=2)
                if response.status_code == 200:
                    self.logger.info("Controller is ready.")
                    return
                else:
                    self.logger.info(f"Controller not ready yet... retrying ({i+1})")
                    time.sleep(2)
            except:
                self.logger.info(f"Controller not ready yet... retrying ({i+1})")
                time.sleep(2)
        raise RuntimeError("Controller failed to become ready.")

    def start_controller(self) -> str:
        ctrl_cfg = copy.deepcopy(self.controller_config)
        log_file = self.log_folder / f"{ctrl_cfg.worker_name}.log"
        
        cmd_dict = dict(ctrl_cfg.cmd)
        script_addr = cmd_dict.pop("script-addr")
        job_name = ctrl_cfg.job_name
        
        command = ["python", script_addr]
        for k, v in cmd_dict.items():
            command.extend([f"--{k}", str(v)])
        
        process = self.run_local_command(
            job_name, command, str(log_file),
            conda_env=ctrl_cfg.get("conda_env", None),
            cuda_visible_devices=ctrl_cfg.get("cuda_visible_devices", None)
        )
        
        self.processes.append({"name": job_name, "process": process})
        self.wait_for_process(process, job_name)
        
        port = self.controller_config.cmd.port
        self.controller_addr = f"http://localhost:{port}"
        self.logger.info(f"Controller is running at: {self.controller_addr}")
        
        controller_addr_dict = {"controller_addr": self.controller_addr}
        if "controller_addr_location" in self.controller_config:
            self.controller_addr_location = self.controller_config.controller_addr_location
        else:
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            self.controller_addr_location = f"{current_file_path}/../../online_workers/controller_addr/controller_addr.json"
            
        controller_addr_dir = os.path.dirname(self.controller_addr_location)
        os.makedirs(controller_addr_dir, exist_ok=True)
        write_json_file(controller_addr_dict, self.controller_addr_location)
        self.logger.info(f"Controller address saved to: {self.controller_addr_location}")
        
        self.wait_for_controller_ready()
        return self.controller_addr

    def start_all_workers(self) -> None:
        self.start_model_worker()
        self.start_tool_worker()

    def start_model_worker(self) -> None:
        for config in self.model_worker_config:
            cfg = list(config.values())[0]
            self.start_worker_by_config(cfg)
    
    def start_tool_worker(self) -> None:
        for config in self.tool_worker_config:
            cfg = list(config.values())[0]
            self.start_worker_by_config(cfg)
    
    def start_worker_by_config(self, config) -> None:
        worker_cfg = copy.deepcopy(config)
        log_file = self.log_folder / f"{worker_cfg.worker_name}_worker.log"
        
        cmd_dict = dict(worker_cfg.cmd)
        script_addr = cmd_dict.pop("script-addr")
        job_name = worker_cfg.job_name
        
        command = [
            "python", script_addr,
            "--controller-address", self.controller_addr,
        ]
        for k, v in cmd_dict.items():
            command.extend([f"--{k}", str(v)])
        
        process = self.run_local_command(
            job_name, command, str(log_file),
            conda_env=worker_cfg.get("conda_env", None),
            cuda_visible_devices=worker_cfg.get("cuda_visible_devices", None)
        )
        
        self.processes.append({"name": job_name, "process": process})
        self.wait_for_process(process, job_name)
        
        if worker_cfg.get("wait_for_self", False):
            self.wait_for_worker_addr(worker_cfg.worker_name)

    def shutdown_services(self) -> None:
        try:
            if hasattr(self, 'controller_addr_location') and os.path.exists(self.controller_addr_location):
                os.remove(self.controller_addr_location)
                self.logger.info("Controller address file removed")
            
            for proc_info in self.processes:
                process = proc_info["process"]
                name = proc_info["name"]
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                        self.logger.info(f"Process {name} (PID: {process.pid}) terminated successfully")
                    except subprocess.TimeoutExpired:
                        process.kill()
                        self.logger.warning(f"Process {name} (PID: {process.pid}) killed forcefully")
                else:
                    self.logger.info(f"Process {name} already finished with exit code {process.returncode}")
            
            self.processes.clear()
            self.logger.info("All services have been shutdown")
        except Exception as e:
            self.logger.error(f"Critical error during shutdown: {e}")
            raise

    # ================= 原 base_manager 工具调用功能融合 =================
    def register_offline_tool(self, tool_name: str, generate_fn: Callable):
        """注册离线工具函数"""
        self.available_offline_tools.add(tool_name)
        self.offline_tool_fns[tool_name] = generate_fn
        self.logger.info(f"Registered offline tool: {tool_name}")

    def register_online_tool(self, tool_name: str):
        """标记为在线工具（实际地址由 Controller 动态发现）"""
        self.available_online_tools.add(tool_name)
        self.logger.info(f"Registered online tool: {tool_name}")

    def call_tool(self, tool_name: str, params: dict) -> dict:
        """统一工具调用入口：兼容离线函数与在线服务，线程安全超时控制"""
        # 1. 动态超时策略（与原逻辑一致）
        if tool_name in ["AddPoisLayer", "ComputeDistance"]:
            timeout_sec = 180
        elif tool_name in ["AddIndexLayer"]:
            timeout_sec = 300
        elif tool_name in ["ChangeDetection", "GetAreaBoundary"]:
            timeout_sec = 120
        else:
            timeout_sec = 60

        ret_message = {"text": f"Failed to call tool {tool_name} for unknown reason", "error_code": 1}

        def _execute():
            # 优先尝试在线工具（通过 Controller 动态路由）
            if self.controller_addr and (tool_name in self.available_online_tools or tool_name not in self.available_offline_tools):
                try:
                    resp = requests.post(
                        f"{self.controller_addr}/get_worker_address",
                        json={"model": tool_name},
                        timeout=5
                    )
                    resp.raise_for_status()
                    worker_addr = resp.json().get("address")
                    if worker_addr:
                        target_url = f"{worker_addr.rstrip('/')}/worker_generate"
                        # 如需禁用代理可在此添加：with self.disable_proxy():
                        ret = requests.post(target_url, json=params, headers=self.headers, timeout=timeout_sec)
                        ret.raise_for_status()
                        return ret.json()
                except Exception as e:
                    self.logger.debug(f"Online tool query/execution failed for {tool_name}: {e}")

            # 回退到离线工具
            if tool_name in self.available_offline_tools:
                try:
                    gen_fn = self.offline_tool_fns.get(tool_name)
                    if gen_fn:
                        return gen_fn(params)
                    return {"text": f"Offline tool generator not registered for {tool_name}", "error_code": 1}
                except Exception as e:
                    return {"text": f"Failed to call offline tool {tool_name}: {e}", "error_code": 1}

            return {"text": f"Tool {tool_name} not found or not ready.", "error_code": 1}

        # 2. 线程安全超时控制（替代 signal.alarm，完美适配 FastAPI）
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_execute)
            try:
                ret_message = future.result(timeout=timeout_sec)
            except FuturesTimeoutError:
                ret_message = {"text": f"Timeout calling tool {tool_name} after {timeout_sec}s", "error_code": 1}
            except Exception as e:
                self.logger.error(f"Tool execution thread error: {e}")
                ret_message = {"text": f"Failed to call tool {tool_name}: {e}", "error_code": 1}

        return ret_message
