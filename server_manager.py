import os
import time
import subprocess
import logging
import copy
from pathlib import Path
import requests
import signal
from typing import Optional, List, Dict
from box import Box
import argparse
import yaml

# 保持与你原路径一致
try:
    from tool_server.utils.utils import load_json_file, write_json_file
except ImportError:
    # 兼容本地测试直接运行的情况
    def load_json_file(path): 
        import json
        with open(path, 'r') as f: return json.load(f)
    def write_json_file(data, path): 
        import json
        with open(path, 'w') as f: json.dump(data, f, indent=2)

class ServerManager:
    """Server Manager Class for local process management (API-Safe Version)"""
    def __init__(self, config: Optional[Dict] = None):
        # Initialize configuration
        self.config = Box(config)
        self.logger = self._setup_logger()
        self.log_folder = Path(self.config.log_folder)
        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.tools_output_dir = Path(self.config.tools_output_dir)
        self.tools_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize status
        self.controller_addr = None
        self._clean_environment()
        
        # ✅ 关键修改 1：移除 os.chdir()，改为保存绝对路径，避免污染全局进程状态
        self.base_dir = Path(self.config.base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.controller_config = self.config.controller_config
        self.model_worker_config = self.config.model_worker_config if "model_worker_config" in self.config else []
        self.tool_worker_config = self.config.tool_worker_config if "tool_worker_config" in self.config else []
        self.processes = []  # Track all started processes

    def _setup_logger(self) -> logging.Logger:
        """Set up logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _clean_environment(self) -> None:
        """Clean environment variables"""
        os.environ["OMP_NUM_THREADS"] = "1"

    def run_local_command(self, job_name: str, command: List[str], log_file: str, 
                          conda_env: str = None, cuda_visible_devices: str = None) -> subprocess.Popen:
        """Run command locally"""
        env = os.environ.copy()
        if cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        
        cmd = []
        if conda_env:
            cmd = ["conda", "run", "--no-capture-output", "-n", conda_env]
        cmd.extend(command)
        
        self.logger.info(f"Starting process: {job_name} with command: {' '.join(cmd)} in cwd: {self.base_dir}")
        with open(log_file, 'w') as f:
            # ✅ 关键修改 2：使用 cwd 参数指定子进程工作目录，替代全局 os.chdir()
            process = subprocess.Popen(cmd, stdout=f, stderr=f, env=env, cwd=str(self.base_dir))
            return process

    def wait_for_process(self, process, job_name: str) -> dict:
        """Wait for process to initialize"""
        self.logger.info(f"Waiting for process to start: {job_name}")
        time.sleep(2)
        if process.poll() is not None:
            self.logger.error(f"Process {job_name} failed to start. Exit code: {process.returncode}")
            raise Exception(f"Process {job_name} failed to start")
        self.logger.info(f"Process {job_name} is running with PID: {process.pid}")
        return {"process": process, "pid": process.pid}

    def wait_for_worker_addr(self, worker_name: str) -> str:
        """Wait for worker address to be available"""
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
        """Start controller"""
        # ✅ 关键修改 3：深拷贝配置避免 pop() 污染原始 config，保证 API 可重复调用
        ctrl_cfg = copy.deepcopy(self.controller_config)
        log_file = self.log_folder / f"{ctrl_cfg.worker_name}.log"
        
        # 安全提取 script-addr
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
        """Start all worker services"""
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
        """Start specific worker"""
        # ✅ 同样使用深拷贝保护原始配置
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
        """Shut down all local processes"""
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
