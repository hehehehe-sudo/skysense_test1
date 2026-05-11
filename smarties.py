import os
import uuid
import argparse
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import requests

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"SmartiesWorker_{worker_id}.log")

class SmartiesWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="Smarties",
                 device="cpu",
                 smarties_api_url="http://10.202.80.17:8001/segment",  # ⚠️ 替换为实际 API 地址
                 limit_model_concurrency=5,
                 host="0.0.0.0",
                 port=None,
                 model_semaphore=None,
                 wait_timeout=120.0,
                 task_timeout=60.0,
                 api_timeout=30.0,
                 output_dir="smarties_segmentation_outputs",
                 **kwargs
                 ):
        self.smarties_api_url = smarties_api_url.rstrip("/")
        self.api_timeout = api_timeout
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        super().__init__(
            controller_addr=controller_addr,
            worker_addr=worker_addr,
            worker_id=worker_id,
            no_register=no_register,
            model_path=None,
            model_base=None,
            model_name=model_name,
            load_8bit=False,
            load_4bit=False,
            device=device,
            limit_model_concurrency=limit_model_concurrency,
            host=host,
            port=port,
            model_semaphore=model_semaphore,
            wait_timeout=wait_timeout,
            task_timeout=task_timeout,
            **kwargs
        )

    def init_model(self):
        logger.info(f"Initializing {self.model_name} worker (API Mode)...")
        # 轻量连通性检查
        try:
            base_url = self.smarties_api_url.rsplit("/", 1)[0]
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                logger.info("Smarties API health check passed.")
        except Exception as e:
            logger.warning(f"Smarties API health check skipped: {e}")

    def generate(self, params):
        required_keys = ("image",)
        missing = [k for k in required_keys if k not in params]
        if missing:
            return {"text": f"Missing required parameter(s): {', '.join(missing)}", "error_code": 2}

        image = params["image"]
        text = params.get("text", "segment")  # 默认提示词，按需覆盖
        
        # ✅ 严格按要求的 payload 格式
        payload = {"image": image, "text": text}

        try:
            # 复用 BaseToolWorker 已封装的 call_api
            result = self.call_api(
                url=self.smarties_api_url,
                payload=payload,
                method="POST",
                headers={"User-Agent": "SmartiesWorker"},
                timeout=self.api_timeout
            )

            # 解析并保存分割掩码
            mask_paths = self._parse_and_save_masks(result)
            output_text = ", ".join(mask_paths) if mask_paths else "No valid segments generated"
            return {"text": output_text, "error_code": 0}

        except requests.exceptions.Timeout:
            return {"text": "Smarties API request timed out", "error_code": 4}
        except Exception as e:
            logger.error(f"Smarties API call failed: {e}")
            return {"text": f"Error calling Smarties API: {str(e)}", "error_code": 4}

    def _parse_and_save_masks(self, result: dict) -> list:
        """
        兼容解析 Smarties API 返回的掩码数据（base64 / numpy array / 文件路径）
        并统一保存到本地 output_dir，返回文件路径列表。
        """
        masks_data = result.get("masks", result.get("results", result.get("segments", [])))
        if not masks_data:
            logger.warning(f"No masks found in Smarties response: {result}")
            return []

        saved_paths = []
        for i, mask_item in enumerate(masks_data):
            try:
                mask_path = os.path.join(self.output_dir, f"smarties_mask_{uuid.uuid4().hex[:8]}_{i}.png")
                
                if isinstance(mask_item, dict):
                    if "base64" in mask_item:
                        img_bytes = base64.b64decode(mask_item["base64"])
                        Image.open(BytesIO(img_bytes)).save(mask_path)
                    elif "array" in mask_item or "mask" in mask_item:
                        arr = np.array(mask_item.get("array", mask_item.get("mask")))
                        if arr.max() <= 1.0:
                            arr = (arr * 255).astype(np.uint8)
                        Image.fromarray(arr.astype(np.uint8), mode="L").save(mask_path)
                elif isinstance(mask_item, (list, np.ndarray)):
                    arr = np.array(mask_item)
                    if arr.max() <= 1.0:
                        arr = (arr * 255).astype(np.uint8)
                    Image.fromarray(arr.astype(np.uint8), mode="L").save(mask_path)
                elif isinstance(mask_item, str) and os.path.exists(mask_item):
                    # API 直接返回了已有路径，软链接或直接记录
                    saved_paths.append(mask_item)
                    continue
                else:
                    continue
                    
                saved_paths.append(mask_path)
            except Exception as e:
                logger.warning(f"Failed to parse/save mask {i}: {e}")
                continue

        return saved_paths

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "Smarties",
                "description": "Perform image segmentation using the Smarties API. Returns local paths to saved mask PNG files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "description": "Image path or base64 string"},
                        "text": {"type": "string", "description": "Segmentation prompt or target object description (optional)"}
                    },
                    "required": ["image"]
                },
            },
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20007)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="Smarties")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--smarties-api-url", type=str, default="http://10.202.80.17:8001/segment")
    parser.add_argument("--api-timeout", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="smarties_segmentation_outputs")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    logger.info(f"Starting SmartiesWorker with args: {args}")
    worker = SmartiesWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        model_name=args.model_name,
        device=args.device,
        smarties_api_url=args.smarties_api_url,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        api_timeout=args.api_timeout,
        output_dir=args.output_dir,
        no_register=args.no_register,
    )
    worker.run()
