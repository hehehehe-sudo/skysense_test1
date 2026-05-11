import os
import uuid
import argparse
import requests
from PIL import Image
from io import BytesIO

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
                 smarties_api_url="http://10.202.80.17:8001/segment",
                 limit_model_concurrency=5,
                 host="0.0.0.0",
                 port=None,
                 model_semaphore=None,
                 wait_timeout=120.0,
                 task_timeout=60.0,
                 api_timeout=30.0,
                 save_path="./smarties_outputs",
                 ):
        # ✅ 自定义参数在 super() 前赋值
        self.smarties_api_url = smarties_api_url.rstrip("/")
        self.api_timeout = api_timeout
        self.save_path = save_path

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
            args=None,
        )

    def init_model(self):
        logger.info(f"Initializing {self.model_name} worker (API Mode)...")
        try:
            base_url = self.smarties_api_url.rsplit("/", 1)[0]
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                logger.info("Smarties API health check passed.")
        except Exception as e:
            logger.warning(f"Smarties API health check skipped: {e}")

    def generate(self, params):
        if "image" not in params:
            return {"text": "Missing required parameter: image", "error_code": 2}

        image_input = params["image"]
        payload = {"image": image_input}

        try:
            # 1. 请求 API，获取二进制图像流
            resp = requests.post(
                self.smarties_api_url,
                json=payload,
                headers={"User-Agent": "SmartiesWorker"},
                timeout=self.api_timeout
            )
            resp.raise_for_status()
            
            # 2. 将二进制流直接转换为 PIL Image 对象
            result_img = Image.open(BytesIO(resp.content))

            # 3. 严格参考你提供的保存样式
            if os.path.exists(image_input):
                image_name = os.path.basename(os.path.splitext(image_input)[0])
            else:
                # 兼容 base64 或远程 URL 输入
                image_name = f"smarties_input_{uuid.uuid4().hex[:8]}"
                
            new_filename = f"{image_name}_smarties_seg.png"
            
            if self.save_path and os.path.isdir(self.save_path):
                save_path = os.path.join(self.save_path, new_filename)
            else:
                if self.save_path:
                    logger.warning(f"Save path '{self.save_path}' is not a valid directory. "
                                   f"Falling back to default ./smarties_outputs/")
                save_dir = os.path.join(os.getcwd(), "smarties_outputs")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, new_filename)
            
            result_img.save(save_path)

            # 4. 返回格式完全对齐你的示例
            txt = f"Segmented image saved to {new_filename}"
            return {"text": txt, "image": save_path, "error_code": 0}

        except requests.exceptions.Timeout:
            return {"text": "Smarties API request timed out", "error_code": 4}
        except Exception as e:
            txt_e = f"Error in SmartiesWorker: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "Smarties",
                "description": "Segmentation model that takes an input image and returns a processed/segmented output image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "description": "Local path or base64 string of the input image"}
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
    parser.add_argument("--save-path", type=str, default="./smarties_outputs")
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
        save_path=args.save_path,
        no_register=args.no_register,
    )
    worker.run()
