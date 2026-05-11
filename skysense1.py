import os
import uuid
import argparse
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import requests

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"SkySenseWorker_{worker_id}.log")

# COCO 80 类映射（供 detect 解析备用）
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
    8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv',
    63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class SkySenseWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="SkySense",
                 device="cuda",
                 limit_model_concurrency=5,
                 host="0.0.0.0",
                 port=None,
                 model_semaphore=None,
                 wait_timeout=120.0,
                 task_timeout=60.0,
                 **kwargs
                 ):
        # ✅ 硬编码 API 地址
        self.skysense_api_url = "http://10.202.80.17:8000/predict"
        
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

    # def init_model(self):
    #     logger.info(f"Initializing {self.model_name} worker (API Mode)...")
    #     # 轻量连通性检查（不阻塞启动）
    #     try:
    #         base_url = self.skysense_api_url.rsplit("/", 1)[0]
    #         resp = requests.get(f"{base_url}/health", timeout=5)
    #         if resp.status_code == 200:
    #             logger.info("SkySense API health check passed.")
    #     except Exception as e:
    #         logger.warning(f"SkySense API health check skipped: {e}")
    def init_model(self):
    logger.info(f"Initializing {self.model_name} worker (API Mode)...")
    try:
        # 严格探测外部 API
        base_url = self.smarties_api_url.rsplit("/", 1)[0]
        resp = requests.get(f"{base_url}/health", timeout=5)
        resp.raise_for_status()
        self.api_connected = True  # ✅ 标记连通
        logger.info("Smarties API health check passed.")
    except Exception as e:
        self.api_connected = False # ❌ 标记断开
        logger.error(f"Smarties API health check FAILED: {e}")

    def generate(self, params):
        required_keys = ("image", "text")
        missing = [k for k in required_keys if k not in params]
        if missing:
            return {"text": f"Missing required parameter(s): {', '.join(missing)}", "error_code": 2}

        image = params["image"]
        text = params["text"]
        task = params.get("task", "detect").lower()

        # ✅ 严格按要求的 payload 格式
        payload = {"image": image, "text": text}

        try:
            # 调用基类已封装的 call_api
            result = self.call_api(
                url=self.skysense_api_url,
                payload=payload,
                method="POST",
                headers={"User-Agent": "SkySenseWorker"},
                timeout=60
            )

            # 根据 task 路由解析逻辑
            if task == "detect":
                parsed_text = self._parse_detect_output(result)
            elif task == "segment":
                parsed_text = self._parse_segment_output(result)
            else:
                # 未知 task 直接返回原始 JSON
                parsed_text = json.dumps(result, ensure_ascii=False)

            return {"text": parsed_text, "error_code": 0}

        except Exception as e:
            logger.error(f"SkySense API call failed: {e}")
            return {"text": f"Error calling SkySense: {str(e)}", "error_code": 4}

    def _parse_detect_output(self, result):
        """解析 detect 输出: bbox + 置信度 + COCO 类别"""
        # 🔍 尝试常见返回结构
        detections = result.get("detections", result.get("results", result.get("boxes", [])))
        if not detections:
            return json.dumps(result, ensure_ascii=False)

        lines = []
        for det in detections:
            box = det.get("box", det.get("bbox", det.get("bboxes", [0,0,0,0])))
            conf = det.get("score", det.get("confidence", det.get("conf", 0.0)))
            cls_id = det.get("class_id", det.get("label", det.get("category_id", None)))
            
            # 转换类别 ID -> 名称
            cls_name = COCO_CLASSES.get(int(cls_id), f"class_{cls_id}") if cls_id is not None else "unknown"

            if isinstance(box, (list, tuple)) and len(box) == 4:
                x1, y1, x2, y2 = box
                lines.append(f"({int(x1)},{int(y1)},{int(x2)},{int(y2)}),{float(conf):.3f},{cls_name}")
            elif isinstance(box, str):
                lines.append(f"{box},{float(conf):.3f},{cls_name}")

        return "\n".join(lines) if lines else json.dumps(result, ensure_ascii=False)

    def _parse_segment_output(self, result):
        """解析 segment 输出: 保存 mask 图层 + 元数据"""
        masks_data = result.get("masks", result.get("segments", result.get("results", [])))
        if not masks_data:
            return json.dumps(result, ensure_ascii=False)

        output_parts = []
        # 临时保存目录（可根据需要改为绝对路径或云存储）
        mask_dir = os.path.join(os.getcwd(), "skysense_masks")
        os.makedirs(mask_dir, exist_ok=True)

        for i, mask_item in enumerate(masks_data):
            mask_path = os.path.join(mask_dir, f"mask_{i}.png")
            meta = ""
            
            try:
                if isinstance(mask_item, dict):
                    # 支持 base64 或 numpy array
                    if "base64" in mask_item:
                        img_bytes = base64.b64decode(mask_item["base64"])
                        Image.open(BytesIO(img_bytes)).save(mask_path)
                    elif "array" in mask_item or "mask" in mask_item:
                        arr = np.array(mask_item.get("array", mask_item.get("mask")))
                        if arr.dtype != np.uint8:
                            arr = (arr * 255).astype(np.uint8)
                        Image.fromarray(arr, mode="L").save(mask_path)
                    
                    meta = f"box={mask_item.get('box','')}, conf={mask_item.get('score','')}, cls={mask_item.get('class','')}"
                    
                elif isinstance(mask_item, (list, np.ndarray)):
                    arr = np.array(mask_item)
                    if arr.dtype != np.uint8:
                        arr = (arr * 255).astype(np.uint8)
                    Image.fromarray(arr, mode="L").save(mask_path)
                else:
                    continue
                    
                output_parts.append(f"mask_{i}: {mask_path} {meta}".strip())
            except Exception as e:
                logger.warning(f"Failed to parse mask {i}: {e}")
                continue

        return "\n".join(output_parts) if output_parts else json.dumps(result, ensure_ascii=False)

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "SkySense",
                "description": "Analyze sky/remote sensing images using SkySense model. Supports object detection (bbox+conf) and segmentation (mask layers).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "description": "Image path or base64 string"},
                        "text": {"type": "string", "description": "Prompt or object description to guide the model"},
                        "task": {
                            "type": "string",
                            "enum": ["detect", "segment"],
                            "description": "Task type: 'detect' returns bounding boxes & confidence; 'segment' returns saved mask layers. Default: 'detect'"
                        }
                    },
                    "required": ["image", "text"]
                },
            },
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20006)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="SkySense")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    logger.info(f"Starting SkySenseWorker with args: {args}")
    worker = SkySenseWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        model_name=args.model_name,
        device=args.device,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        no_register=args.no_register,
    )
    worker.run()
