# 在 BaseToolWorker 类末尾添加
def call_api(self, url: str, payload: dict, method: str = "POST", 
             timeout: float = None, headers: dict = None) -> dict:
    """统一的外部 API 调用封装，子类可直接复用"""
    timeout = timeout or getattr(self, "api_timeout", 30.0)
    headers = headers or {"User-Agent": "BaseToolWorker"}
    
    resp = requests.request(method, url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


    def setup_routes(self):
        # ... 原有路由 ...

        @self.app.post("/worker_health")
        async def worker_health(request: Request):
            """统一健康检查接口，兼容本地模型与 API 工具"""
            # 非 API 工具默认 api_connected 为 True
            api_status = getattr(self, "api_connected", True)
            return JSONResponse({
                "model_name": self.model_name,
                "worker_id": self.worker_id,
                "api_connected": api_status,
                "status": "online"
            })
