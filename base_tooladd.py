# 在 BaseToolWorker 类末尾添加
def call_api(self, url: str, payload: dict, method: str = "POST", 
             timeout: float = None, headers: dict = None) -> dict:
    """统一的外部 API 调用封装，子类可直接复用"""
    timeout = timeout or getattr(self, "api_timeout", 30.0)
    headers = headers or {"User-Agent": "BaseToolWorker"}
    
    resp = requests.request(method, url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
