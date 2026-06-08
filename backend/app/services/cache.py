import time
import hashlib
from typing import Dict, Any, Optional

class SimpleTTLCache:
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def _make_key(self, query: str) -> str:
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict]:
        key = self._make_key(query)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["data"]
            else:
                del self.cache[key]
        return None
    
    def set(self, query: str, data: Dict):
        key = self._make_key(query)
        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }
    
    def clear(self):
        self.cache.clear()