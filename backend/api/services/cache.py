import hashlib, json, time
from pathlib import Path
from functools import lru_cache
from collections import OrderedDict

DISK_CACHE = Path("backend/cache")
DISK_CACHE.mkdir(exist_ok=True)

def cif_hash(cif_string: str) -> str:
    """Stable hash for a CIF string — used as cache key."""
    normalized = " ".join(cif_string.split())   # strip whitespace variations
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]

class PredictionCache:
    """
    Two-level cache: in-memory LRU (fast) + disk JSON (persistent across restarts).
    Capacity: 512 structures in memory.
    """
    def __init__(self, maxsize: int = 512):
        self._mem: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, cif: str) -> dict | None:
        key = cif_hash(cif)
        # 1. Memory cache
        if key in self._mem:
            self._mem.move_to_end(key)   # LRU update
            return self._mem[key]
        # 2. Disk cache
        path = DISK_CACHE / f"{key}.json"
        if path.exists():
            with open(path) as f:
                result = json.load(f)
            self._mem[key] = result   # promote to memory
            return result
        return None

    def set(self, cif: str, result: dict):
        key = cif_hash(cif)
        # Evict oldest if full
        if len(self._mem) >= self._maxsize:
            self._mem.popitem(last=False)
        result["cached_at"] = time.time()
        self._mem[key] = result
        # Write to disk
        with open(DISK_CACHE / f"{key}.json", "w") as f:
            json.dump(result, f)

    def stats(self) -> dict:
        return {"memory_entries": len(self._mem),
                "disk_entries":   len(list(DISK_CACHE.glob("*.json")))}
