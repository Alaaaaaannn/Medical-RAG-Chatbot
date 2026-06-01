import threading
import time
from collections import OrderedDict


class DailyBudget:
    def __init__(self, max_per_day: int):
        self.max_per_day = max_per_day
        self._day = -1
        self._count = 0
        self._lock = threading.Lock()

    @staticmethod
    def _today() -> int:
        return int(time.time() // 86400)

    def allow(self) -> bool:
        with self._lock:
            day = self._today()
            if day != self._day:
                self._day, self._count = day, 0
            if self._count >= self.max_per_day:
                return False
            self._count += 1
            return True

    def remaining(self) -> int:
        with self._lock:
            if self._today() != self._day:
                return self.max_per_day
            return max(0, self.max_per_day - self._count)


class SessionStore:
    def __init__(self, max_sessions: int = 1000, ttl_seconds: int = 3600):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._store: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()

    def _evict(self, now: float) -> None:
        cutoff = now - self.ttl_seconds
        while self._store:
            sid, entry = next(iter(self._store.items()))
            if entry["ts"] < cutoff:
                del self._store[sid]
            else:
                break
        while len(self._store) > self.max_sessions:
            self._store.popitem(last=False)

    def get(self, sid: str) -> list:
        now = time.time()
        with self._lock:
            self._evict(now)
            entry = self._store.get(sid)
            if entry is None:
                entry = {"messages": [], "ts": now}
                self._store[sid] = entry
            entry["ts"] = now
            self._store.move_to_end(sid)
            return entry["messages"]

    def reset(self, sid: str) -> None:
        with self._lock:
            if sid in self._store:
                self._store[sid]["messages"] = []
