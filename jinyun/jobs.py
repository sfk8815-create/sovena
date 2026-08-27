"""jinyun 任务调度：串行作业队列 + 资源守卫。

- 单工作线程串行执行 prepare 作业（OCR 引擎一次性加载/释放，防止超载）
- psutil 内存守卫：可用内存低于阈值时作业保持排队，不启动
- 作业可取消（通过进度回调注入 CancelledError）
- 供 Web UI 与 MCP 服务共享（同进程内单例）
"""
from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import psutil

MEM_GUARD_MB = int(os.environ.get("JINYUN_MEM_GUARD_MB", "12288"))  # 12GB
MAX_LOG = 200


class JobCancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    kind: str                      # prepare / search
    params: dict
    status: str = "queued"         # queued / running / done / error / cancelled / deferred
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    log: list = field(default_factory=list)

    def to_dict(self, with_log: bool = True) -> dict:
        d = {
            "id": self.id,
            "kind": self.kind,
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }
        if with_log:
            d["log"] = self.log[-50:]
        return d


class JobManager:
    """串行作业管理器（线程安全）。"""

    def __init__(self, mem_guard_mb: int = MEM_GUARD_MB):
        self._q: "queue.Queue[Job]" = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._mem_guard_mb = mem_guard_mb
        self._cancel_events: dict[str, threading.Event] = {}
        self._worker = threading.Thread(target=self._run, daemon=True, name="jinyun-worker")
        self._worker.start()

    # ------------------------------------------------------------------

    def submit(self, kind: str, params: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
        with self._lock:
            self._jobs[job.id] = job
        self._q.put(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict(with_log=False) for j in jobs[:100]]

    def active_count(self) -> int:
        """排队/运行/推迟中的作业数。"""
        with self._lock:
            return sum(
                1 for j in self._jobs.values()
                if j.status in ("queued", "running", "deferred")
            )

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            ev = self._cancel_events.get(job_id)
        if job is None:
            return False
        if job.status in ("queued", "deferred", "running"):
            if ev:
                ev.set()
            job.status = "cancelled"
            job.finished_at = time.time()
            return True
        return False

    # ------------------------------------------------------------------

    def _run(self):
        while True:
            job = self._q.get()
            self._wait_memory(job)
            if job.status == "cancelled":
                continue
            self._execute(job)

    def _wait_memory(self, job: Job):
        """内存不足时推迟执行，每 20s 复查一次，直到内存充足或被取消。"""
        while True:
            if job.status == "cancelled":
                return
            avail_mb = psutil.virtual_memory().available // (1024 * 1024)
            if avail_mb >= self._mem_guard_mb:
                if job.status == "deferred":
                    self._log(job, f"内存恢复({avail_mb}MB)，开始执行")
                    job.status = "queued"
                return
            if job.status != "deferred":
                job.status = "deferred"
                self._log(job, f"可用内存 {avail_mb}MB 低于阈值 {self._mem_guard_mb}MB，推迟执行")
            time.sleep(20)

    def _execute(self, job: Job):
        cancel_ev = threading.Event()
        with self._lock:
            self._cancel_events[job.id] = cancel_ev
        job.status = "running"
        job.started_at = time.time()
        self._log(job, f"开始执行 {job.kind} {job.params}")
        try:
            result = self._dispatch(job, cancel_ev)
            job.status = "done"
            job.result = result
        except JobCancelled:
            job.status = "cancelled"
            job.error = "用户取消"
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            self._log(job, job.error)
        finally:
            job.finished_at = time.time()
            with self._lock:
                self._cancel_events.pop(job.id, None)

    def _dispatch(self, job: Job, cancel_ev: threading.Event) -> dict:
        if job.kind == "prepare":
            return self._run_prepare(job, cancel_ev)
        if job.kind == "adhoc":
            return self._run_adhoc(job, cancel_ev)
        raise ValueError(f"未知作业类型: {job.kind}")

    def _progress_cb(self, job: Job, cancel_ev: threading.Event):
        def progress(stage, info):
            if cancel_ev.is_set():
                raise JobCancelled()
            if stage == "convert":
                self._log(job, f"[{info['done']}/{info['total']}] {info.get('route', '')} {info.get('title', '')}")
            elif stage == "ocr":
                self._log(job, f"OCR {info['title']}: {info['page']}/{info['pages']} 页")
            elif stage == "collect":
                self._log(job, f"采集 {info['collection']}: {info['total']} 条")
            elif stage == "done":
                self._log(job, f"完成: {info.get('files', info.get('records'))} 条, "
                               f"索引 {info.get('chunks_indexed')} 块, "
                               f"耗时 {info.get('elapsed_sec')}s")
        return progress

    def _run_prepare(self, job: Job, cancel_ev: threading.Event) -> dict:
        from .pipeline import Pipeline

        pipe = Pipeline()
        params = job.params
        return pipe.prepare(
            params["collection"],
            limit=params.get("limit"),
            use_ocr=params.get("use_ocr", True),
            progress=self._progress_cb(job, cancel_ev),
            rebuild=params.get("rebuild", False),
        )

    def _run_adhoc(self, job: Job, cancel_ev: threading.Event) -> dict:
        from .adhoc import AdhocProcessor
        from .pipeline import Pipeline

        pipe = Pipeline()
        processor = AdhocProcessor(
            pipe.packager, pipe.index, pipe.embedder, pipe.ocr
        )
        params = job.params
        return processor.process(
            params["paths"],
            name=params["name"],
            use_ocr=params.get("use_ocr", True),
            recursive=params.get("recursive", True),
            index=params.get("index", True),
            progress=self._progress_cb(job, cancel_ev),
        )

    def _log(self, job: Job, msg: str):
        with self._lock:
            job.log.append({"t": round(time.time(), 1), "msg": msg})
            if len(job.log) > MAX_LOG:
                del job.log[: len(job.log) - MAX_LOG]


_manager: Optional[JobManager] = None
_manager_lock = threading.Lock()


def get_manager() -> JobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager


def system_status() -> dict:
    vm = psutil.virtual_memory()
    root = os.environ.get("JINYUN_ROOT") or os.path.expanduser("~/jinyun_data")
    disk = None
    try:
        p = root if os.path.isdir(root) else os.path.expanduser("~")
        du = psutil.disk_usage(p)
        disk = {
            "path": p,
            "total_gb": round(du.total / (1024 ** 3), 1),
            "free_gb": round(du.free / (1024 ** 3), 1),
        }
    except OSError:
        pass
    return {
        "mem_total_mb": vm.total // (1024 * 1024),
        "mem_available_mb": vm.available // (1024 * 1024),
        "mem_guard_mb": MEM_GUARD_MB,
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "loadavg": [round(x, 2) for x in psutil.getloadavg()],
        "disk": disk,
        "active_jobs": get_manager().active_count(),
    }
