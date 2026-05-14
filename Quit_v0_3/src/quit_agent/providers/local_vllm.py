from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


@dataclass(slots=True)
class GPUInfo:
    """Detected NVIDIA GPU metadata."""

    index: int
    name: str
    memory_total_mb: int | None = None
    memory_free_mb: int | None = None


@dataclass(slots=True)
class LocalVLLMServerConfig:
    """Configuration for auto-starting a local vLLM OpenAI-compatible server."""

    model_path: str
    served_model_name: str
    host: str = "127.0.0.1"
    port: int = 8000
    gpu_memory_utilization: float | None = None
    gpu_memory_safety_margin: float = 0.10
    tensor_parallel_size: int | str | None = 1
    startup_timeout_seconds: int = 300
    keep_alive: bool = False
    stream: bool | None = None
    extra_body: dict[str, Any] | None = None


@dataclass(slots=True)
class LocalVLLMRuntimeInfo:
    """Runtime details returned after vLLM is ready."""

    base_url: str
    served_model_name: str
    command: list[str]
    gpus: list[GPUInfo]
    selected_gpu_indices: list[int]
    gpu_memory_utilization: float
    tensor_parallel_size: int
    reused_existing_server: bool


def parse_nvidia_smi_csv(output: str) -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        memory_total_mb = None
        memory_free_mb = None
        if len(parts) >= 3:
            try:
                memory_total_mb = int(parts[2].replace("MiB", "").strip())
            except ValueError:
                pass
        if len(parts) >= 4:
            try:
                memory_free_mb = int(parts[3].replace("MiB", "").strip())
            except ValueError:
                pass
        try:
            index = int(parts[0])
        except ValueError:
            continue
        gpus.append(GPUInfo(index=index, name=parts[1], memory_total_mb=memory_total_mb, memory_free_mb=memory_free_mb))
    return gpus


def detect_gpus() -> list[GPUInfo]:
    """Detect local NVIDIA GPUs via nvidia-smi, then torch as fallback."""

    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            gpus = parse_nvidia_smi_csv(result.stdout)
            if gpus:
                return gpus
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    gpus = []
    for index in range(torch.cuda.device_count()):
        memory_free_mb = memory_total_mb = None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            memory_free_mb = int(free_bytes / 1024 / 1024)
            memory_total_mb = int(total_bytes / 1024 / 1024)
        except RuntimeError:
            pass
        gpus.append(GPUInfo(index=index, name=torch.cuda.get_device_name(index), memory_total_mb=memory_total_mb, memory_free_mb=memory_free_mb))
    return gpus


def choose_tensor_parallel_size(gpus: list[GPUInfo], requested: int | str | None) -> int:
    """Resolve tensor parallelism degree.

    Defaults to 1 to avoid hanging when other devices are busy.
    Pass "auto" to use every detected GPU.
    """

    if requested is None:
        return 1
    if isinstance(requested, str):
        if requested.lower() == "auto":
            return max(len(gpus), 1)
        try:
            requested = int(requested)
        except ValueError as exc:
            raise ValueError("tensor_parallel_size must be an integer or 'auto'") from exc
    if requested < 1:
        raise ValueError("tensor_parallel_size must be >= 1")
    return requested


def choose_gpu_memory_utilization(
    requested: float | None,
    gpus: list[GPUInfo] | None = None,
    safety_margin: float = 0.10,
    tensor_parallel_size: int | None = None,
) -> float:
    """Choose a vLLM memory target capped by currently free GPU memory.

    Caps the requested value to (free/total) - safety_margin across selected GPUs
    so that OOM errors are caught before vLLM starts loading the model.
    """

    if not 0.0 <= safety_margin < 1.0:
        raise ValueError("gpu_memory_safety_margin must be in [0.0, 1.0)")
    value = 0.95 if requested is None else requested
    if not 0.0 < value <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0.0, 1.0]")
    selected_gpus = (gpus or [])[:tensor_parallel_size or None]
    free_ratios = [
        gpu.memory_free_mb / gpu.memory_total_mb
        for gpu in selected_gpus
        if gpu.memory_free_mb is not None and gpu.memory_total_mb
    ]
    if free_ratios:
        max_safe = min(free_ratios) - safety_margin
        if max_safe <= 0:
            raise RuntimeError(
                "Not enough free GPU memory to start vLLM after applying safety margin. "
                "Stop other GPU processes or lower gpu_memory_safety_margin."
            )
        value = min(value, max_safe)
    return value


def select_gpus_for_vllm(gpus: list[GPUInfo], tensor_parallel_size: int) -> list[GPUInfo]:
    """Select GPUs by highest free memory so vLLM doesn't default to a busy GPU 0."""

    if not gpus:
        return []
    ranked = sorted(
        gpus,
        key=lambda g: (
            g.memory_free_mb if g.memory_free_mb is not None else -1,
            g.memory_total_mb if g.memory_total_mb is not None else -1,
        ),
        reverse=True,
    )
    return ranked[:tensor_parallel_size]


def resolve_vllm_resources(config: LocalVLLMServerConfig, gpus: list[GPUInfo]) -> tuple[int, float, list[GPUInfo]]:
    tensor_parallel_size = choose_tensor_parallel_size(gpus, config.tensor_parallel_size)
    selected_gpus = select_gpus_for_vllm(gpus, tensor_parallel_size)
    gpu_memory_utilization = choose_gpu_memory_utilization(
        requested=config.gpu_memory_utilization,
        gpus=selected_gpus,
        safety_margin=config.gpu_memory_safety_margin,
        tensor_parallel_size=tensor_parallel_size,
    )
    return tensor_parallel_size, gpu_memory_utilization, selected_gpus


def build_vllm_command(config: LocalVLLMServerConfig, gpus: list[GPUInfo]) -> list[str]:
    tensor_parallel_size, gpu_memory_utilization, _ = resolve_vllm_resources(config, gpus)
    if shutil.which("vllm"):
        command = ["vllm", "serve", config.model_path, "--host", config.host, "--port", str(config.port)]
    else:
        command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", config.model_path, "--host", config.host, "--port", str(config.port)]
    command += ["--served-model-name", config.served_model_name, "--gpu-memory-utilization", str(gpu_memory_utilization)]
    if tensor_parallel_size > 1:
        command += ["--tensor-parallel-size", str(tensor_parallel_size)]
    return command


def build_vllm_env(selected_gpus: list[GPUInfo]) -> dict[str, str]:
    env = os.environ.copy()
    if selected_gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu.index) for gpu in selected_gpus)
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    return env


def ensure_vllm_available() -> None:
    if shutil.which("vllm"):
        return
    if importlib.util.find_spec("vllm") is not None:
        return
    raise RuntimeError(
        "local-vllm requires vLLM. Install with: pip install vllm"
    )


def is_vllm_ready(base_url: str, timeout_seconds: float = 2.0) -> bool:
    try:
        with urlopen(f"{base_url.rstrip('/')}/models", timeout=timeout_seconds) as response:
            return 200 <= response.status < 300
    except URLError:
        return False


class ManagedVLLMServer:
    """Context manager that starts a local vLLM server and optionally stops it on exit."""

    def __init__(self, config: LocalVLLMServerConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[Any] | None = None
        self.runtime_info: LocalVLLMRuntimeInfo | None = None

    def __enter__(self) -> LocalVLLMRuntimeInfo:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self.config.keep_alive:
            self.stop()

    def start(self) -> LocalVLLMRuntimeInfo:
        base_url = f"http://{self.config.host}:{self.config.port}/v1"
        gpus = detect_gpus()
        command = build_vllm_command(self.config, gpus)

        if is_vllm_ready(base_url):
            tensor_parallel_size, gpu_memory_utilization, selected_gpus = resolve_vllm_resources(self.config, gpus)
            self.runtime_info = LocalVLLMRuntimeInfo(
                base_url=base_url,
                served_model_name=self.config.served_model_name,
                command=command,
                gpus=selected_gpus or gpus,
                selected_gpu_indices=[gpu.index for gpu in selected_gpus],
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                reused_existing_server=True,
            )
            return self.runtime_info

        model_path = Path(self.config.model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {model_path}")
        ensure_vllm_available()

        tensor_parallel_size, gpu_memory_utilization, selected_gpus = resolve_vllm_resources(self.config, gpus)
        log_file = tempfile.NamedTemporaryFile(
            mode="w+",
            encoding="utf-8",
            prefix="quit_agent_vllm_",
            suffix=".log",
            delete=False,
        )
        self.process = subprocess.Popen(
            command,
            env=build_vllm_env(selected_gpus),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                log_file.flush()
                gpu_text = ", ".join(
                    f"{g.index}:{g.name} free={g.memory_free_mb}MiB total={g.memory_total_mb}MiB"
                    for g in selected_gpus
                )
                log_tail = _tail_text(Path(log_file.name), max_lines=80)
                raise RuntimeError(
                    f"vLLM server exited early with code {self.process.returncode}. "
                    f"GPUs: {gpu_text or 'none'}. "
                    f"gpu_memory_utilization={gpu_memory_utilization:.3f}, "
                    f"tensor_parallel_size={tensor_parallel_size}. "
                    f"Command: {' '.join(command)}. "
                    f"Log: {log_file.name}\n"
                    f"--- vLLM log tail ---\n{log_tail}"
                )
            if is_vllm_ready(base_url):
                self.runtime_info = LocalVLLMRuntimeInfo(
                    base_url=base_url,
                    served_model_name=self.config.served_model_name,
                    command=command,
                    gpus=selected_gpus or gpus,
                    selected_gpu_indices=[gpu.index for gpu in selected_gpus],
                    gpu_memory_utilization=gpu_memory_utilization,
                    tensor_parallel_size=tensor_parallel_size,
                    reused_existing_server=False,
                )
                return self.runtime_info
            time.sleep(2)
        log_file.flush()
        self.stop()
        raise TimeoutError(
            f"vLLM server did not become ready within {self.config.startup_timeout_seconds}s. "
            f"Command: {' '.join(command)}. Log: {log_file.name}\n"
            f"--- vLLM log tail ---\n{_tail_text(Path(log_file.name), max_lines=80)}"
        )

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), 15)
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), 9)
                self.process.wait(timeout=10)
        except ProcessLookupError:
            pass


def _tail_text(path: Path, *, max_lines: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"<could not read log: {exc}>"
    return "\n".join(lines[-max_lines:]) if lines else "<empty log>"
