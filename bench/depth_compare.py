#!/usr/bin/env python3
"""Repeatable ONNX depth inference comparison for RapidRAW model candidates.

Dependencies: numpy, Pillow, onnxruntime, psutil.
Optional: pynvml for approximate per-process NVIDIA VRAM sampling.

This is a research benchmark, not the production RapidRAW preprocessing path.
It explicitly models the current DA2 square/pad adapter and a DA3
aspect-preserving, multiple-of-14 adapter. Validate against the exact export
before using its results as a release gate.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


class MemorySampler:
    def __init__(self, gpu_index: int = 0) -> None:
        self.process = psutil.Process(os.getpid())
        self.gpu_index = gpu_index
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.peak_rss = 0
        self.peak_vram: int | None = None
        self._nvml = None
        self._device = None
        self._pid = os.getpid()

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._device = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception:
            self._nvml = None
            self._device = None

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            except (psutil.Error, OSError):
                pass
            if self._nvml is not None and self._device is not None:
                try:
                    funcs = [
                        getattr(self._nvml, "nvmlDeviceGetComputeRunningProcesses", None),
                        getattr(self._nvml, "nvmlDeviceGetGraphicsRunningProcesses", None),
                    ]
                    for fn in funcs:
                        if fn is None:
                            continue
                        for proc in fn(self._device):
                            if proc.pid == self._pid and proc.usedGpuMemory:
                                current = int(proc.usedGpuMemory)
                                self.peak_vram = max(self.peak_vram or 0, current)
                except Exception:
                    pass
            self.stop_event.wait(0.02)

    def __enter__(self) -> "MemorySampler":
        try:
            self.peak_rss = self.process.memory_info().rss
        except (psutil.Error, OSError):
            self.peak_rss = 0
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


def contain_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    new_w = max(1, round(image.width * scale))
    new_h = max(1, round(image.height * scale))
    # Pillow interpolation differs slightly from image::FilterType::Triangle.
    resample = Image.Resampling.BILINEAR
    return image.resize((new_w, new_h), resample=resample)


def preprocess_v2(image: Image.Image, size: int = 518) -> np.ndarray:
    """Approximate current code: resize within square, ImageNet normalize, zero-pad."""
    resized = contain_resize(image, size, size)
    arr = np.asarray(resized.convert("RGB"), dtype=np.float32) / 255.0
    normalized = (arr - MEAN.transpose(1, 2, 0)) / STD.transpose(1, 2, 0)
    h, w, _ = normalized.shape
    out = np.zeros((3, size, size), dtype=np.float32)
    top = (size - h) // 2
    left = (size - w) // 2
    out[:, top : top + h, left : left + w] = normalized.transpose(2, 0, 1)
    return out[None, ...]


def nearest_multiple(value: int, multiple: int = 14) -> int:
    lower = max(multiple, (value // multiple) * multiple)
    upper = lower + multiple
    return upper if abs(upper - value) <= abs(value - lower) else lower


def preprocess_da3(image: Image.Image, process_res: int = 504) -> np.ndarray:
    """DA3 reference-style upper-bound resize and nearest multiple of patch size 14."""
    scale = process_res / float(max(image.width, image.height))
    new_w = max(1, round(image.width * scale))
    new_h = max(1, round(image.height * scale))
    # Reference InputProcessor uses cubic for upscaling and area for downscaling.
    resample = Image.Resampling.BICUBIC if scale > 1.0 else Image.Resampling.BOX
    resized = image.resize((new_w, new_h), resample=resample)
    new_w = nearest_multiple(resized.width)
    new_h = nearest_multiple(resized.height)
    if (new_w, new_h) != resized.size:
        resample = Image.Resampling.BICUBIC if new_w > resized.width or new_h > resized.height else Image.Resampling.BOX
        resized = resized.resize((new_w, new_h), resample=resample)
    arr = np.asarray(resized.convert("RGB"), dtype=np.float32) / 255.0
    normalized = (arr - MEAN.transpose(1, 2, 0)) / STD.transpose(1, 2, 0)
    return normalized.transpose(2, 0, 1)[None, ...].astype(np.float32, copy=False)


def tensor_to_2d(output: np.ndarray, model: Path) -> np.ndarray:
    depth = np.asarray(output, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"{model}: expected one 2D depth map after squeeze; got shape {output.shape}")
    if not np.isfinite(depth).any():
        raise ValueError(f"{model}: model output has no finite values")
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def save_map(depth: np.ndarray, path: Path, invert: bool = False) -> None:
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError(f"Cannot save a depth map without finite values: {path}")
    values = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    lo, hi = np.percentile(values[finite], [0.5, 99.5])
    if hi <= lo:
        vis = np.zeros(values.shape, dtype=np.uint8)
    else:
        vis = np.clip((values - lo) / (hi - lo), 0, 1)
        if invert:
            vis = 1.0 - vis
        vis = (vis * 255.0).round().astype(np.uint8)
    Image.fromarray(vis, mode="L").save(path)


def image_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "medianMs": round(statistics.median(values), 3),
        "p95Ms": round(float(np.percentile(values, 95)), 3),
        "meanMs": round(statistics.mean(values), 3),
        "stdevMs": round(statistics.pstdev(values), 3),
        "minMs": round(min(values), 3),
        "maxMs": round(max(values), 3),
    }


def validate_input_shape(session: ort.InferenceSession, tensor: np.ndarray, model_path: Path) -> None:
    input_meta = session.get_inputs()[0]
    expected = input_meta.shape
    if len(expected) != tensor.ndim:
        raise ValueError(f"{model_path}: graph expects input shape {expected}, preprocessor produced {tensor.shape}")
    for axis, dim in enumerate(expected):
        if isinstance(dim, int) and dim > 0 and dim != tensor.shape[axis]:
            raise ValueError(
                f"{model_path}: static input axis {axis} is {dim}, but preprocessor produced {tensor.shape}; "
                "use a matching export resolution or re-export dynamic axes."
            )


def provider_candidates(requested: str, available: list[str]) -> list[str]:
    """Return an ordered provider preference list; CPU is always the final fallback."""
    if requested != "auto":
        if requested not in available:
            raise RuntimeError(f"Requested provider {requested!r} is unavailable. Available: {available}")
        candidates = [requested]
    elif sys.platform == "win32":
        candidates = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "OpenVINOExecutionProvider",
            "WebGPUExecutionProvider",
        ]
    elif sys.platform == "darwin":
        candidates = ["CoreMLExecutionProvider", "WebGPUExecutionProvider"]
    else:
        candidates = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "MIGraphXExecutionProvider",
            "ROCMExecutionProvider",
            "OpenVINOExecutionProvider",
            "WebGPUExecutionProvider",
        ]
    return [name for name in candidates if name in available] + (
        ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else []
    )


def open_session(
    model_path: Path,
    options: ort.SessionOptions,
    requested_provider: str,
    available: list[str],
) -> tuple[ort.InferenceSession, str, list[dict[str, str]]]:
    attempts: list[dict[str, str]] = []
    for candidate in provider_candidates(requested_provider, available):
        # OpenVINO's AUTO policy can choose Intel CPU/GPU/NPU where available.
        provider_spec: list[Any] = [candidate]
        if candidate == "OpenVINOExecutionProvider":
            provider_spec = [(candidate, {"device_type": "AUTO"})]
        if candidate != "CPUExecutionProvider":
            provider_spec.append("CPUExecutionProvider")
        candidate_options = options
        if candidate == "DmlExecutionProvider":
            # DirectML requires sequential execution and disabled memory-pattern allocation.
            candidate_options = ort.SessionOptions()
            candidate_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            candidate_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            candidate_options.enable_mem_pattern = False
            candidate_options.intra_op_num_threads = options.intra_op_num_threads
        try:
            session = ort.InferenceSession(
                str(model_path), sess_options=candidate_options, providers=provider_spec
            )
            actual = session.get_providers()
            if candidate != "CPUExecutionProvider" and candidate not in actual:
                attempts.append({"provider": candidate, "result": f"not registered; actual={actual}"})
                del session
                continue
            attempts.append({"provider": candidate, "result": f"session created; actual={actual}"})
            return session, candidate, attempts
        except Exception as exc:
            attempts.append({"provider": candidate, "result": f"failed: {exc}"})
    raise RuntimeError(f"No ONNX Runtime provider could load the model. Attempts: {attempts}")


def benchmark_one(
    model_path: Path,
    mode: str,
    images: list[Path],
    output_dir: Path,
    provider: str,
    runs: int,
    warmups: int,
    resolution: int,
    invert_preview: bool,
    gpu_index: int,
    intra_threads: int,
) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")
    model_out = output_dir / mode
    model_out.mkdir(parents=True, exist_ok=True)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = intra_threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    available = ort.get_available_providers()

    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    load_started = time.perf_counter()
    with MemorySampler(gpu_index) as memory:
        session, selected_provider, provider_attempts = open_session(
            model_path, options, provider, available
        )
        load_ms = (time.perf_counter() - load_started) * 1000
        rss_after_load = proc.memory_info().rss
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        per_image = []

        for image_path in images:
            with Image.open(image_path) as opened:
                original = opened.convert("RGB")
            preprocess = preprocess_v2 if mode == "v2" else preprocess_da3
            pre_start = time.perf_counter()
            tensor = preprocess(original, 518 if mode == "v2" else resolution)
            preprocess_ms = (time.perf_counter() - pre_start) * 1000
            validate_input_shape(session, tensor, model_path)
            feeds = {input_meta.name: tensor}

            for _ in range(warmups):
                session.run([output_meta.name], feeds)

            inference_times = []
            for _ in range(runs):
                start = time.perf_counter()
                result = session.run([output_meta.name], feeds)[0]
                inference_times.append((time.perf_counter() - start) * 1000)

            depth = tensor_to_2d(result, model_path)
            raw_path = model_out / f"{image_path.stem}.npy"
            vis_path = model_out / f"{image_path.stem}.png"
            np.save(raw_path, depth)
            save_map(depth, vis_path, invert_preview)
            per_image.append(
                {
                    "image": str(image_path),
                    "originalSize": [original.width, original.height],
                    "inputTensorShape": list(tensor.shape),
                    "outputShape": list(result.shape),
                    "preprocessMsSingleSample": round(preprocess_ms, 3),
                    "inference": summarize(inference_times),
                    "rawMap": str(raw_path),
                    "preview": str(vis_path),
                }
            )

    result = {
        "modelPath": str(model_path.resolve()),
        "modelBytes": model_path.stat().st_size,
        "mode": mode,
        "providerRequested": provider,
        "providerSelected": selected_provider,
        "providerAttempts": provider_attempts,
        "activeSessionProviders": session.get_providers(),
        "intraOpThreads": intra_threads,
        "onnxruntimeVersion": ort.__version__,
        "availableProviders": available,
        "inputName": input_meta.name,
        "inputType": input_meta.type,
        "inputShapeFromGraph": input_meta.shape,
        "outputName": output_meta.name,
        "outputType": output_meta.type,
        "outputShapeFromGraph": output_meta.shape,
        "sessionLoadMs": round(load_ms, 3),
        "rssBeforeSessionMB": round(rss_before / 1024**2, 2),
        "rssAfterLoadMB": round(rss_after_load / 1024**2, 2),
        "rssDeltaAfterLoadMB": round((rss_after_load - rss_before) / 1024**2, 2),
        "sampledPeakRssMB": round(memory.peak_rss / 1024**2, 2),
        "sampledPeakProcessVramMB": round(memory.peak_vram / 1024**2, 2) if memory.peak_vram is not None else None,
        "vramNote": "null means NVIDIA per-process VRAM was unavailable (no pynvml, non-NVIDIA provider, or driver does not expose it).",
        "runsPerImage": runs,
        "warmupsPerImage": warmups,
        "resolutionArgument": resolution if mode == "da3" else 518,
        "images": per_image,
        "overallInference": summarize([img["inference"]["medianMs"] for img in per_image]),
    }
    del session
    gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-model", type=Path, required=True, help="RapidRAW Depth Anything V2 ONNX model")
    parser.add_argument("--da3-model", type=Path, required=True, help="Validated DA3-SMALL ONNX model")
    parser.add_argument("--images", type=Path, required=True, help="One image or directory tree of representative images")
    parser.add_argument("--provider", default="auto", help="Provider preference: auto, or an exact installed provider name; CPU is always retained as fallback (default: auto)")
    parser.add_argument("--intra-threads", type=int, default=0, help="ONNX intra-op CPU threads; 0 chooses min(logical CPUs minus one, 8), bounded to at least 1")
    parser.add_argument("--runs", type=int, default=10, help="Measured inference runs per image")
    parser.add_argument("--warmups", type=int, default=2, help="Discarded warmup runs per image")
    parser.add_argument("--da3-resolution", type=int, default=504, help="DA3 upper-bound resize target (normally 504)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON report, raw maps, and preview PNGs")
    parser.add_argument("--invert-previews", action="store_true", help="Invert only saved preview maps; raw .npy output stays unchanged")
    parser.add_argument("--gpu-index", type=int, default=0, help="NVIDIA device index used for optional VRAM sampling")
    args = parser.parse_args()

    if args.runs < 1 or args.warmups < 0:
        parser.error("--runs must be >= 1 and --warmups must be >= 0")
    images = image_files(args.images)
    if not images:
        parser.error(f"No supported images found under {args.images}")
    if args.da3_resolution < 14:
        parser.error("--da3-resolution must be >= 14")

    available_providers = ort.get_available_providers()
    logical_cpus = psutil.cpu_count(logical=True) or 1
    intra_threads = args.intra_threads if args.intra_threads > 0 else max(1, min(8, logical_cpus - 1))
    if args.intra_threads < 0:
        parser.error("--intra-threads must be 0 (auto) or a positive integer")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "createdAtLocal": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": sys.platform,
        "platformRelease": platform.platform(),
        "machineArchitecture": platform.machine(),
        "cpuDescription": platform.processor() or platform.uname().processor or "not reported by Python",
        "logicalCpuCount": logical_cpus,
        "systemRamMB": round(psutil.virtual_memory().total / 1024**2, 2),
        "pythonVersion": sys.version,
        "pid": os.getpid(),
        "providerRequested": args.provider,
        "availableProviders": available_providers,
        "intraOpThreads": intra_threads,
        "imageCount": len(images),
        "images": [str(p.resolve()) for p in images],
        "notes": [
            "Run both models on the same machine/provider and the same image list.",
            "V2 preprocessing approximates RapidRAW's Rust image resizer; resampling can differ slightly.",
            "DA3 preprocessing follows the reference aspect-preserving and multiple-of-14 path.",
            "Previews use robust percentile normalization and are not numerical depth ground truth.",
            "Session-load time is not whole-app startup time.",
            "Auto provider selection tries available accelerators in platform-preferred order and always retains CPU as fallback.",
            "A registered provider does not prove every graph node ran there; use ORT profiling on each target machine to confirm placement.",
            "DirectML is configured for sequential execution with memory-pattern allocation disabled.",
            "OpenVINO is requested with device_type=AUTO so compatible Intel systems may select CPU/GPU/NPU.",
            "Thread cap reserves one logical CPU up to a maximum of 8; compare explicit --intra-threads values for a particular machine.",
            "GPU memory is sampled only when NVIDIA per-process reporting is available; iGPU shared-memory accounting is platform-dependent.",
        ],
    }
    report["models"] = []
    for model_path, mode in ((args.v2_model, "v2"), (args.da3_model, "da3")):
        try:
            print(f"Benchmarking {mode}: {model_path}", flush=True)
            model_report = benchmark_one(
                model_path=model_path,
                mode=mode,
                images=images,
                output_dir=args.output_dir,
                provider=args.provider,
                runs=args.runs,
                warmups=args.warmups,
                resolution=args.da3_resolution,
                invert_preview=args.invert_previews,
                gpu_index=args.gpu_index,
                intra_threads=intra_threads,
            )
            report["models"].append(model_report)
        except Exception as exc:
            report["models"].append({"mode": mode, "modelPath": str(model_path), "error": str(exc)})
            print(f"ERROR benchmarking {mode}: {exc}", file=sys.stderr)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report: {report_path}")
    print(json.dumps(report, indent=2))
    return 1 if any("error" in model for model in report["models"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
