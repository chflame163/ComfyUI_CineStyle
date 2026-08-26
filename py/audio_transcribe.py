"""Audio transcription nodes with the local MOSS backend.

Model weights are downloaded lazily into ``models/audio_models/moss`` and reused
by the transcription node through an internal normalized ``AudioASRModel`` handle.
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import logging
import re
import sys
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import folder_paths
from comfy_api.latest import ComfyExtension, io


_CATEGORY = "😺dzNodes/CineStyle/Audio"
_SAMPLE_RATE = 16_000
_AUDIO_MODEL_ROOT = Path(folder_paths.models_dir) / "audio_models"
_DOWNLOAD_LOCK = threading.RLock()
_MODEL_CACHE: dict[tuple[str, str, str], "AudioASRModel"] = {}
_LOGGER = logging.getLogger("CineStyleMossAudioTranscribe")

MOSS_MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"


@contextmanager
def _quiet_transformers_warnings():
    """Suppress Transformers warning/info noise during compatibility loading."""
    transformers_logging = None
    previous_verbosity = None
    try:
        from transformers.utils import logging as transformers_logging

        previous_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
    except Exception:
        transformers_logging = None
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`MossTranscribeDiarizeProcessor` defines `feature_extractor_class.*",
        )
        try:
            yield
        finally:
            if transformers_logging is not None and previous_verbosity is not None:
                transformers_logging.set_verbosity(previous_verbosity)

def _resolve_device() -> torch.device:
    requested = "auto"
    if requested == "auto":
        try:
            import comfy.model_management as model_management

            requested = str(model_management.get_torch_device())
        except Exception:
            requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        requested = "cuda:0"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but the current ComfyUI PyTorch build cannot see CUDA.")
    if device.type not in {"cpu", "cuda"}:
        raise RuntimeError(f"Unsupported audio transcription device: {device}.")
    return device


def _resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return torch.bfloat16 if _cuda_bf16_supported(device) else torch.float16


def _cuda_bf16_supported(device: torch.device) -> bool:
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        with torch.cuda.device(index):
            return bool(torch.cuda.is_bf16_supported())
    except (RuntimeError, AssertionError):
        return False


def _model_root(name: str) -> Path:
    path = _AUDIO_MODEL_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _has_model_files(path: Path, backend: str) -> bool:
    if backend == "moss":
        return (path / "config.json").is_file() and any(path.glob("*.safetensors"))
    return False


def _snapshot_download(repo_id: str, target: Path) -> Path:
    """Download the official MOSS snapshot from Hugging Face."""
    with _DOWNLOAD_LOCK:
        target.mkdir(parents=True, exist_ok=True)
        _LOGGER.info("[CS MOSS Audio Transcribe] downloading official MOSS weights: %s", repo_id)
        try:
            from huggingface_hub import snapshot_download

            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            except TypeError:
                # Newer huggingface_hub releases removed legacy keyword arguments.
                snapshot_download(repo_id=repo_id, local_dir=str(target))
        except Exception as exc:  # noqa: BLE001 - preserve the actionable error
            raise RuntimeError(
                f"Unable to download the official MOSS model into {target}. "
                "Install huggingface_hub or place the weights there manually."
            ) from exc
        return target


def _download_moss() -> Path:
    target = _model_root("moss")
    if _has_model_files(target, "moss"):
        _LOGGER.info("[CS MOSS Audio Transcribe] stage 2/6: using cached MOSS weights: %s", target)
    else:
        _snapshot_download(MOSS_MODEL_ID, target)
    return target


def _load_moss_vendor():
    package_name = "_cinestyle_audio_moss_vendor"
    package_dir = Path(__file__).with_name("audio_moss_vendor")
    loaded = sys.modules.get(package_name)
    if loaded is not None:
        return loaded
    init_file = package_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load vendored MOSS package from {package_dir}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def _load_moss_runtime(path: Path, device: torch.device) -> dict[str, Any]:
    _LOGGER.info("[CS MOSS Audio Transcribe] stage 4/6: loading MOSS runtime on %s", device)
    vendor = _load_moss_vendor()
    compat = importlib.import_module(f"{vendor.__name__}.transformers_compat")
    dtype = _resolve_dtype(device)
    compat.require_compatible_transformers()
    with _quiet_transformers_warnings():
        model, processor = compat.load_local_model_and_processor(path, load_dtype=dtype, local_files_only=True)
    model = model.to(device=device, dtype=dtype).eval()
    _LOGGER.info("[CS MOSS Audio Transcribe] MOSS runtime ready: dtype=%s", dtype)
    return {"model": model, "processor": processor, "device": device, "dtype": dtype}


class MOSSAdapter:
    """Local-vendor adapter for MOSS Transcribe-Diarize."""

    backend = "moss"

    @staticmethod
    def load(path: Path, device: torch.device) -> dict[str, Any]:
        return _load_moss_runtime(path, device)

    @staticmethod
    def transcribe(handle: "AudioASRModel", samples: np.ndarray, duration: float, language: str) -> list[dict[str, Any]]:
        return _moss_transcribe(handle, samples, duration, language)


@dataclass
class AudioASRModel:
    backend: str
    model_path: Path
    runtime: dict[str, Any]
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def release(self) -> None:
        device = self.runtime.get("device")
        self.runtime.clear()
        with _DOWNLOAD_LOCK:
            for key, candidate in list(_MODEL_CACHE.items()):
                if candidate is self:
                    _MODEL_CACHE.pop(key, None)
        gc.collect()
        if isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _cached_model(
    backend: str,
    model_path: Path,
    device: torch.device,
    loader,
) -> AudioASRModel:
    key = (backend, str(model_path.resolve()), str(device))
    with _DOWNLOAD_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None and cached.runtime:
            _LOGGER.info("[CS MOSS Audio Transcribe] stage 4/6: reusing cached MOSS runtime")
            return cached
        _LOGGER.info("[CS MOSS Audio Transcribe] stage 4/6: initializing MOSS runtime")
        runtime = loader()
        result = AudioASRModel(
            backend=backend,
            model_path=model_path,
            runtime=runtime,
        )
        _MODEL_CACHE[key] = result
        return result


def _audio_to_mono_16k(audio: dict[str, Any]) -> tuple[np.ndarray, float]:
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("audio must be a standard ComfyUI AUDIO dictionary.")
    waveform = audio["waveform"]
    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim != 2:
        raise ValueError("audio waveform must have shape [channels, samples] or [batch, channels, samples].")
    if waveform.shape[-1] == 0:
        raise ValueError("audio waveform is empty.")
    waveform = waveform.mean(dim=0, keepdim=True)
    sample_rate = int(audio.get("sample_rate", _SAMPLE_RATE))
    if sample_rate <= 0:
        raise ValueError("audio sample_rate must be positive.")
    if sample_rate != _SAMPLE_RATE and waveform.shape[-1] > 1:
        try:
            import torchaudio

            waveform = torchaudio.functional.resample(waveform, sample_rate, _SAMPLE_RATE)
        except Exception:
            target_length = max(1, int(round(waveform.shape[-1] * _SAMPLE_RATE / sample_rate)))
            waveform = F.interpolate(waveform.unsqueeze(0), size=target_length, mode="linear", align_corners=False)[0]
        sample_rate = _SAMPLE_RATE
    return waveform[0].numpy().astype(np.float32, copy=False), float(sample_rate)


def _clean_text(text: Any) -> str:
    value = str(text or "").replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\[(?:S|SPK)[-_]?\d+\]", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:S|SPK)[-_]?\d+\s*[:：]\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _parse_moss_segments(text: str, duration: float) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"\[(?P<start>\d+(?:\.\d+)?)\]\s*(?:\[(?:S|SPK)[-_]?\d+\])?"
        r"(?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]",
        re.DOTALL | re.IGNORECASE,
    )
    segments: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        start = float(match.group("start"))
        end = float(match.group("end"))
        value = _clean_text(match.group("text"))
        if value and end > start:
            segments.append({"start": start, "end": end, "text": value})
    if segments:
        return segments
    value = _clean_text(re.sub(r"\[\d+(?:\.\d+)?\]", "", text))
    return [{"start": 0.0, "end": max(duration, 0.1), "text": value}] if value else []


def _moss_transcribe(handle: AudioASRModel, samples: np.ndarray, duration: float, language: str) -> list[dict[str, Any]]:
    runtime = handle.runtime
    model = runtime["model"]
    processor = runtime["processor"]
    prompt = (
        "请将音频转写为文本，每一段自然句或完整对白都必须单独成段，不能把整段音频合并成一个段落。"
        "每段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
        "并在段末标注结束时间戳，严格使用 [起始秒数][S01]台词内容[结束秒数] 格式。"
        "只转写人类语言，不要描述音乐、环境声音或事件；最终结果会移除说话人编号。"
    )
    if language and language != "auto":
        prompt += f"主要语言是{language}。"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": samples},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text,
        audio=[samples],
        max_length=131072,
        audio_kwargs={"sampling_rate": _SAMPLE_RATE},
        return_tensors="pt",
    ).to(runtime["device"])
    prompt_len = int(inputs["attention_mask"][0].sum().item())
    with torch.inference_mode(), (
        torch.amp.autocast("cuda", dtype=runtime["dtype"])
        if runtime["device"].type == "cuda" and runtime["dtype"] in (torch.float16, torch.bfloat16)
        else torch.no_grad()
    ):
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"],
            audio_feature_lengths=inputs["audio_feature_lengths"],
            audio_chunk_mapping=inputs["audio_chunk_mapping"],
            max_new_tokens=max(2048, min(65536, int(max(duration, 1.0) * 12))),
            do_sample=False,
        )
    decoded = processor.tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
    return _parse_moss_segments(decoded, duration)


def _format_srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(float(seconds) * 1000.0)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds_value, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d},{millis:03d}"


def _to_srt(segments: list[dict[str, Any]], max_chars_per_line: int = 0) -> str:
    blocks: list[str] = []
    index = 0
    for segment in segments:
        text = _clean_text(segment.get("text", ""))
        if not text:
            continue
        index += 1
        if max_chars_per_line > 0 and len(text) > max_chars_per_line:
            text = "\n".join(
                text[start : start + max_chars_per_line]
                for start in range(0, len(text), max_chars_per_line)
            )
        start = max(0.0, float(segment.get("start", 0.0)))
        end = max(start + 0.05, float(segment.get("end", start + 0.05)))
        blocks.append(f"{index}\n{_format_srt_time(start)} --> {_format_srt_time(end)}\n{text}")
    return "\n\n".join(blocks) + ("\n\n" if blocks else "")


class CSAudioTranscribe(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_MOSS_Audio_Transcribe",
            display_name="CS MOSS Audio Transcribe",
            category=_CATEGORY,
            essentials_category="Audio",
            search_aliases=["SRT", "subtitle", "MOSS", "speech recognition"],
            description="Downloads and runs the official MOSS model on a standard ComfyUI AUDIO input and returns SRT text.",
            inputs=[
                io.Audio.Input("audio", tooltip="Standard ComfyUI AUDIO input."),
                io.Combo.Input("language", options=["auto", "中文", "English"], default="auto", advanced=True),
                io.Int.Input("max_chars_per_line", default=0, min=0, max=200, step=1, advanced=True),
                io.Boolean.Input("auto_unload_model", default=True, advanced=True, tooltip="Unload MOSS weights after each transcription."),
            ],
            outputs=[io.String.Output("srt", display_name="SRT")],
        )

    @classmethod
    def execute(
        cls,
        audio: dict[str, Any],
        language: str = "auto",
        max_chars_per_line: int = 0,
        auto_unload_model: bool = True,
    ) -> io.NodeOutput:
        _LOGGER.info("[CS MOSS Audio Transcribe] start")
        _LOGGER.info("[CS MOSS Audio Transcribe] stage 1/6: normalizing audio input")
        samples, sample_rate = _audio_to_mono_16k(audio)
        duration = float(samples.shape[0]) / sample_rate
        _LOGGER.info(
            "[CS MOSS Audio Transcribe] audio ready: %.2fs at %d Hz",
            duration,
            int(sample_rate),
        )
        _LOGGER.info("[CS MOSS Audio Transcribe] stage 2/6: resolving official MOSS weights")
        path = _download_moss()
        _LOGGER.info("[CS MOSS Audio Transcribe] stage 3/6: resolving ComfyUI compute device")
        device = _resolve_device()
        _LOGGER.info("[CS MOSS Audio Transcribe] compute device: %s", device)
        model = _cached_model(
            "moss",
            path,
            device,
            lambda: MOSSAdapter.load(path, device),
        )
        try:
            _LOGGER.info("[CS MOSS Audio Transcribe] stage 5/6: transcribing audio")
            with model.lock:
                segments = MOSSAdapter.transcribe(model, samples, duration, language)
            _LOGGER.info("[CS MOSS Audio Transcribe] transcription complete: %d segments", len(segments))
            _LOGGER.info("[CS MOSS Audio Transcribe] stage 6/6: generating SRT output")
            result = _to_srt(segments, max_chars_per_line=int(max_chars_per_line))
            _LOGGER.info("[CS MOSS Audio Transcribe] SRT output ready: %d characters", len(result))
        finally:
            if auto_unload_model:
                _LOGGER.info("[CS MOSS Audio Transcribe] unloading MOSS runtime")
                model.release()
                _LOGGER.info("[CS MOSS Audio Transcribe] MOSS runtime unloaded")
        return io.NodeOutput(result)


async def comfy_entrypoint() -> ComfyExtension:
    class _AudioTranscribeExtension(ComfyExtension):
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [CSAudioTranscribe]

    return _AudioTranscribeExtension()
