from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from packaging.version import Version


MIN_TRANSFORMERS = Version("4.52.1")
MAX_TRANSFORMERS = Version("6.0.0")
TESTED_TRANSFORMERS = ("4.52.1", "4.57.6", "5.6.0", "5.15.1")


@dataclass(frozen=True, slots=True)
class TransformersCompatibility:
    installed: str
    supported: bool
    generation: int
    minimum: str = str(MIN_TRANSFORMERS)
    maximum_exclusive: str = str(MAX_TRANSFORMERS)
    tested: tuple[str, ...] = TESTED_TRANSFORMERS
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def transformers_version() -> Version:
    try:
        return Version(version("transformers"))
    except PackageNotFoundError as exc:
        raise RuntimeError("Transformers is not installed.") from exc


def compatibility_report() -> TransformersCompatibility:
    installed = transformers_version()
    supported = MIN_TRANSFORMERS <= installed < MAX_TRANSFORMERS
    reason = ""
    if installed < MIN_TRANSFORMERS:
        reason = f"Transformers {installed} is too old; install >= {MIN_TRANSFORMERS}."
    elif installed >= MAX_TRANSFORMERS:
        reason = f"Transformers {installed} is newer than the validated < {MAX_TRANSFORMERS} range."
    return TransformersCompatibility(
        installed=str(installed),
        supported=supported,
        generation=installed.major,
        reason=reason,
    )


def require_compatible_transformers() -> TransformersCompatibility:
    report = compatibility_report()
    if not report.supported:
        raise RuntimeError(report.reason)
    return report


def pretrained_dtype_kwargs(load_dtype: Any = "auto", installed: Version | None = None) -> dict[str, Any]:
    installed = installed or transformers_version()
    key = "dtype" if installed.major >= 5 else "torch_dtype"
    return {key: load_dtype}


def load_local_model_and_processor(
    model_path: str | Path,
    *,
    load_dtype: Any = "auto",
    local_files_only: bool = True,
):
    """Load the audited local implementation without executing Hub remote code.

    The model snapshot still carries its upstream remote-code files for provenance,
    but desktop and ComfyUI runtimes use the compatibility-patched package code.
    """
    require_compatible_transformers()
    from .configuration_moss_transcribe_diarize import MossTranscribeDiarizeConfig
    from .modeling_moss_transcribe_diarize import MossTranscribeDiarizeForConditionalGeneration
    from .processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor

    model_path = str(Path(model_path).expanduser().resolve())
    config = MossTranscribeDiarizeConfig.from_pretrained(model_path, local_files_only=local_files_only)
    model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        local_files_only=local_files_only,
        **pretrained_dtype_kwargs(load_dtype),
    )
    processor = MossTranscribeDiarizeProcessor.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    return model, processor


__all__ = [
    "MAX_TRANSFORMERS",
    "MIN_TRANSFORMERS",
    "TESTED_TRANSFORMERS",
    "TransformersCompatibility",
    "compatibility_report",
    "load_local_model_and_processor",
    "pretrained_dtype_kwargs",
    "require_compatible_transformers",
    "transformers_version",
]
