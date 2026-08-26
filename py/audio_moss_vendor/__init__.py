"""Vendored MOSS model implementation used by CineStyle.

Only the model/config/processor files are included here.  Subtitle parsing and
export are deliberately implemented by the CineStyle node so all backends share
one output contract.
"""

from .configuration_moss_transcribe_diarize import MossTranscribeDiarizeConfig
from .modeling_moss_transcribe_diarize import (
    MossTranscribeDiarizeForConditionalGeneration,
    MossTranscribeDiarizeModel,
    MossTranscribeDiarizePreTrainedModel,
    VQAdaptor,
)
from .processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor

__all__ = [
    "MossTranscribeDiarizeConfig",
    "MossTranscribeDiarizeForConditionalGeneration",
    "MossTranscribeDiarizeModel",
    "MossTranscribeDiarizePreTrainedModel",
    "MossTranscribeDiarizeProcessor",
    "VQAdaptor",
]
