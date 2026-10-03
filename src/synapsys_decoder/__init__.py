"""SynapSys SSVEP decoder: reference Python DSP + CCA pipeline.

Pipeline: raw (n_channels, n_samples) EEG -> preprocessing -> fixed window
-> CCA vs. sin/cos harmonic references -> per-class scores -> gating -> decision.
"""

from synapsys_decoder.cca import DegenerateSignalError, canonical_correlations, cca_score
from synapsys_decoder.classifier import classify_window, classify_window_with_config, decide
from synapsys_decoder.config import DecoderConfig, load_config
from synapsys_decoder.pipeline import DecodedTrial, decode_trial
from synapsys_decoder.types import REJECT, ClassificationResult, Trial

__all__ = [
    "REJECT",
    "ClassificationResult",
    "DecodedTrial",
    "DecoderConfig",
    "DegenerateSignalError",
    "Trial",
    "canonical_correlations",
    "cca_score",
    "classify_window",
    "classify_window_with_config",
    "decide",
    "decode_trial",
    "load_config",
]
__version__ = "0.1.0"
