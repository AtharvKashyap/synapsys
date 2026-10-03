"""SynapSys SSVEP decoder: reference Python DSP + CCA pipeline.

Pipeline: raw (n_channels, n_samples) EEG -> preprocessing -> fixed window
-> CCA vs. sin/cos harmonic references -> per-class scores -> gating -> decision.
"""

from synapsys_decoder.config import DecoderConfig, load_config
from synapsys_decoder.types import REJECT, ClassificationResult, Trial

__all__ = ["REJECT", "ClassificationResult", "DecoderConfig", "Trial", "load_config"]
__version__ = "0.1.0"
