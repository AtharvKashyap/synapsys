# synapsys-decoder

Reference Python implementation of the SynapSys SSVEP decoder:

```
raw multichannel EEG -> preprocessing (causal IIR) -> fixed window -> CCA vs. sin/cos references
                     -> 3 scores -> confidence/margin gating -> YES / NO / HELP / REJECT
```

This package is the algorithmic source of truth for the later ESP32-S3 C/C++ port.

*(Full documentation is filled in as milestones complete.)*
