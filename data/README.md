# data/

Raw EEG is **not** committed (size + licensing). Everything here except this
README and `.gitkeep` is git-ignored.

## Wang et al. 2017 SSVEP Benchmark (Tsinghua)

Place files as:

```
data/benchmark/
├── S1.mat ... S35.mat
├── Freq_Phase.mat
└── 64-channels.loc
```

Source: Tsinghua BCI lab (http://bci.med.tsinghua.edu.cn/download.html).
Citation: Y. Wang, X. Chen, X. Gao, S. Gao, "A Benchmark Dataset for
SSVEP-Based Brain–Computer Interfaces," IEEE TNSRE 25(10), 2017.

Each `S*.mat` has `data` shaped `[64 channels, 1500 samples, 40 targets, 6 blocks]`
at 250 Hz, in microvolts: 0.5 s pre-stimulus, 5 s stimulation, 0.5 s post.

## Own SynapSys recordings

`data/synapsys/` — CSV format documented in `src/synapsys_decoder/io/csv_eeg.py`.
