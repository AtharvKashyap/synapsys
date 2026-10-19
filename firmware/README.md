# firmware/ — SynapSys decoder port (C++ / ESP32-S3)

Status: **planned, no code yet.** Read `firmware/PLAN.md` first.

The port reproduces the frozen Python reference (`docs/reference_v1.md`, tag
`python-ref-v1`) and is accepted only when it matches `fixtures/` trial by trial.

Intended layout (created as each phase starts):

```
firmware/
├── PLAN.md                 phased plan, acceptance gates, open decisions
├── decoder/                platform-independent C++17 core (no ESP-IDF, no heap in hot path)
│   ├── include/synapsys/   public headers
│   ├── src/                biquad cascade, ring buffer, references, CCA, decision, re-arm
│   └── tests/              host unit tests + fixture-equivalence runner (CMake + ctest)
├── tools/                  config/SOS -> generated C++ header, fixture comparison reports
└── esp32/                  ESP-IDF project: ADS1299 SPI, acquisition task, BLE (later)
```
