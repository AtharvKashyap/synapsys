# SynapSys Stimulus Renderer

Frame-locked visual stimulus renderer for the SynapSys SSVEP
brain-computer interface.

The interface presents three communication targets:

- YES
- NO
- HELP

Each target uses periodic luminance modulation intended to evoke a
steady-state visually evoked potential (SSVEP). The three targets are
arranged top-left / top-right / bottom-center rather than in a row, so
a viewer fixating on one target keeps the other two further into the
periphery. A single static fixation point sits in the middle of the
layout, between the tiles: the viewer rests their gaze there, then looks
at the flickering target they mean to select.

## Current stimulus configuration

The initial implementation assumes a 60 Hz display:

| Command | Period    | Nominal frequency |
|---------|----------:|-------------------:|
| YES     | 7 frames  | 8.571 Hz           |
| NO      | 5 frames  | 12.000 Hz          |
| HELP    | 4 frames  | 15.000 Hz          |

The frame period, rather than a rounded frequency value, is stored as
the source of truth: only frequencies of the form `60 / N` for integer
`N` are exactly representable as a short repeating frame pattern on a
60 Hz display. For example, 8, 11, and 14 Hz are not exactly
representable as short integer-period sequences at 60 Hz — this
implementation picked the nearest mutually non-harmonic triplet
instead (8.571 / 12 / 15 Hz). Sampled sinusoidal stimulation, which
gives every frame its own luminance value rather than repeating a
short pattern, is a valid alternative that *can* target an arbitrary
frequency such as 11 Hz exactly; this renderer intentionally doesn't
use it yet, favoring the simpler, more debuggable integer-period model
until our own EEG data says otherwise.

At startup the renderer measures the display's actual refresh rate
(`Window.getActualFrameRate()`) and logs it alongside the assumed
60 Hz, along with the resulting actual frequencies. If the measured
rate deviates from 60 Hz by more than `REFRESH_TOLERANCE_HZ` (1 Hz by
default), a warning is logged — the renderer does not auto-reconfigure
itself, since a fixed, reproducible display configuration matters more
than automatically adapting to whatever display is plugged in.

## Requirements

- Python 3.11–3.13
- PsychoPy
- A display explicitly configured to run at a stable 60 Hz — do not
  assume the OS default is 60 Hz (e.g. ProMotion/120 Hz laptop
  displays); check the logged measured refresh rate at startup

## Setup

This folder is a self-contained Python project with its own virtual environment
(PsychoPy is large and is not needed by the decoder). From the repository root:

```bash
cd stimulus
```

Create an isolated virtual environment:

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Run

```bash
synapsys-stimulus
```

or:

```bash
python -m synapsys_stimulus
```

Press Esc to stop the renderer.

## Tests

```bash
pytest
```

## Linting

```bash
ruff check .
```

## Frame logging

Each run records frame timestamps under `logs/`.

These logs are useful for detecting dropped or unusually delayed
frames during development. Software timestamps alone do not prove the
physical refresh timing of the display; final timing validation should
use an external measurement method such as a photodiode.

## Architecture

```
config.py
    |
    v
patterns.py ----> renderer.py ----> PsychoPy Window
                       |
                       v
                 frame timing log
```

- `config.py` defines experimental parameters.
- `patterns.py` generates deterministic frame-indexed modulation.
- `renderer.py` owns the PsychoPy window and presentation loop.

## Project status

This renderer is experimental software for the SynapSys senior design
prototype. It is not a medical device and is not intended for clinical
use.
