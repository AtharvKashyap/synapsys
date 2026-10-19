# Raw-sample stream: firmware in QEMU vs Python (2026-10-19)

Firmware: `firmware/esp32` at 912f48b + raw-stream changes (this commit), built with
`sdkconfig.defaults;sdkconfig.qemu.defaults` + `CONFIG_SYNAPSYS_RAW_STREAM=y`, run in
Espressif QEMU (ESP32-S3), replaying the 6 embedded synthetic fixtures in real time.

Checks:
- RAW lines: 9000 (6 x 1500), `seq` contiguous per stream, no `raw_gap`, `raw_dropped` 0.
- RAW values vs `fixtures/synthetic/<id>/raw.csv` x 1e6: max |diff| 0.0004999 uV (= %.3f rounding).
- Replay self-check: 48/48 PASS (unchanged).
- Host chain: `scripts/verify_device_log.py serial.log` parses the RAW samples, decodes them in
  Python on the device schedule and compares with the device's own decision lines:

```
syn_yes_snr-15           samples   1500  points    8  failed   0  max score err 6.36e-06
syn_no_snr-20            samples   1500  points    8  failed   0  max score err 6.63e-06
syn_help_snr-20          samples   1500  points    8  failed   0  max score err 6.06e-06
syn_noise_only           samples   1500  points    8  failed   0  max score err 6.14e-05
syn_dead_channel         samples   1500  points    8  failed   0  max score err 9.30e-06
syn_flat                 samples   1500  points    8  failed   0  max score err n/a
48 points compared, 0 failed, 0 near a tie, 0 stream(s) skipped (no raw_info / not from seq 0)
```
