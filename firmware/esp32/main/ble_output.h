// BLE output: NimBLE GATT peripheral (written in C, like the ESP-IDF NimBLE examples).
//
// Service   a5f30001-2c7e-4b8a-9d61-53594e415053  "SynapSys BCI"
//   Command a5f30002-...  read + notify  UTF-8 "YES" / "NO" / "HELP", one notify per emitted command
//   Debug   a5f30003-...  notify         compact text per decision point (scores, margin, decision)
//   Status  a5f30004-...  read           firmware version, input source, measured sample rate, ...
//
// All functions are safe to call from any task once ble_output_init() returned true.
// With CONFIG_SYNAPSYS_BLE_ENABLE=n they compile to no-ops.
#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool ble_output_init(const char* device_name);
void ble_output_command(const char* label);   // update Command value + notify subscribers
void ble_output_debug(const char* text);      // notify Debug subscribers
void ble_output_set_status(const char* text); // value returned by Status reads
bool ble_output_connected(void);

#ifdef __cplusplus
}
#endif
