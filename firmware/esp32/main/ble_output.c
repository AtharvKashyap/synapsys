#include "ble_output.h"

#include <string.h>

#include "sdkconfig.h"

#if CONFIG_SYNAPSYS_BLE_ENABLE

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

static const char* TAG = "ble";

// 128-bit UUIDs a5f3XXXX-2c7e-4b8a-9d61-53594e415053 (last 6 bytes = ASCII "SYNAPS").
// BLE_UUID128_INIT takes the bytes little-endian (reversed string order).
#define SYN_UUID(lo) \
  BLE_UUID128_INIT(0x53, 0x50, 0x41, 0x4e, 0x59, 0x53, 0x61, 0x9d, 0x8a, 0x4b, 0x7e, 0x2c, (lo), \
                   0x00, 0xf3, 0xa5)
static const ble_uuid128_t s_svc_uuid = SYN_UUID(0x01);
static const ble_uuid128_t s_cmd_uuid = SYN_UUID(0x02);
static const ble_uuid128_t s_dbg_uuid = SYN_UUID(0x03);
static const ble_uuid128_t s_status_uuid = SYN_UUID(0x04);

enum { CHR_COMMAND = 1, CHR_DEBUG = 2, CHR_STATUS = 3 };

static portMUX_TYPE s_lock = portMUX_INITIALIZER_UNLOCKED;
static char s_command[16] = "";
static char s_debug[96] = "";
static char s_status[192] = "booting";
static char s_name[32] = "SynapSys";

static uint16_t s_cmd_handle, s_dbg_handle, s_status_handle;
static volatile uint16_t s_conn = BLE_HS_CONN_HANDLE_NONE;
static volatile bool s_sub_cmd, s_sub_dbg;
static uint8_t s_own_addr_type;
static bool s_ready;

static void copy_locked(char* dst, size_t cap, const char* src) {
  taskENTER_CRITICAL(&s_lock);
  strncpy(dst, src, cap - 1);
  dst[cap - 1] = '\0';
  taskEXIT_CRITICAL(&s_lock);
}

static int chr_access(uint16_t conn_handle, uint16_t attr_handle,
                      struct ble_gatt_access_ctxt* ctxt, void* arg) {
  (void)conn_handle;
  (void)attr_handle;
  if (ctxt->op != BLE_GATT_ACCESS_OP_READ_CHR) return BLE_ATT_ERR_UNLIKELY;
  char buf[192];
  const char* src = "";
  switch ((int)(intptr_t)arg) {
    case CHR_COMMAND: src = s_command; break;
    case CHR_DEBUG: src = s_debug; break;
    case CHR_STATUS: src = s_status; break;
    default: return BLE_ATT_ERR_UNLIKELY;
  }
  taskENTER_CRITICAL(&s_lock);
  strncpy(buf, src, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';
  taskEXIT_CRITICAL(&s_lock);
  return os_mbuf_append(ctxt->om, buf, strlen(buf)) == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
}

static const struct ble_gatt_svc_def s_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &s_svc_uuid.u,
        .characteristics =
            (struct ble_gatt_chr_def[]){
                {
                    .uuid = &s_cmd_uuid.u,
                    .access_cb = chr_access,
                    .arg = (void*)(intptr_t)CHR_COMMAND,
                    .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
                    .val_handle = &s_cmd_handle,
                },
                {
                    .uuid = &s_dbg_uuid.u,
                    .access_cb = chr_access,
                    .arg = (void*)(intptr_t)CHR_DEBUG,
                    .flags = BLE_GATT_CHR_F_NOTIFY,
                    .val_handle = &s_dbg_handle,
                },
                {
                    .uuid = &s_status_uuid.u,
                    .access_cb = chr_access,
                    .arg = (void*)(intptr_t)CHR_STATUS,
                    .flags = BLE_GATT_CHR_F_READ,
                    .val_handle = &s_status_handle,
                },
                {0},
            },
    },
    {0},
};

static int gap_event(struct ble_gap_event* event, void* arg);

static void advertise(void) {
  struct ble_hs_adv_fields fields;
  memset(&fields, 0, sizeof(fields));
  fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
  fields.name = (uint8_t*)s_name;
  fields.name_len = (uint8_t)strlen(s_name);
  fields.name_is_complete = 1;
  int rc = ble_gap_adv_set_fields(&fields);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_gap_adv_set_fields rc=%d (name too long?)", rc);
    return;
  }
  // 128-bit service UUID in the scan response (does not fit next to the name).
  struct ble_hs_adv_fields rsp;
  memset(&rsp, 0, sizeof(rsp));
  rsp.uuids128 = &s_svc_uuid;
  rsp.num_uuids128 = 1;
  rsp.uuids128_is_complete = 1;
  rc = ble_gap_adv_rsp_set_fields(&rsp);
  if (rc != 0) ESP_LOGW(TAG, "ble_gap_adv_rsp_set_fields rc=%d", rc);

  struct ble_gap_adv_params adv;
  memset(&adv, 0, sizeof(adv));
  adv.conn_mode = BLE_GAP_CONN_MODE_UND;
  adv.disc_mode = BLE_GAP_DISC_MODE_GEN;
  rc = ble_gap_adv_start(s_own_addr_type, NULL, BLE_HS_FOREVER, &adv, gap_event, NULL);
  if (rc != 0 && rc != BLE_HS_EALREADY) {
    ESP_LOGE(TAG, "ble_gap_adv_start rc=%d", rc);
    return;
  }
  ESP_LOGI(TAG, "advertising as \"%s\"", s_name);
}

static int gap_event(struct ble_gap_event* event, void* arg) {
  (void)arg;
  switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
      if (event->connect.status == 0) {
        s_conn = event->connect.conn_handle;
        ESP_LOGI(TAG, "connected (handle %u)", (unsigned)s_conn);
      } else {
        advertise();
      }
      break;
    case BLE_GAP_EVENT_DISCONNECT:
      ESP_LOGI(TAG, "disconnected (reason 0x%x)", event->disconnect.reason);
      s_conn = BLE_HS_CONN_HANDLE_NONE;
      s_sub_cmd = false;
      s_sub_dbg = false;
      advertise();
      break;
    case BLE_GAP_EVENT_ADV_COMPLETE:
      advertise();
      break;
    case BLE_GAP_EVENT_SUBSCRIBE:
      if (event->subscribe.attr_handle == s_cmd_handle) {
        s_sub_cmd = event->subscribe.cur_notify;
        ESP_LOGI(TAG, "Command notify %s", s_sub_cmd ? "on" : "off");
      } else if (event->subscribe.attr_handle == s_dbg_handle) {
        s_sub_dbg = event->subscribe.cur_notify;
        ESP_LOGI(TAG, "Debug notify %s", s_sub_dbg ? "on" : "off");
      }
      break;
    case BLE_GAP_EVENT_MTU:
      ESP_LOGI(TAG, "MTU %u", (unsigned)event->mtu.value);
      break;
    default:
      break;
  }
  return 0;
}

static void on_sync(void) {
  int rc = ble_hs_util_ensure_addr(0);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_hs_util_ensure_addr rc=%d", rc);
    return;
  }
  rc = ble_hs_id_infer_auto(0, &s_own_addr_type);
  if (rc != 0) {
    ESP_LOGE(TAG, "ble_hs_id_infer_auto rc=%d", rc);
    return;
  }
  advertise();
}

static void on_reset(int reason) { ESP_LOGW(TAG, "NimBLE host reset, reason %d", reason); }

static void host_task(void* param) {
  (void)param;
  nimble_port_run();  // returns only after nimble_port_stop()
  nimble_port_freertos_deinit();
}

bool ble_output_init(const char* device_name) {
  if (device_name != NULL && device_name[0] != '\0') {
    strncpy(s_name, device_name, sizeof(s_name) - 1);
    s_name[sizeof(s_name) - 1] = '\0';
  }
  esp_err_t err = nimble_port_init();  // also initialises the BT controller (IDF >= 5.0)
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "nimble_port_init failed: %s", esp_err_to_name(err));
    return false;
  }
  ble_hs_cfg.reset_cb = on_reset;
  ble_hs_cfg.sync_cb = on_sync;

  ble_svc_gap_init();
  ble_svc_gatt_init();
  int rc = ble_gatts_count_cfg(s_svcs);
  if (rc == 0) rc = ble_gatts_add_svcs(s_svcs);
  if (rc != 0) {
    ESP_LOGE(TAG, "GATT service registration failed rc=%d", rc);
    return false;
  }
  rc = ble_svc_gap_device_name_set(s_name);
  if (rc != 0) ESP_LOGW(TAG, "ble_svc_gap_device_name_set rc=%d", rc);

  nimble_port_freertos_init(host_task);
  s_ready = true;
  return true;
}

static void notify(uint16_t val_handle, bool subscribed, const char* text) {
  const uint16_t conn = s_conn;
  if (!s_ready || conn == BLE_HS_CONN_HANDLE_NONE || !subscribed) return;
  struct os_mbuf* om = ble_hs_mbuf_from_flat(text, (uint16_t)strlen(text));
  if (om == NULL) return;
  // Consumes om. Payload beyond ATT_MTU - 3 is truncated by the stack.
  int rc = ble_gatts_notify_custom(conn, val_handle, om);
  if (rc != 0) ESP_LOGD(TAG, "notify rc=%d", rc);
}

void ble_output_command(const char* label) {
  copy_locked(s_command, sizeof(s_command), label);
  notify(s_cmd_handle, s_sub_cmd, label);
}

void ble_output_debug(const char* text) {
  copy_locked(s_debug, sizeof(s_debug), text);
  notify(s_dbg_handle, s_sub_dbg, text);
}

void ble_output_set_status(const char* text) { copy_locked(s_status, sizeof(s_status), text); }

bool ble_output_connected(void) { return s_conn != BLE_HS_CONN_HANDLE_NONE; }

#else  // !CONFIG_SYNAPSYS_BLE_ENABLE

bool ble_output_init(const char* device_name) {
  (void)device_name;
  return false;
}
void ble_output_command(const char* label) { (void)label; }
void ble_output_debug(const char* text) { (void)text; }
void ble_output_set_status(const char* text) { (void)text; }
bool ble_output_connected(void) { return false; }

#endif
