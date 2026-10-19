idf_component_register(
  SRCS "app_main.cpp"
       "ads1299.cpp"
       "acquisition.cpp"
       "decoder_task.cpp"
       "replay.cpp"
       "serial_output.cpp"
       "ble_output.c"
  INCLUDE_DIRS "."
  REQUIRES decoder driver esp_timer esp_app_format nvs_flash bt)
