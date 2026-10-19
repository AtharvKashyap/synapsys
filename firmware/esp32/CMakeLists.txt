# SynapSys SSVEP BCI -- ESP-IDF (v5.3) application for the ESP32-S3.
# The platform-independent decoder core lives in ../decoder and is pulled in as the
# ESP-IDF component "decoder" (its CMakeLists.txt registers a component when ESP_PLATFORM
# is set). This project never modifies the core.
cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS ${CMAKE_CURRENT_LIST_DIR}/../decoder)
set(PROJECT_VER "0.1.0")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(synapsys_bci)
