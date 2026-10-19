# SynapSys decoder core.
#  * ESP-IDF: this directory is a component (float build, Real = float).
#  * Host:    cmake -S firmware/decoder -B firmware/decoder/build -G Ninja
#             cmake --build firmware/decoder/build && ctest --test-dir firmware/decoder/build
#    builds the library in float (default) and double (SYNAPSYS_REAL_DOUBLE) and runs the
#    unit tests + fixture-equivalence runner for both.
if(ESP_PLATFORM)
  idf_component_register(SRCS "src/decoder.cpp" "src/rearm.cpp"
                         INCLUDE_DIRS "include"
                         PRIV_INCLUDE_DIRS "src")
  target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra -Werror)
else()
  cmake_minimum_required(VERSION 3.16)
  project(synapsys_decoder LANGUAGES CXX)

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()

  get_filename_component(SYNAPSYS_REPO "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  set(SYNAPSYS_FIXTURES "${SYNAPSYS_REPO}/fixtures")
  set(SYNAPSYS_TOOLS "${SYNAPSYS_REPO}/firmware/tools")
  set(SYNAPSYS_REPORTS "${SYNAPSYS_REPO}/firmware/reports")
  set(SYNAPSYS_GENERATED "${CMAKE_BINARY_DIR}/generated/fixtures")

  # Prefer the project venv; the tools only need the standard library.
  if(EXISTS "${SYNAPSYS_REPO}/.venv/bin/python")
    set(SYNAPSYS_PYTHON "${SYNAPSYS_REPO}/.venv/bin/python")
  else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(SYNAPSYS_PYTHON "${Python3_EXECUTABLE}")
  endif()

  execute_process(COMMAND git -C "${SYNAPSYS_REPO}" describe --always --dirty --tags
                  OUTPUT_VARIABLE SYNAPSYS_GIT_REV OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
  if(NOT SYNAPSYS_GIT_REV)
    set(SYNAPSYS_GIT_REV "unknown")
  endif()

  set(SYNAPSYS_FLAGS -Wall -Wextra -Werror -fno-exceptions -fno-rtti)

  enable_testing()

  # Flatten expected.json -> CSV for the C++ runner (outside fixtures/), on every build.
  add_custom_target(flatten_fixtures ALL
    COMMAND "${SYNAPSYS_PYTHON}" "${SYNAPSYS_TOOLS}/flatten_fixtures.py"
            --fixtures "${SYNAPSYS_FIXTURES}" --out "${SYNAPSYS_GENERATED}"
    COMMENT "Flattening fixture expected.json files"
    VERBATIM)

  add_test(NAME gen_config_check
           COMMAND "${SYNAPSYS_PYTHON}" "${SYNAPSYS_TOOLS}/gen_config.py" --check)

  foreach(prec float double)
    add_library(synapsys_decoder_${prec} STATIC src/decoder.cpp src/rearm.cpp)
    target_include_directories(synapsys_decoder_${prec} PUBLIC include PRIVATE src)
    target_compile_options(synapsys_decoder_${prec} PRIVATE ${SYNAPSYS_FLAGS})
    if(prec STREQUAL "double")
      target_compile_definitions(synapsys_decoder_${prec} PUBLIC SYNAPSYS_REAL_DOUBLE)
    endif()

    add_executable(test_unit_${prec} tests/test_unit.cpp)
    target_include_directories(test_unit_${prec} PRIVATE src)
    target_link_libraries(test_unit_${prec} PRIVATE synapsys_decoder_${prec})
    target_compile_options(test_unit_${prec} PRIVATE ${SYNAPSYS_FLAGS})
    add_test(NAME unit_${prec} COMMAND test_unit_${prec})

    add_executable(test_equivalence_${prec} tests/test_equivalence.cpp)
    target_link_libraries(test_equivalence_${prec} PRIVATE synapsys_decoder_${prec})
    target_compile_options(test_equivalence_${prec} PRIVATE ${SYNAPSYS_FLAGS})
    add_dependencies(test_equivalence_${prec} flatten_fixtures)
    add_test(NAME equivalence_${prec}
             COMMAND test_equivalence_${prec}
                     --fixtures "${SYNAPSYS_FIXTURES}"
                     --generated "${SYNAPSYS_GENERATED}"
                     --report "${SYNAPSYS_REPORTS}/equivalence_${prec}.md"
                     --rev "${SYNAPSYS_GIT_REV}")
  endforeach()
  # Float build fed through push() with samples already rounded to float32 volts.
  add_test(NAME equivalence_float_real_input
           COMMAND test_equivalence_float
                   --fixtures "${SYNAPSYS_FIXTURES}"
                   --generated "${SYNAPSYS_GENERATED}"
                   --report "${SYNAPSYS_REPORTS}/equivalence_float_real_input.md"
                   --rev "${SYNAPSYS_GIT_REV}"
                   --input real)
  add_library(synapsys_decoder ALIAS synapsys_decoder_float)

  add_executable(test_rearm tests/test_rearm.cpp)
  target_link_libraries(test_rearm PRIVATE synapsys_decoder_float)
  target_compile_options(test_rearm PRIVATE ${SYNAPSYS_FLAGS})
  add_test(NAME rearm_vs_python COMMAND test_rearm "${SYNAPSYS_FIXTURES}/rearm/sequences.csv")

  add_executable(bench_float tests/bench.cpp)
  target_link_libraries(bench_float PRIVATE synapsys_decoder_float)
  target_compile_options(bench_float PRIVATE ${SYNAPSYS_FLAGS})
  add_test(NAME bench_float COMMAND bench_float)
endif()
