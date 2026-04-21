cmake_minimum_required(VERSION 3.20)
project(HypercubeRCPython LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── pybind11 ──
find_package(pybind11 CONFIG REQUIRED)

# ── Optimization flags (match main project) ──
# HYPERCUBE_ARCH controls -march. Defaults to "native" for local dev builds.
# cibuildwheel overrides to "x86-64-v2" (x86_64) or "none" (ARM, MSVC).
set(HYPERCUBE_ARCH "native" CACHE STRING "Target architecture for -march (native, x86-64-v2, none)")

if(MSVC)
    add_compile_options(/O2 /fp:fast)
else()
    add_compile_options(-O3 -ffast-math)
    if(NOT HYPERCUBE_ARCH STREQUAL "none")
        if(HYPERCUBE_ARCH STREQUAL "native")
            add_compile_options(-march=native -mtune=native)
        else()
            add_compile_options(-march=${HYPERCUBE_ARCH} -mtune=generic)
        endif()
    endif()
    add_compile_options(-Wall -Wextra -Wno-unknown-pragmas)
endif()

# ── Core sources compiled directly into the module (PIC required) ──
set(CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CORE_SOURCES
    ${CORE_DIR}/Reservoir.cpp
    ${CORE_DIR}/ESN.cpp
    ${CORE_DIR}/readout/RidgeRegression.cpp
)

# ── Build the Python extension module ──
pybind11_add_module(_core bindings.cpp ${CORE_SOURCES})
target_include_directories(_core PRIVATE ${CORE_DIR} ${CORE_DIR}/readout)

# ── Linking ──
if(MINGW)
    # Statically link MinGW runtime DLLs so the .pyd is self-contained.
    target_link_options(_core PRIVATE -static-libgcc -static-libstdc++)
    target_link_libraries(_core PRIVATE -Wl,-Bstatic winpthread -Wl,-Bdynamic)
endif()

# ── Install into the hypercube_rc package directory ──
install(TARGETS _core DESTINATION hypercube_rc)
