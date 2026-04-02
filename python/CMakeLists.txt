cmake_minimum_required(VERSION 3.20)
project(HypercubeRCPython LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── pybind11 ──
find_package(pybind11 CONFIG REQUIRED)

# ── Optimization flags (match main project) ──
if(NOT MSVC)
    add_compile_options(-O3 -ffast-math -fopenmp)
    if(NOT CMAKE_CROSSCOMPILING)
        add_compile_options(-march=native -mtune=native)
    endif()
    add_compile_options(-Wall -Wextra -Wno-unknown-pragmas)
elseif(MSVC)
    add_compile_options(/openmp)
endif()

# ── Core sources compiled directly into the module (PIC required) ──
set(CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CORE_SOURCES
    ${CORE_DIR}/Reservoir.cpp
    ${CORE_DIR}/ESN.cpp
    ${CORE_DIR}/TranslationLayer.cpp
    ${CORE_DIR}/readout/LinearReadout.cpp
    ${CORE_DIR}/readout/RidgeRegression.cpp
)

# ── Build the Python extension module ──
pybind11_add_module(_core bindings.cpp ${CORE_SOURCES})
target_include_directories(_core PRIVATE ${CORE_DIR} ${CORE_DIR}/readout)

# ── Linking ──
if(MINGW)
    # Statically link ALL MinGW runtime DLLs so the .pyd is self-contained.
    # -fopenmp in compile flags enables OpenMP pragmas; we link gomp statically.
    target_link_options(_core PRIVATE -static-libgcc -static-libstdc++)
    target_link_libraries(_core PRIVATE -Wl,-Bstatic gomp pthread winpthread -Wl,-Bdynamic)
elseif(NOT MSVC)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# ── Install into the hypercube_rc package directory ──
install(TARGETS _core DESTINATION hypercube_rc)
