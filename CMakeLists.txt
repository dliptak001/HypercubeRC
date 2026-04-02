cmake_minimum_required(VERSION 4.1)
project(HypercubeRC VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Prefer -std=c++23 over GNU extensions

# ——— Static library: core pipeline (Reservoir, ESN, TranslationLayer, readouts) ———
file(GLOB READOUT_SOURCES ${PROJECT_SOURCE_DIR}/readout/*.cpp)
add_library(HypercubeRCCore STATIC
    ${PROJECT_SOURCE_DIR}/Reservoir.cpp
    ${PROJECT_SOURCE_DIR}/ESN.cpp
    ${PROJECT_SOURCE_DIR}/TranslationLayer.cpp
    ${READOUT_SOURCES}
)

# Public headers — consumers need these via target_link_libraries
set(PUBLIC_HEADERS
    ${PROJECT_SOURCE_DIR}/ESN.h
    ${PROJECT_SOURCE_DIR}/Reservoir.h
    ${PROJECT_SOURCE_DIR}/TranslationLayer.h
    ${PROJECT_SOURCE_DIR}/readout/LinearReadout.h
    ${PROJECT_SOURCE_DIR}/readout/RidgeRegression.h
)

# Include directories: PROJECT_SOURCE_DIR and readout/ for in-tree builds,
# install prefix include/HypercubeRC for installed consumers
target_include_directories(HypercubeRCCore
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/readout>
        $<INSTALL_INTERFACE:include/HypercubeRC>
)

# ——— Executables link against the static library ———
add_executable(HypercubeRC main.cpp)
add_executable(BasicPrediction examples/BasicPrediction.cpp)
add_executable(SignalClassification examples/SignalClassification.cpp)
add_executable(StreamingAnomaly examples/StreamingAnomaly.cpp)
add_executable(StandaloneESNSweep diagnostics/StandaloneESNSweep.cpp)
add_executable(SeedSurvey diagnostics/SeedSurvey.cpp)

set(DIAGNOSTIC_TARGETS HypercubeRC StandaloneESNSweep SeedSurvey)
set(EXAMPLE_TARGETS BasicPrediction SignalClassification StreamingAnomaly)
set(EXE_TARGETS ${DIAGNOSTIC_TARGETS} ${EXAMPLE_TARGETS})
set(ALL_TARGETS HypercubeRCCore ${EXE_TARGETS})

foreach(tgt IN LISTS EXE_TARGETS)
    target_link_libraries(${tgt} PRIVATE HypercubeRCCore)
endforeach()

# ——— OpenMP (diagnostics only) ———
# Core library does NOT use OpenMP.  Diagnostic executables use it for
# parallel seed sweeps and memory-capacity computation.
if(MINGW)
    foreach(tgt IN LISTS DIAGNOSTIC_TARGETS)
        target_compile_options(${tgt} PRIVATE -fopenmp)
        target_link_options(${tgt} PRIVATE -fopenmp)
    endforeach()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        foreach(tgt IN LISTS DIAGNOSTIC_TARGETS)
            target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
        endforeach()
    endif()
elseif(MSVC)
    foreach(tgt IN LISTS DIAGNOSTIC_TARGETS)
        target_compile_options(${tgt} PRIVATE /openmp)
    endforeach()
endif()

foreach(tgt IN LISTS ALL_TARGETS)
    # Optimization and warning flags
    target_compile_options(${tgt} PRIVATE
            $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math>
            $<$<CONFIG:Debug>:-O0 -g>
            -Wall -Wextra -Wno-unknown-pragmas
    )

    # Static linking on MinGW for easier distribution
    if(MINGW)
        target_link_options(${tgt} PRIVATE -static-libgcc -static-libstdc++)
    endif()
endforeach()

# ——— Install rules ———
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install the static library
install(TARGETS HypercubeRCCore
    EXPORT HypercubeRCTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install public headers into include/HypercubeRC/
install(FILES ${PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/HypercubeRC
)

# Generate and install CMake package config
install(EXPORT HypercubeRCTargets
    FILE HypercubeRCTargets.cmake
    NAMESPACE HypercubeRC::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HypercubeRC
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/HypercubeRCConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeRCConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HypercubeRC
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeRCConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeRCConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeRCConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HypercubeRC
)
