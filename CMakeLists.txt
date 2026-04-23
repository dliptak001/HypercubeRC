cmake_minimum_required(VERSION 4.1)
project(HypercubeRC VERSION 0.2.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Prefer -std=c++23 over GNU extensions

# ——— Static library: core pipeline (Reservoir, ESN, Readout) ———
add_library(HypercubeRCCore STATIC
    ${PROJECT_SOURCE_DIR}/Reservoir.cpp
    ${PROJECT_SOURCE_DIR}/ESN.cpp
    ${PROJECT_SOURCE_DIR}/Readout.cpp
)

# Public headers — consumers need these via target_link_libraries
set(PUBLIC_HEADERS
    ${PROJECT_SOURCE_DIR}/ESN.h
    ${PROJECT_SOURCE_DIR}/Reservoir.h
    ${PROJECT_SOURCE_DIR}/Readout.h
)

# ——— HypercubeCNN dependency (CNN readout) ———
# Sibling project at ../HypercubeCNN.  Built as a static library.
set(HCNN_DIR "${PROJECT_SOURCE_DIR}/../HypercubeCNN")
set(HCNN_BUILD_DIR "${HCNN_DIR}/cmake-build-release")

add_library(HypercubeCNNCore STATIC IMPORTED)
set_target_properties(HypercubeCNNCore PROPERTIES
    IMPORTED_LOCATION "${HCNN_BUILD_DIR}/libHypercubeCNNCore.a"
    INTERFACE_INCLUDE_DIRECTORIES "${HCNN_DIR}"
)

# Include directories: PROJECT_SOURCE_DIR for in-tree builds,
# install prefix include/HypercubeRC for installed consumers
target_include_directories(HypercubeRCCore
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/HypercubeRC>
)

target_link_libraries(HypercubeRCCore PUBLIC HypercubeCNNCore)

# ——— Optimization and warning flags (library) ———
# Note: -flto is unavailable here — CLion 2024.3.2's bundled MinGW gcc 13.1
# was built without LTO support (cc1plus: "LTO support has not been enabled
# in this configuration").  Would need a different MinGW distribution.
target_compile_options(HypercubeRCCore PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math>
        $<$<CONFIG:Debug>:-O0 -g>
        -Wall -Wextra -Wno-unknown-pragmas
)

if(MINGW)
    target_link_options(HypercubeRCCore PRIVATE -static-libgcc -static-libstdc++)
endif()

# ——— Executables: only when building standalone (not via FetchContent) ———
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_executable(HypercubeRC main.cpp)
    add_executable(BasicPrediction examples/BasicPrediction.cpp)
    add_executable(SignalClassification examples/SignalClassification.cpp)
    add_executable(StreamingAnomaly examples/StreamingAnomaly.cpp)
    file(GLOB HRCCNN_LM_TEXT_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/examples/HRCCNN_LM_Text/*.cpp)
    add_executable(HRCCNN_LM_Text ${HRCCNN_LM_TEXT_SOURCES})
    add_executable(CoreSmokeTest diagnostics/CoreSmokeTest.cpp)

    set(EXAMPLE_TARGETS BasicPrediction SignalClassification StreamingAnomaly HRCCNN_LM_Text)
    set(TEST_TARGETS CoreSmokeTest)
    set(EXE_TARGETS HypercubeRC ${EXAMPLE_TARGETS} ${TEST_TARGETS})

    foreach(tgt IN LISTS EXE_TARGETS)
        target_link_libraries(${tgt} PRIVATE HypercubeRCCore)
    endforeach()

    foreach(tgt IN LISTS EXE_TARGETS)
        target_compile_options(${tgt} PRIVATE
                $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math>
                $<$<CONFIG:Debug>:-O0 -g>
                -Wall -Wextra -Wno-unknown-pragmas
        )

        if(MINGW)
            target_link_options(${tgt} PRIVATE -static-libgcc -static-libstdc++)
            # Workaround for a gcc 13.1 MinGW bug where
            # `std::type_info::operator==` in <typeinfo> is emitted out-of-line
            # at -O0 (Debug) and collides with libstdc++.a(tinfo.o). Release
            # (-O3) inlines it and is unaffected. Fixed upstream in gcc 13.2,
            # but CLion 2024.3.2 / 2026.1 both still bundle 13.1. Debug-only
            # so we don't mask real multi-def bugs in Release.
            target_link_options(${tgt} PRIVATE
                $<$<CONFIG:Debug>:-Wl,--allow-multiple-definition>)
        endif()
    endforeach()
endif()

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
