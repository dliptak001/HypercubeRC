cmake_minimum_required(VERSION 4.1)
project(HypercubeRC)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Prefer -std=c++23 over GNU extensions

# Source collection via GLOB — new .cpp files in project root and readout/ are picked up automatically
file(GLOB SOURCES
        ${PROJECT_SOURCE_DIR}/*.cpp
        ${PROJECT_SOURCE_DIR}/readout/*.cpp
)

add_executable(HypercubeRC ${SOURCES})

# ——— All targets ———
file(GLOB READOUT_SOURCES ${PROJECT_SOURCE_DIR}/readout/*.cpp)
set(CORE_SOURCES ${PROJECT_SOURCE_DIR}/Reservoir.cpp ${READOUT_SOURCES})

add_executable(BasicPrediction examples/BasicPrediction.cpp ${CORE_SOURCES})
add_executable(SignalClassification examples/SignalClassification.cpp ${CORE_SOURCES})
add_executable(StreamingAnomaly examples/StreamingAnomaly.cpp ${CORE_SOURCES})

set(ALL_TARGETS HypercubeRC BasicPrediction SignalClassification StreamingAnomaly)

foreach(tgt IN LISTS ALL_TARGETS)
    target_include_directories(${tgt} PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/readout)

    # Optimization and warning flags
    target_compile_options(${tgt} PRIVATE
            $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math>
            $<$<CONFIG:Debug>:-O0 -g>
            -Wall -Wextra -Wno-unknown-pragmas
    )

    # OpenMP — compiler-specific handling
    if(MINGW)
        target_compile_options(${tgt} PRIVATE -fopenmp)
        target_link_options(${tgt} PRIVATE -fopenmp)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU" AND NOT MINGW)
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${tgt} OpenMP::OpenMP_CXX)
        endif()
    elseif(MSVC)
        target_compile_options(${tgt} PRIVATE /openmp)
    endif()

    # Static linking on MinGW for easier distribution
    if(MINGW)
        target_link_options(${tgt} PRIVATE -static-libgcc -static-libstdc++)
    endif()
endforeach()
