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

target_include_directories(HypercubeRC PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/readout)

# ——— Compiler-specific OpenMP handling ———
if(MINGW)
    # MinGW requires -fopenmp for both compilation and linking to enable libgomp
    target_compile_options(HypercubeRC PRIVATE -fopenmp)
    target_link_options(HypercubeRC PRIVATE -fopenmp)
    # -fopenmp automatically defines _OPENMP for the preprocessor
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU" AND NOT MINGW)
    # Standard CMake OpenMP support works well on Linux/macOS GCC/Clang
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(HypercubeRC OpenMP::OpenMP_CXX)
    endif()
elseif(MSVC)
    # MSVC has built-in OpenMP support via /openmp flag
    target_compile_options(HypercubeRC PRIVATE /openmp)
endif()

# ——— Optimization and warning flags ———
target_compile_options(HypercubeRC PRIVATE
        $<$<CONFIG:Release>:
        -O3 -march=native -mtune=native -ffast-math
        >
        $<$<CONFIG:Debug>:
        -O0 -g
        >
        -Wall -Wextra -Wno-unknown-pragmas
)

# Optional: Static linking of libgcc/libstdc++ for easier distribution on Windows
if(MINGW)
    target_link_options(HypercubeRC PRIVATE -static-libgcc -static-libstdc++)
endif()

# ——— Examples ———
file(GLOB READOUT_SOURCES ${PROJECT_SOURCE_DIR}/readout/*.cpp)
set(CORE_SOURCES ${PROJECT_SOURCE_DIR}/Reservoir.cpp ${READOUT_SOURCES})

add_executable(BasicPrediction examples/BasicPrediction.cpp ${CORE_SOURCES})
target_include_directories(BasicPrediction PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/readout)
target_compile_options(BasicPrediction PRIVATE $<$<CONFIG:Release>:-O3 -march=native -ffast-math> -Wall -Wextra -Wno-unknown-pragmas)

add_executable(StreamingAnomaly examples/StreamingAnomaly.cpp ${CORE_SOURCES})
target_include_directories(StreamingAnomaly PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/readout)
target_compile_options(StreamingAnomaly PRIVATE $<$<CONFIG:Release>:-O3 -march=native -ffast-math> -Wall -Wextra -Wno-unknown-pragmas)
