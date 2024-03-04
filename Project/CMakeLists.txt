cmake_minimum_required(VERSION 3.20)
project(Project)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-undefined")

add_executable(project ${HEADER_FILES} ${GLSOURCES} ${SOURCE_FILES} layer.cpp layer.h main.cpp network.cpp network.h UI.cpp UI.h parallel.cpp parallel.h simd.cpp simd.h sisd.cpp sisd.h simd_and_parallel.cpp simd_and_parallel.h)

target_compile_options(project PUBLIC -fopenmp -ffast-math -O2)
target_link_options(project PUBLIC -fopenmp)
target_include_directories(project PUBLIC ${GRAPHLIB_DIR})
target_include_directories(project PUBLIC include)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-undefined")
