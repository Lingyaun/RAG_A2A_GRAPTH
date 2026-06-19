cmake_minimum_required(VERSION 3.10)
project(RAGDemo LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE CGRAPH_SRC "../../src/*.cpp")
include_directories(../..)
# libcurl + nlohmann/json 头文件来自 A2A SDK 的 third_party
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../a2a-cpp-sdk-main/a2a-cpp-sdk-main/third_party)

add_executable(build_index build_index.cpp ${CGRAPH_SRC})
add_executable(query query.cpp ${CGRAPH_SRC})
add_executable(demo demo.cpp ${CGRAPH_SRC})
add_executable(query_parallel query_parallel.cpp ${CGRAPH_SRC})
add_executable(benchmark benchmark.cpp ${CGRAPH_SRC})
add_executable(distributed_demo distributed_demo.cpp ${CGRAPH_SRC})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

foreach(tgt build_index query demo query_parallel benchmark distributed_demo)
    target_link_libraries(${tgt} Threads::Threads)
endforeach()

if(MINGW)
    foreach(tgt build_index query demo query_parallel benchmark distributed_demo)
        target_link_libraries(${tgt} ws2_32)
        target_link_directories(${tgt} PRIVATE "E:/Mingw/mingw64/bin")
        target_link_libraries(${tgt} curl-4)
    endforeach()
endif()
