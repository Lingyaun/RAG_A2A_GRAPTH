cmake_minimum_required(VERSION 3.10)
project(rag_agents LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(CURL REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/../multi_agent_demo
)
link_directories(${CMAKE_BINARY_DIR})

add_executable(retriever_agent retriever_agent.cpp)
target_link_libraries(retriever_agent a2a CURL::libcurl pthread)

add_executable(generator_agent generator_agent.cpp)
target_link_libraries(generator_agent a2a CURL::libcurl pthread)

add_executable(orchestrator_agent orchestrator_agent.cpp)
target_link_libraries(orchestrator_agent a2a CURL::libcurl pthread)

install(TARGETS retriever_agent generator_agent orchestrator_agent DESTINATION bin)
install(FILES rag_common.hpp DESTINATION include/rag_agents)
