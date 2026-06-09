cmake_minimum_required(VERSION 3.14)
project(solver_tools CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options(/O2 /DNDEBUG /EHsc)
else()
    # -march=native uses the host CPU's instruction set (e.g. AVX-512 on Ice Lake).
    # If cross-compiling, replace with -march=icelake-server.
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
endif()

include(FetchContent)

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE)

set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(json)

add_executable(solve_sa src/main.cpp)
target_include_directories(solve_sa PRIVATE src)
target_link_libraries(solve_sa PRIVATE nlohmann_json::nlohmann_json)

# D-Wave quantum annealing solver
add_executable(solve_qa src/main_qa.cpp)
target_include_directories(solve_qa PRIVATE src)
target_link_libraries(solve_qa PRIVATE nlohmann_json::nlohmann_json)

# Unit tests (SA)
add_executable(tests tests/tests.cpp)
target_include_directories(tests PRIVATE src)
target_link_libraries(tests PRIVATE nlohmann_json::nlohmann_json)

# Unit + integration tests (QA)
add_executable(tests_qa tests/tests_qa.cpp)
target_include_directories(tests_qa PRIVATE src)
target_link_libraries(tests_qa PRIVATE nlohmann_json::nlohmann_json)
