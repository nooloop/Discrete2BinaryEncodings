cmake_minimum_required(VERSION 3.14)
project(encoding_tools CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options(/O2 /DNDEBUG /EHsc)
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
endif()

include(FetchContent)

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE)

FetchContent_Declare(Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE)

set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(json)

set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen3)

find_package(OpenMP)

add_executable(encode_cfn src/main.cpp)
target_include_directories(encode_cfn PRIVATE src)
target_link_libraries(encode_cfn PRIVATE nlohmann_json::nlohmann_json Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(encode_cfn PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(encode_cfn PRIVATE HAS_OPENMP)
endif()
