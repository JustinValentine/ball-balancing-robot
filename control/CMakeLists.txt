cmake_minimum_required(VERSION 3.16)
project(ball_control LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Shared library used by the MuJoCo simulator through ctypes.
add_library(ballctl SHARED capi.cpp)
target_compile_options(ballctl PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test-ballctl test_ballctl.cpp)
target_compile_options(test-ballctl PRIVATE -Wall -Wextra -Wpedantic)

# Hardware runtime: three moteus controllers on a Linux SocketCAN interface.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include(FetchContent)
  FetchContent_Declare(moteus
    GIT_REPOSITORY https://github.com/mjbots/moteus.git
    GIT_TAG 44b4cc07f4c746285bf70236907dfd55dff894fa)
  FetchContent_GetProperties(moteus)
  if(NOT moteus_POPULATED)
    FetchContent_Populate(moteus)
  endif()
  find_package(Threads REQUIRED)
  add_executable(ball-control ball_control.cpp)
  target_include_directories(ball-control PRIVATE ${moteus_SOURCE_DIR}/lib/cpp/mjbots/moteus)
  target_link_libraries(ball-control PRIVATE Threads::Threads)
  target_compile_options(ball-control PRIVATE -Wall -Wextra)
endif()
