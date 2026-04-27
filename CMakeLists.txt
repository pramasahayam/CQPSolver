cmake_minimum_required(VERSION 3.20)
project(CQPSolver LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(UMFPACK REQUIRED)

find_path(MATIO_INCLUDE_DIR matio.h HINTS /opt/homebrew/include)
find_library(MATIO_LIBRARY matio    HINTS /opt/homebrew/lib)
if(NOT MATIO_INCLUDE_DIR OR NOT MATIO_LIBRARY)
  message(FATAL_ERROR "libmatio not found. Run: brew install libmatio")
endif()

add_library(cqpsolver_lib STATIC cqpsolver.cpp)
target_include_directories(cqpsolver_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${MATIO_INCLUDE_DIR})
target_link_libraries(cqpsolver_lib PUBLIC Eigen3::Eigen SuiteSparse::UMFPACK)

add_executable(run_mm_benchmark run_mm_benchmark.cpp)
target_link_libraries(run_mm_benchmark PRIVATE cqpsolver_lib ${MATIO_LIBRARY})
