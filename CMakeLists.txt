cmake_minimum_required(VERSION 2.6)

set(PROJECT_NAME "ann")
set(MY_SOURCE_DIR "./src")

project(${PROJECT_NAME})

# get all source files
FILE(GLOB_RECURSE sourceFiles ${MY_SOURCE_DIR}/*.cpp)
set(SOURCE ${sourceFiles})

# get all source reader files
FILE(GLOB_RECURSE headerFiles ${MY_SOURCE_DIR}/*.h)
set(HEADERS ${headerFiles})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

add_executable(${PROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${PROJECT_NAME})