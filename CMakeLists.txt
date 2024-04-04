cmake_minimum_required(VERSION 3.28)
project(mrhappy C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")

add_executable(mrhappy main.c)
