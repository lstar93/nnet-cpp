cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(matplotlib_cpp LANGUAGES CXX)

set(SOURCE 
  main.cpp 
  plot.cpp
  layer.cpp
  nnet.cpp
)

set(HEADERS 
  plot.h
  data.h
  layer.h
  nnet.h
  cost.h
  activation.h
  helpers.h
)

add_executable(nnet ${SOURCE} ${HEADERS})
target_compile_features(nnet PRIVATE cxx_std_20)

target_include_directories(nnet PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}
  "$ENV{PYTHONHOME}\\include"
  "$ENV{PYTHONHOME}\\Lib\\site-packages\\numpy\\core\\include"
)
target_link_directories(nnet PRIVATE 
  "$ENV{PYTHONHOME}"
  "$ENV{PYTHONHOME}\\libs"
)