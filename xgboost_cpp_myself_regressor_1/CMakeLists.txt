SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)

ADD_EXECUTABLE(xgboost_cpp_myself_regressor_1 main.cpp)