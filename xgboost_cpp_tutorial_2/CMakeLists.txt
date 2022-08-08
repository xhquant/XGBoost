SET(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin)

ADD_EXECUTABLE(xgboost_cpp_tutorial_2 main.cpp)
TARGET_LINK_LIBRARIES(xgboost_cpp_tutorial_2 xgboost::xgboost)