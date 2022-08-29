CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

PROJECT(xgboost)

find_package(xgboost REQUIRED)

ADD_SUBDIRECTORY(xgboost_cpp_tutorial_1)
ADD_SUBDIRECTORY(xgboost_cpp_tutorial_2)
ADD_SUBDIRECTORY(xgboost_cpp_myself_regressor_1)