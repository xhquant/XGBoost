#include <iostream>
#include <fstream>
#include <chrono>

#include "xhquant_xgboost_regressor.hpp"


int main()
{
    std::vector<std::string> features{"f0", "f1", "f2", "f3", "f4"};
    std::vector<float> input{-0.35005823, -0.83965004, 0.39151503, 2.13081938, 0.8018929};

    std::ifstream inputStream("../model/model.json");
    char model[1024 * 1024];
    inputStream.read(model, 1024 * 1024);
    std::string model_str(model);

    xhquant::model::xhquant_xgboost_regressor regressor(features, 0.5);
    regressor.init(model_str);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i != 1000000; ++i)
    {
        regressor.forward(input.data());
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
    std::cout << regressor.forward(input.data()) << std::endl;
}

