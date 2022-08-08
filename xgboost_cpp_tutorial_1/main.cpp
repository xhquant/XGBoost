#include <fstream>
#include <iostream>

#include "spdlog/spdlog.h"

#include "xgboost/c_api.h"

int main(int argc, char **argv)
{
    int silent = 0;
    int use_gpu = 0;

    ////////////////////////////////////////////////////////////////////////
    /// 行数和列数
    bst_ulong num_rows = std::numeric_limits<long>::quiet_NaN();
    bst_ulong num_columns = std::numeric_limits<long>::quiet_NaN();

    ////////////////////////////////////////////////////////////////////////
    /// 版本号
    int major = 0;
    int minor = 0;
    int patch = 0;
    XGBoostVersion(&major, &minor, &patch);
    spdlog::info("主版本号: {}, 次版本号: {}, 补丁号: {}", major, minor, patch);

    ////////////////////////////////////////////////////////////////////////
    /// 输出全局配置参数
    const char *jsonName = "";
    spdlog::info("空配置参数: {}", jsonName);
    XGBGetGlobalConfig(&jsonName);
    spdlog::info("全局配置参数: {}", jsonName);

    ////////////////////////////////////////////////////////////////////////
    /// 从文件加载数据集
    DMatrixHandle data_train, data_test;
    std::string file_name = "../dataset/agaricus.txt.train";
    int rtn = XGDMatrixCreateFromFile(file_name.c_str(), silent, &data_train);
    if (rtn)
    {
        spdlog::info("加载数据集{}错误", file_name);
    }
    else
    {
        XGDMatrixNumRow(data_train, &num_rows);
        XGDMatrixNumCol(data_train, &num_columns);
        spdlog::info("行数: {}, 列数: {}", num_rows, num_columns);
    }

    file_name = "../dataset/agaricus.txt.test";
    rtn = XGDMatrixCreateFromFile(file_name.c_str(), silent, &data_test);
    if (rtn)
    {
        spdlog::info("加载数据集{}错误", file_name);
    }
    else
    {
        XGDMatrixNumRow(data_test, &num_rows);
        XGDMatrixNumCol(data_test, &num_columns);
        spdlog::info("行数: {}, 列数: {}", num_rows, num_columns);
    }

    ////////////////////////////////////////////////////////////////////////
    /// 创建booster学习器
    BoosterHandle booster;
    DMatrixHandle eval_matrix[2] = {data_train, data_test};
    XGBoosterCreate(eval_matrix, 2, &booster);

    ////////////////////////////////////////////////////////////////////////
    /// 配置参数
    XGBoosterSetParam(booster, "tree_method", use_gpu ? "gpu_hist" : "hist");    // tree method
    if (use_gpu) XGBoosterSetParam(booster, "gpu_id", "0");                      // use gpu
    else XGBoosterSetParam(booster, "gpu_id", "-1");
    XGBoosterSetParam(booster, "objective", "binary:logistic");
    XGBoosterSetParam(booster, "min_child_weight", "1");
    XGBoosterSetParam(booster, "gamma", "0.1");
    XGBoosterSetParam(booster, "max_depth", "3");
    XGBoosterSetParam(booster, "verbosity", silent ? "0" : "1");
    XGBoosterSetParam(booster, "eval_metric", "logloss");

    ////////////////////////////////////////////////////////////////////////
    /// 训练，进行十次迭代.
    int num_trees = 10;
    const char *eval_names[2] = {"train", "test"};
    const char *eval_result = nullptr;
    for (int i = 0; i != num_trees; ++i)
    {
        XGBoosterUpdateOneIter(booster, i, data_train);
        XGBoosterEvalOneIter(booster, i, eval_matrix, eval_names, 2, &eval_result);
        spdlog::info("eval_result:{}", eval_result);
    }

    ////////////////////////////////////////////////////////////////////////
    /// 获取特征树量
    bst_ulong num_features = 0;
    XGBoosterGetNumFeature(booster, &num_features);
    spdlog::info("特征树量: {}", num_features);

    ////////////////////////////////////////////////////////////////////////
    /// 预测
    bst_ulong out_len = 0;
    const float *out_result = nullptr;
    int num_print = 10;
    XGBoosterPredict(booster, data_test, 0, 0, 0, &out_len, &out_result);
    spdlog::info("y_pred:");
    for (int i = 0; i != num_print; ++i)
    {
        spdlog::info("{}", out_result[i]);
    }

    ////////////////////////////////////////////////////////////////////////
    /// 打印真实标签
    XGDMatrixGetFloatInfo(data_test, "label", &out_len, &out_result);
    spdlog::info("y_test:");
    for (int i = 0; i != num_print; ++i)
    {
        spdlog::info("{}", out_result[i]);
    }

    ////////////////////////////////////////////////////////////////////////
    /// 从Mat创建DMatrix
    const float values[] = {0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
                            1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
    DMatrixHandle data_matrix;
    XGDMatrixCreateFromMat(values, 1, 127, 0.0, &data_matrix);
    XGBoosterPredict(booster, data_matrix, 0, 0, 0, &out_len, &out_result);
    spdlog::info("out length: {}", out_len);
    for (int i = 0; i != out_len; ++i)
    {
        spdlog::info("result: {}", out_result[i]);
    }
    XGDMatrixFree(data_matrix);



    ////////////////////////////////////////////////////////////////////////
    /// 释放所有内存
    XGBoosterFree(booster);
    XGDMatrixFree(data_train);
    XGDMatrixFree(data_test);


    return 0;
}
