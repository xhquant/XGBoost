#!/usr/bin/env python3

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import xgboost
import lightgbm
import pandas
import numpy

if __name__ == "__main__":
    # X, y = make_regression(n_samples=10000, n_features=10, n_informative=4, random_state=999)
    df = pandas.read_csv("../dataset/000651.dat", index_col=0, sep="\s+")
    X = df.iloc[:, 9:]
    y = df["opportunity"]

    x_train, x_, y_train, y_ = train_test_split(X, y, test_size=0.04, random_state=999)
    x_test, x_valid, y_test, y_valid = train_test_split(x_, y_, test_size=0.5, random_state=999)

    # 建立booster
    xgb_model = xgboost.XGBRegressor(max_depth=6, max_leaves=50, learning_rate=0.01, random_state=999)
    lgb_model = lightgbm.LGBMRegressor(max_depth=6, num_leaves=50, learning_rate=0.01, random_state=999)

    # 训练模型
    xgb_model.fit(X=x_train, y=y_train, eval_set=[(x_test, y_test)])
    lgb_model.fit(X=x_train, y=y_train, eval_set=[(x_test, y_test)])

    # 预测
    xgb_y_pred = xgb_model.predict(X=x_valid)
    lgb_y_pred = lgb_model.predict(X=x_valid)
    print(numpy.corrcoef(xgb_y_pred, y_test.values))
    # for t, xgb_p, lgb_p in zip(y_test, xgb_y_pred, lgb_y_pred):
    #     print("%.2f %.2f %.2f" % (t, xgb_p, lgb_p))

    # plt.plot(xgb_y_pred * 2, color="red", linewidth=2, label="xgb")
    # plt.plot(lgb_y_pred * 2, color="blue", linewidth=2, label="lgb")
    plt.plot((xgb_y_pred + lgb_y_pred), color="green", linewidth=2, label="mean")
    plt.plot(y_test.values, color="black", linewidth=2, label="test")
    plt.legend()

    # 获取参数
    params = xgb_model.get_params()
    print(params)

    # 获取特征数目
    n_features = xgb_model.n_features_in_
    print(n_features)

    # best score
    best_iteration = xgb_model.best_iteration
    print(best_iteration)

    # 获取objective
    objective_str = xgb_model.objective
    print(objective_str)

    # 获取booster对象
    booster_model = xgb_model.get_booster()
    print(booster_model)

    # 获取评价结果
    eval_result = xgb_model.evals_result_
    print(eval_result)

    # 获取特征重要性
    feature_importance = xgb_model.feature_importances_
    print(feature_importance)

    # 画特征重要性
    xgboost.plot_importance(xgb_model.get_booster())
    plt.show()
