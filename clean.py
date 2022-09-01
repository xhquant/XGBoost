#!/usr/bin/env python3

import os
import subprocess


def remove_all(path_name):
    files = os.listdir(path_name)
    for file in files:
        subprocess.call(("rm", "-rf", path_name + "/" + file))


def remove():
    paths = ["build", "bin", "xgboost_src/lib"]
    for path in paths:
        remove_all(path)


def remove_xgboost_src():
    subprocess.call(("rm", "-rf", "xgboost_src/xgboost"))


def remove_pynb():
    files = os.listdir(".")
    for file in files:
        if file.startswith(".ipynb"):
            subprocess.call(("rm", "-rf", file))


if __name__ == "__main__":
    remove()
    remove_xgboost_src()
    remove_pynb()
