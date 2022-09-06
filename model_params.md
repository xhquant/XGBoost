## scikit-learn版本参数

### 构造函数参数

max_depth: 基学习器参数。

max_leaves: 最大叶子数，0表示没有限制。

max_bin: 每个特征具有的最大分bin数。

grow_policy: 树的生长策略。

learning_rate: 学习率。

n_estimators: 基学习器最大树量。

verbosity: 日志的详细程度。

objective: 目标函数。("multi:softmax", "multi:softprob", "reg:squarederror", "reg:squaredlogerror", "binary:logistic", "reg:linear")

booster: 基学习器类型。("gplinear", "gbtree", "dart")

n_jobs: 线程数量。

gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree。

subsample: 行采样率。

sampling_method: 采样方法。("uniform", "gradient_based")

colsample_bytree: 列采样率。

reg_alpha: L1 regularization term on weights。

reg_lambda: L2 regularization term on weights。

base_score: 0.5。

random_state: 随机状态。

eval_metric: 评价矩阵。("rmse", "mae", "logloss", "error", "merror", "mlogloss", "auc", "map")

early_stopping_rounds: 早期停止轮数。

### 拟合fit函数参数

X: 特征数据。

y: 标签数据

eval_set: 评价数据集。

