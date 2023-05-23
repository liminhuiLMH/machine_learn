print("""
随机森林对鸢尾花进行分类
""")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.tree import export_graphviz

# 导入数据集
iris = load_iris()
# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)
print("""
* class sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None, bootstrap=True, random_state=None, min_samples_split=2)
    * 随机森林分类器
    * n_estimators：integer，optional（default = 10）森林里的树木数量120,200,300,500,800,1200
    * criteria：string，可选（default =“gini”）分割特征的测量方法
    * max_depth：integer或None，可选（默认=无）树的最大深度 5,8,15,25,30
    * max_features="auto”,每个决策树的最大特征数量
        * If "auto", then max_features=sqrt(n_features).
        * If "sqrt", then max_features=sqrt(n_features) (same as "auto").
        * If "log2", then max_features=log2(n_features).
        * If None, then max_features=n_features.
    * bootstrap：boolean，optional（default = True）是否在构建树时使用放回抽样
    * min_samples_split:节点划分最少样本数
    * min_samples_leaf:叶子节点的最小样本数
* 超参数：n_estimator, max_depth, min_samples_split,min_samples_leaf

""")
# 训练模型
estimator = RandomForestClassifier()

estimator.fit(x_train, y_train)
# 模型评估
y_predict = estimator.predict(x_test)
print("评估结果：\n", y_test == y_predict)
print("准确率：\n", estimator.score(x_test, y_test))

# 模型调优
print("""
# 模型调优
""")
estimator_2 = RandomForestClassifier()
param_grid = {"max_depth": [1, 3, 5, 7, 9], "n_estimators": [1, 3, 5, 7, 9, 11, 13, 15]}
gcv = GridSearchCV(estimator_2, param_grid)
# 训练模型
gcv.fit(x_train, y_train)

# 模型评估
g_predict = gcv.predict(x_test)
print("评估结果：\n", y_test == g_predict)
print("准确率：\n", gcv.score(x_test, y_test))

# 训练验证集的结果
print("在交叉验证当中验证的最好结果：", gcv.best_score_)
print("gc选择了的模型K值是：", gcv.best_estimator_)
print("每次交叉验证的结果为：", gcv.cv_results_)
