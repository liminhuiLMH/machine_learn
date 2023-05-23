from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1、导入数据集
iris = load_iris()
# 2、特征工程
## 数据集划分
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
print("特征值训练集大小：", x_train.shape)
print("特征值测试集大小：", x_test.shape)
print("目标值训练集大小：", y_train.shape)
print("目标值测试集大小：", y_test.shape)

## 标准化
std = StandardScaler()
x_train = std.fit_transform(x_train)
print("训练集标准化结果：\n", x_train)

## 这里调用的是 transform 方法，因为测试集是为了验证训练集的结果，所以应该使用训练集的平均值和标准差，而训练集已经调用过 fit 方法
x_test = std.transform(x_test)
print("测试集标准化结果：\n", x_test)

# 3、算法的输入训练预测
knn = KNeighborsClassifier()

## 调用fit() ,fit 方法用于训练模型，即学习模型的参数和权重
## 输入特征值训练数据集和目标值训练数据集
knn.fit(x_train, y_train)

# 预测测试数据集，得出准确率，即，输入测试数据，评估训练模型的效果如何
y_predict = knn.predict(x_test)

print("比对方法1：比对目标测试集和预测结果：\n", y_test == y_predict)

print("预测测试集类别：", y_predict)

print("准确率为：", knn.score(x_test, y_test))

print("""
使用网格搜索和交叉验证找到合适的参数
模型选择与调优
sklearn.model_selection.GridSearchCV(estimator, param_grid=None,cv=None)
对估计器的指定参数值进行详尽搜索
* estimator：估计器对象
* param_grid：估计器参数(dict){“n_neighbors”:[1,3,5]}
* cv：指定几折交叉验证

* fit：输入训练数据
* score：准确率

结果分析：
* bestscore:在交叉验证中验证的最好结果_
* bestestimator：最好的参数模型
* cvresults:每次交叉验证后的验证集准确率结果和训练集准确率结果
""")
# 使用网格搜索和交叉验证找到合适的参数
param = {"n_neighbors": [1, 3, 5, 7, 9, 11]}

gc = GridSearchCV(knn, param_grid=param, cv=2)

gc.fit(x_train, y_train)

print("选择了某个模型测试集当中预测的准确率为：", gc.score(x_test, y_test))

# 训练验证集的结果
print("在交叉验证当中验证的最好结果：", gc.best_score_)
print("gc选择了的模型K值是：", gc.best_estimator_)
print("每次交叉验证的结果为：", gc.cv_results_)
