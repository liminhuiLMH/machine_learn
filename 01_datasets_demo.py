from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
print("返回值：\n", iris)
print("数据集：\n", iris.data)
print("大小：\n", iris.data.shape)
print("目标值：\n", iris.target)
print("大小：\n", iris.target.shape)
print("特征名称：\n", iris.feature_names)
print("目标名称：\n", iris.target_names)
print("描述：\n", iris.DESCR)

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

print("特征值训练集：\n", x_train)
print("特征值测试集：\n", x_test)
print("目标值训练集：\n", y_train)
print("目标值测试集：\n", y_test)
print("x_train:\n", x_train.shape)
# 随机数种子
x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target, random_state=6)
x_train2, x_test2, y_train2, y_test2 = train_test_split(iris.data, iris.target, random_state=6)
print("如果随机数种子不一致：\n", x_train == x_train1)
print("如果随机数种子一致：\n", x_train1 == x_train2)
