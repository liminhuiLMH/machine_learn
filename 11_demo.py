from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

# 获取数据集
boston = fetch_california_housing()

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.3)
print("训练集大小：\n", x_train.shape)
print("训练集-随便看几条：\n", x_train[:10])
print("目标集-随便看几条：\n", y_train[:10])
# 数据集特征工程-标准化
# std = StandardScaler()
std = MinMaxScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

print("标准化后-随便看几条：\n", x_train[:10])

print("""
第二种方式：梯度下降

""")
estimator_2 = SGDRegressor()
estimator_2.fit(x_train, y_train)

print("权重：", estimator_2.coef_)
print("偏置：", estimator_2.intercept_)
y_predict = estimator_2.predict(x_test)
print("梯度下降-预测结果：\n", y_predict[:20])
print("梯度下降-预测结果比对：\n", y_test[:10] == y_predict[:10])
print("梯度下降-均方误差：\n", mean_squared_error(y_test, y_predict))
