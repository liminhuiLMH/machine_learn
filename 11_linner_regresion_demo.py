print("""
线性回归-预测波士顿房价
优化方法
* 正规方程：使用公式一次性计算出结果
* 梯度下降：不断的计算特征


* sklearn.linear_model.LinearRegression(fit_intercept=True)
    * 通过正规方程优化
    * fit_intercept：是否计算偏置
    * LinearRegression.coef_：回归系数
    * LinearRegression.intercept_：偏置
    
    
* sklearn.linear_model.SGDRegressor(loss="squared_loss", fit_intercept=True, learning_rate ='invscaling', eta0=0.01)
    * SGDRegressor类实现了随机梯度下降学习，它支持不同的loss函数和正则化惩罚项来拟合线性回归模型。
    * loss:损失类型
        * loss=”squared_loss”: 普通最小二乘法
    * fit_intercept：是否计算偏置
    * learning_rate : string, optional
        * 学习率填充
        * 'constant': eta = eta0
        * 'optimal': eta = 1.0 / (alpha * (t + t0)) [default]
        * 'invscaling': eta = eta0 / pow(t, power_t)
            * power_t=0.25:存在父类当中
        * 对于一个常数值的学习率来说，可以使用learning_rate=’constant’ ，并使用eta0来指定学习率。
    * SGDRegressor.coef_：回归系数
    * SGDRegressor.intercept_：偏置
    
-- 评估模型

* sklearn.metrics.mean_squared_error(y_true, y_pred)
    * 均方误差回归损失
    * y_true:真实值
    * y_pred:预测值
    * return:浮点数结果
""")
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
print("随便看几条：\n", x_train[:10])
# 数据集特征工程-标准化
# std = StandardScaler()
std = MinMaxScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

print("标准化后-随便看几条：\n", x_train[:10])

# 模型训练
print("""
第一种方式:正规方程

""")
estimator = LinearRegression()

estimator.fit(x_train, y_train)
print("""
得出模型
""")
print("权重：", estimator.coef_)
print("偏置：", estimator.intercept_)
y_predict = estimator.predict(x_test)
print("正规方程-预测结果：\n", y_test[:10] == y_predict[:10])
print("正规方程-均方误差：\n", mean_squared_error(y_test, y_predict))

print("""
第二种方式：梯度下降

""")
estimator_2 = SGDRegressor()
estimator_2.fit(x_train, y_train)

print("权重：", estimator_2.coef_)
print("偏置：", estimator_2.intercept_)
y_predict = estimator_2.predict(x_test)
print("梯度下降-预测结果：\n", y_test[:10] == y_predict[:10])
print("梯度下降-均方误差：\n", mean_squared_error(y_test, y_predict))

print("""
梯度下降            正规方程
需要选择学习率        不需要 
需要迭代求解         一次运算得出 
特征数量较大可以使用   需要计算方程，时间复杂度高O(n3) 


均方误差（Mean Squared Error，简称MSE）是衡量回归模型预测结果误差的常用指标。如果MSE返回的结果很大，可能说明模型的预测结果与真实结果之间存在较大的误差。下面列出一些可能导致MSE返回结果很大的原因：

数据量过大：如果数据量很大，那么MSE计算出来的误差平方和也会很大。这时需要考虑对数据进行采样或者降维等处理，以减少数据量。

数据分布不均匀：如果数据分布不均匀，那么MSE可能会被一些极端值或者离群点影响，导致结果很大。这时需要考虑对数据进行归一化、去除异常值等处理，以使数据更加平滑。

模型拟合不足：如果模型的拟合能力不足，那么MSE可能会返回很大的结果。这时需要考虑改进模型，增加特征、调整超参数等，以提高模型的拟合能力。

特征选择不当：如果特征选择不当，那么MSE可能会返回很大的结果。这时需要考虑重新选择特征或者采用更加合适的特征选择方法，以提高模型的泛化能力。

总之，MSE返回结果很大可能是由于数据量过大、数据分布不均匀、模型拟合不足或者特征选择不当等原因造成的。在应用MSE时，需要根据具体情况选择合适的处理方法，以获得更加准确的结果。
""")
