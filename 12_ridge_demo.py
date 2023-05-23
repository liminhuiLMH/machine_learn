print("""
* sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True,solver="auto", normalize=False)
    * 具有l2正则化的线性回归
    * alpha:正则化力度，也叫 λ
        * λ取值：0~1 1~10
    * solver:会根据数据自动选择优化方法
        * sag:如果数据集、特征都比较大，选择该随机梯度下降优化
    * normalize:数据是否进行标准化
        * normalize=False:可以在fit之前调用preprocessing.StandardScaler标准化数据
    * Ridge.coef_:回归权重
    * Ridge.intercept_:回归偏置
""")

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

# 导入数据集
house = fetch_california_housing()
# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(house.data, house.target, test_size=0.3)
# 模型训练
print("""
* 正则化力度越大，权重系数会越小
* 正则化力度越小，权重系数会越大
alpha=0
[ 4.39786741e-01  9.48043341e-03 -1.08251629e-01  6.53915969e-01
 -2.24443656e-06 -3.75539554e-03 -4.22468411e-01 -4.34886758e-01]
 
alpha=1
 [ 4.39786741e-01  9.48043341e-03 -1.08251629e-01  6.53915969e-01
 -2.24443656e-06 -3.75539554e-03 -4.22468411e-01 -4.34886758e-01]
""")
estimator = Ridge(alpha=1000, max_iter=10000)

estimator.fit(x_train, y_train)

print("权重：", estimator.coef_)
print("偏置：", estimator.intercept_)
print("均方误差：", mean_squared_error(y_test, estimator.predict(x_test)))
