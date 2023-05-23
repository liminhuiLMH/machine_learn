from sklearn.decomposition import PCA

print(
    """
    对数据进行PCA降维
    """)
data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]

"""
sklearn.decomposition.PCA(n_components=None)
将数据分解为较低维数空间
n_components:
* 小数：表示保留百分之多少的信息
* 整数：减少到多少特征
PCA.fit_transform(X) X:numpy array格式的数据[n_samples,n_features]
返回值：转换后指定维度的array
"""

transfer = PCA(n_components=0.9)

ret_01 = transfer.fit_transform(data)
print("保留90%的信息，降维结果为：\n", ret_01)

# 1、实例化PCA, 整数——指定降维到的维数
transfer2 = PCA(n_components=3)
# 2、调用fit_transform
data2 = transfer2.fit_transform(data)
print("降维到3维的结果：\n", data2)
