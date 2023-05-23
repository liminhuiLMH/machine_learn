from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from scipy.stats import pearsonr

data = pd.read_csv("factor_returns.csv")
data = data.iloc[:, 1:9]
print(data)

print("""
特征降维：数据中包含冗余或无关变量（或称特征、属性、指标等），旨在从原有特征中找出主要特征
方法：
* 特征选择
    * Filter(过滤式)：主要探究特征本身特点、特征与特征和目标值之间关联
        * 方差选择法：低方差特征过滤
        * 相关系数：皮尔逊系数
    * Embedded (嵌入式)：算法自动选择特征（特征与目标值之间的关联）
        * 决策树:信息熵、信息增益
        * 正则化：L1、L2
        * 深度学习：卷积等
* 主成分分析（可以理解一种特征提取的方式）

删除所有低方差特征
Variance.fit_transform(X)
X:numpy array格式的数据[n_samples,n_features]
返回值：训练集差异低于threshold的特征将被删除。默认值是保留所有非零方差特征，即删除所有样本中具有相同值的特征。
""")
transfer = VarianceThreshold(threshold=1)

ret = transfer.fit_transform(data)
"""
[
 [ 5.95720000e+00  1.18180000e+00  8.52525509e+10  8.00800000e-01 1.49403000e+01  1.21144486e+12  2.07014010e+10]
 [ 7.02890000e+00  1.58800000e+00  8.41133582e+10  1.64630000e+00 7.86560000e+00  3.00252062e+11  2.93083692e+10]
 [-2.62746100e+02  7.00030000e+00  5.17045520e+08 -5.67800000e-01 -5.94300000e-01  7.70517753e+08  1.16798290e+07]
 [ 1.64760000e+01  3.71460000e+00  1.96804560e+10  5.60360000e+00 1.46170000e+01  2.80091592e+10  9.18938688e+09]
 [ 1.25878000e+01  2.56160000e+00  4.17272149e+10  2.87290000e+00 1.09097000e+01  8.12473804e+10  8.95145349e+09]
]
"""
print("删除低方差特征的结果：\n", ret)
print("形状：\n", ret.shape)

print("""
相关系数计算
相关系数的值介于–1与+1之间，即–1≤ r ≤+1。其性质如下：

当r>0时，表示两变量正相关，r<0时，两变量为负相关
当|r|=1时，表示两变量为完全相关，当r=0时，表示两变量间无相关关系
当0<|r|<1时，表示两变量存在一定程度的相关。且|r|越接近1，两变量间线性关系越密切；|r|越接近于0，表示两变量的线性相关越弱
一般可按三级划分：|r|<0.4为低度相关；0.4≤|r|<0.7为显著性相关；0.7≤|r|<1为高度线性相关
""")
data = pd.read_csv("factor_returns.csv")

factor = ['pe_ratio', 'pb_ratio', 'market_cap', 'return_on_asset_net_profit', 'du_return_on_equity', 'ev',
          'earnings_per_share', 'revenue', 'total_expense']

for i in range(len(factor)):
    for j in range(i, len(factor) - 1):
        print(
            "指标%s与指标%s之间的相关性大小为%f" % (factor[i], factor[j + 1], pearsonr(data[factor[i]], data[factor[j + 1]])[0]))

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 8), dpi=100)
plt.scatter(data['revenue'], data['total_expense'])
plt.show()