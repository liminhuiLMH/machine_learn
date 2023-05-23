import pandas as pd

# 创建数据集
data = {'gender': ['male', 'male', 'female', 'male', 'female', 'female', 'male', 'female'],
        'age': [20, 25, 30, 35, 40, 45, 50, 55],
        'income': ['low', 'high', 'medium', 'high', 'low', 'medium', 'low', 'high'],
        'purchase': ['yes', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes']}
df = pd.DataFrame(data)

# 计算交叉表
cross_table = pd.crosstab(df['gender'], df['income'])

# 计算列联表卡方检验
from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency(cross_table)

# 输出结果
print(
    """
    交叉分析（Contingency Analysis）是一种用于研究两个或多个变量之间关系的方法，通常用于探索因果关系、协方关系
    需要注意的是，交叉分析可以用于探索变量之间的关系，但不能确定因果关系
    """
)
print("原始数据：\n", df)
print("交叉表：\n", cross_table)
print('chi-squared statistic:', chi2)
print('p-value:', p)
