from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = load_iris()

# 创建 PCA 模型，将数据降至 2 维
pca = PCA(n_components=2)

# 使用 PCA 模型进行降维
X_pca = pca.fit_transform(iris.data)
print("降维后结果：\n", X_pca)

# 绘制降维后的数据
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
