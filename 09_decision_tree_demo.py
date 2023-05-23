from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

print("""
鸢尾花决策树分类
""")
# 获取数据集
iris = load_iris()

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

print("""
class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, max_depth=None,random_state=None)
决策树分类器
criterion:默认是’gini’系数，也可以选择信息增益的熵’entropy’
max_depth:树的深度大小
random_state:随机数种子
其中会有些超参数：max_depth:树的深度大小

""")
# 模型训练
estimator = DecisionTreeClassifier()

# 模型生成，输入特征训练集和目标测试集
estimator.fit(x_train, y_train)

# 模型评估，输入特征测试集，生成目标集
y_predict = estimator.predict(x_test)
print("模型预估结果：\n", y_predict)
print("和真实值比对结果：\n", y_test == y_predict)

# 模型准确率
score = estimator.score(x_test, y_test)
print("模型准确率：\n", score)

print("""
1、sklearn.tree.export_graphviz() 该函数能够导出DOT格式
tree.export_graphviz(estimator,out_file='tree.dot’,feature_names=[‘’,’’])
* 生成的内容可以复制到这个网站去解析：http://webgraphviz.com/
2、工具:(能够将dot文件转换为pdf、png)
安装graphviz
ubuntu:sudo apt-get install graphviz Mac:brew install graphviz
3、运行命令
然后我们运行这个命令
dot -Tpng tree.dot -o tree.png
""")
export_graphviz(estimator, out_file="tree.dot", feature_names=iris.feature_names)
