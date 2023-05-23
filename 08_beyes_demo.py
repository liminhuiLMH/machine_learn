print("""
朴素贝叶斯算法

一些概率知识：
* 联合概率：多个事情同时发生的概率，记做 P(A,B)
    * P(A,B)=P(A)P(B)
* 条件概率：在 B 发生的情况下 A 发生的概率，记做：P(A|B)
    * P(A1,A2|B)=P(A1|B)P(A2|B)
公式：
            P(w|c)P(c)
P(c|w) = ---------------
                P(w)
- w ： 特征词频
- c ： 文章类别

在文章分类中可以这么理解

                    P(F1,F2...|c)P(c)
P(C|F1,F2...) = ---------------
                    P(F1,F2...)
                    
公式分为三个部分：
* P(C)：每个文档类别的概率(某文档类别数／总文档数量)
* P(W│C)：给定类别下特征（被预测文档中出现的词）的概率
    * 计算方法：P(F1│C)=Ni/N （训练文档中去计算）
        * Ni为该F1词在C类别所有文档中出现的次数
        * N为所属类别C下的文档所有词出现的次数和
* P(F1,F2,…) 预测文档中每个词的概率

""")

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# * 分割数据集
# news_data = load_files("20news-bydate")
news_data = fetch_20newsgroups(subset="all")
print("特征集大小：\n", len(news_data.data))
print("目标集大小：\n", len(news_data.target))

# 特征训练集，特征测试集，目标训练集，目标训练集
x_train, x_test, y_train, y_test = train_test_split(news_data.data, news_data.target)
# * tfidf进行的特征抽取
transfer = TfidfVectorizer()
# transfer = CountVectorizer()

x_train = transfer.fit_transform(x_train)
print("特征值：\n", transfer.get_feature_names_out())

# 这是用测试集的 fit 参数
x_test = transfer.transform(x_test)

# * 朴素贝叶斯预测
print("""
sklearn.naive_bayes.MultinomialNB(alpha = 1.0)
朴素贝叶斯分类
alpha：拉普拉斯平滑系数
""")
# estimator估计器流程
mlb = MultinomialNB(alpha=1)

# 模型生成
mlb.fit(x_train, y_train)

# 模型评估
predict = mlb.predict(x_test)
print("预测结果：\n", predict[:10])
print("真实结果：\n", y_test[:10])
# 精准度评估

score = mlb.score(x_test, y_test)
print("精准度:\n", score)
