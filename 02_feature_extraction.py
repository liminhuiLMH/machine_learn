from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import math

"""
字典数据类型特征提取
"""
print("""
字典数据类型特征提取
""")
data = [{'city': '北京', 'temperature': 100},
        {'city': '上海', 'temperature': 60},
        {'city': '深圳', 'temperature': 30}]

transfer_0 = DictVectorizer(sparse=True)

ret_0 = transfer_0.fit_transform(data)

# 转换之前的数据格式
print(transfer_0.inverse_transform(ret_0))

# 特征名称
print(transfer_0.get_feature_names_out())
print(ret_0)
print("================== 稀疏矩阵结果")
transfer = DictVectorizer(sparse=False)

ret = transfer.fit_transform(data)
print(transfer.get_feature_names_out())
# ”one-hot“编码：
print(ret)

"""
文本数据类型特征提取
"""
print("""
文本数据类型特征提取
""")
# data = ["life is short,i like like python", "life is too long,i dislike python"]
# fixme 中文如何处理？
# 英文单词是通过空格进行分割的，所以这里使用 jieba 分割中文句子，然后通过空格分割
data = ["人生苦短喜欢，我喜欢Python", "生活太长久，我不喜欢Python"]
print("jieba 分词器结果：\n")
text_list = []
for str in data:
    text_list.append(" ".join(list(jieba.cut(str))))
print(text_list)
# 使用 jieba 分词器
# 实例化
# text_transfer = CountVectorizer(sparse=False)
text_transfer = CountVectorizer()
# 特征提取和转换
# text_ret_0 = text_transfer.fit_transform(data)
text_ret_0 = text_transfer.fit_transform(text_list)
print("特征提取原始结果：\n", text_ret_0)
# 特征名称
print(text_transfer.get_feature_names_out())
print("特征提取结果")
print(text_ret_0.toarray())

"""
TfIdf 文本数据类型特征提取
"""
print("""
TfIdf 文本数据类型特征提取
Tf（term frequency，tf）:词频，指的是某一个给定的词语在该文件中出现的频率，例如：假如一篇文件的总词语数是100个，「非常」出现过5次，那么 td=5/100=0.05
Idf(inverse document frequency):逆向文档频率，是一个词语普遍重要性的度量。某一特定词语的idf，可以[由总文件数目除以包含该词语之文件的数目，
再将得到的商取以 10 为底的对数得到],例如：包含「非常」的文件有100个，总文件有1000个，那么idf=lg(1000/100)=1

Tfidf=0.05*1=0.05
""")

tfidf_transfer = TfidfVectorizer(stop_words=["太长久"])
tfidf_ret = tfidf_transfer.fit_transform(text_list)
print("原始结果：\n", tfidf_ret)
print("特征值：\n", tfidf_transfer.get_feature_names_out())
"""
['人生 苦短 喜欢 ， 我 喜欢 Python', '生活 太长久 ， 我 不 喜欢 Python']

['python' '人生' '喜欢' '太长久' '生活' '苦短']
[[1 1 2 0 0 1]
 [1 0 1 1 1 0]]
 
 转换结果：
 [[0.33425073 0.46977774 0.66850146 0.         0.         0.46977774]
 [0.40993715 0.         0.40993715 0.57615236 0.57615236 0.        ]]
 
 
「人生」
tf=1/7
idf=lg(2/1)=0

「生活」
tf=1
idf=lg(2/1)=0


"""
print(math.log10(1))
print("转换结果：\n", tfidf_ret.toarray())
