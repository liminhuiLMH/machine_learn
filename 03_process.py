from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd

print(
    """
    归一化处理
    公式：
          x - min            
    xx = --------     , xxx = xx(mx-mi)+mi
          max - min  
    max 和 min 取 x 这一列的最大值和最小值，mx,mi表示归一化的范围区间，默认是（1，0），所以，mx=1,mi=0
    
    归一化意义：特征的单位或者大小相差较大，或者某特征的方差相比其他的特征要大出几个数量级，容易影响（支配）目标结果，使得一些算法无法学习到其它的特征
    缺点：注意最大值最小值是变化的，另外，最大值与最小值非常容易受异常点影响，所以这种方法鲁棒性较差，只适合传统精确小数据场景。
    """
)
transfer = MinMaxScaler()

data = [{"milage": 40920, "Liters": 8.326976, "Consumtime": 0.953952, "target": 3},
        {"milage": 14488, "Liters": 7.153469, "Consumtime": 1.673904, "target": 2},
        {"milage": 26052, "Liters": 1.441871, "Consumtime": 0.805124, "target": 1},
        {"milage": 75136, "Liters": 13.147394, "Consumtime": 0.428964, "target": 1},
        {"milage": 48111, "Liters": 9.134528, "Consumtime": 0.728045, "target": 3},
        {"milage": 43757, "Liters": 7.882601, "Consumtime": 1.332446, "target": 3}]
data = pd.DataFrame(data)
print(data)
ret = transfer.fit_transform(data)
"""
target=3
   3-1
x= --- = 1 , xx=1(1-0)+0
   3-1
   
target=2
   2-1
x= --- = 0.5 , xx=1(1-0)+0
   3-1
   
Liters=8.326976
   8.326976-1.441871=6.885105
x= ---------------------------  =0.58819286, xx=x(1-0)+0
   13.147394-1.441871=11.705523
"""
print("最小值最大值归一化处理的结果：\n", ret)
print("====================================")
print("====================================")
print("====================================")
print(
    """
    标准化处理
    公式：
          x - mean            
    xx = ----------    
             q  
    mean 为 x 列的平均值，q 为 x 列的方差
    
    * 方差公式
       (n1-m)^2 + (n2-m)^2 + ... +(nn-m)^2
    q= -----------------------------------  , m 表示平均数，n 表示样本个数
                    n

    对于归一化来说：如果出现异常点，影响了最大值和最小值，那么结果显然会发生改变
    对于标准化来说：如果出现异常点，由于具有一定数据量，少量的异常点对于平均值的影响并不大，从而方差改变较小。
    意义：在已有样本足够多的情况下比较稳定，适合现代嘈杂大数据场景。
    """
)

standar_transfer = StandardScaler()
s_ret = standar_transfer.fit_transform(data)
print("标准化结果：\n", s_ret)
print("每一列特征的平均值：\n", standar_transfer.mean_)
print("每一列特征的方差：\n", standar_transfer.var_)
