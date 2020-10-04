# 波士顿房价线性回归预测学习笔记

## 背景知识

**波士顿房价数据集**


波士顿房价数据集是内置在[scikit-learn](https://sklearn.org/index.html)python工具包中的小型数据集，用于练习与测试工具包的相关功能。

scikit-learn是基于python的一个开源且可商用的工具包，常用于数据挖掘和数据分析。

该数据集有506条数据，每条数据包含13个属性值，每个属性的含义如下：第一个纯大写词为属性名称，如CRIM。之后为该属性的含义
>
    CRIM per capita crime rate by town
    ZN proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS proportion of non-retail business acres per town
    CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    NOX nitric oxides concentration (parts per 10 million)
    RM average number of rooms per dwelling
    AGE proportion of owner-occupied units built prior to 1940
    DIS weighted distances to five Boston employment centres
    RAD index of accessibility to radial highways
    TAX full-value property-tax rate per $10,000
    PTRATIO pupil-teacher ratio by town
    B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    LSTAT % lower status of the population
    MEDV Median value of owner-occupied homes in $1000’s
>


**线性回归预测**

对一个二维的数据点集合进行拟合，寻找一个一次函数，使得y的预测值与实际值之差的平方和取最小值，该方法称为线性回归预测。若将y=ax+b中的a和b视为自变量，将平方和视为二元二次函数，则可以通过求极值点得到最小值点。

具体的数学证明过程可以参考[CSDN上这篇文档](https://blog.csdn.net/Android_xue/article/details/97614045?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param)

---

## 代码解释

该代码的实现，引用了许多工具包，如果没有相关的包，可以通过pip命令或者conda命令下载包。由于不同的实验可能需要不同版本的包，所以有必要用conda为项目建立单独的环境，来存放需要的包。当项目完成后，通过删除环境可以方便的将安装的包一并删除。相关的教程请自行搜索。

**引入包分析**

- **numpy**

    numpy是基于python的科学计算包
    
    ``import numpy as np``

- **sklearn**

    sklearn又叫做scikit-learn，是用于可预测数据的分析工具。该工具包含有一些样本数据集，本实验所使用的波士顿房价数据即来自于该包。

    ```py
        from sklearn import datasets    #引入数据集
        from sklearn.linear_model import LinearRegression   #引入线性回归预测工具 
        v_housing=datasets.load_boston()    #载入数据集中的波士顿房价数据
        print(v_housing.data.shape) #输出数据的结构
    ```
    - **boston**

        波士顿房价数据集的调用方式在sklearn. datasets中定义
        其数据集格式如下
        ```py
        mydata=sklearn.datasets.load_boston() #引入波士顿数据集
        mydata.data     #波士顿数据，二维矩阵
        mydata.target   #波士顿数据集的最后一列
        mydata.feature  #属性名称列表
        ```
    - **LinearRegression**

        线性回归预测的调用方式在sklearn. linear_model. LinearRegression()中定义
        其[官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)中有较多数学类英语名词，故不作分析，欢迎有能力者提交PR补充。
- **pandas**

    是一款基于Numpy的分析结构化数据的工具集，本实验使用了包中的DataFrame模块，这个模块以表格的方式存储数据，方便展示以及矩阵运算。

    ```python
        v_bos=pd.DataFrame(v_housing.data)  #载入波士顿数据
        print(v_bos.head(5))    #展示前五行数据
        print(v_bos)            #展示所有数据
    ```
    该模块在本实验中没有影响，只是用于展示数据信息

- **matplotlib**
    
    [matplotlib](https://matplotlib.org/)是一款详尽的绘图python库，而[pylot](https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py)(以下简称plt)定义的函数则提供了MatLab式的接口。

    plt.plot( ) 用于设定数据点的坐标，当只给出一维数组时，将其默认为y坐标，而x坐标则自动设置为0, 1, ..若要具体定义每个点，可以通过plt.plot([ 第一维坐标序列 ],[ 第二维坐标序列 ], ... ,[ 第n维坐标序列 ]，"格式字符串")。其中，格式字符串用来设定数据显示的具体格式，省略后默认为'b-'的形式，即蓝色实线 ( 具体的格式参考MatlLab ) 。

    plt.axis( ) 用于设定坐标轴的取值范围，用[ Axis1Start, Axits1End [ , AxisNextStart,AxisNextEnd [ ... ] ] ]表示
    比如[ 0, 5, 0, 10]表示x轴为0-5，y轴为0-10。

    pyplot.xlabel('xname') 用于设定轴名称。

    plt.show( ) 用于显示绘制好的图。

    plt.scatter(x,y) 绘制散点图

    关于方法详细的使用方式请见官方文档

## 源代码

```py
import pandas as pd #只有被注释的第一段代码需要使用
import numpy as np
from sklearn import datasets 
from sklearn.linear_model import LinearRegression 
from matplotlib import pyplot as plt 

plt.rcParams['font.sans-serif']=['SimHei']  #使用支持中文的字体
v_housing=datasets.load_boston()            #导入波士顿房价数据集

'''这些代码都是用于其他目的，可自行取消注释并使用

#本段代码用于展示数据
v_bos=pd.DataFrame(v_housing.data)  #pd库能使数据以更加清晰的方式存储，方便显示
print(v_bos.head(5))    #打印前五行数据
------------------------------------------------------------------------------

#本段代码用散点图展示不同特征与价格之间的关系
x_data=v_housing.data       #导入全部数据
y_data=v_housing.target     #导入价格数据，一维数组
v_namedata=v_housing.feature_names  #导入特征的名称

##分别按13种属性值绘图，初步判断关系
for i in range(13):
    plt.subplot(7,2,i+1)    #设置子图参数
    plt.scatter(x_data[:,i],y_data,s=20)    #散点图，填入第i列数据值与其价格
    plt.title(v_namedata[i])    #标题取对应特征的名称
    plt.show()
-------------------------------------------------------------------------
'''

x=v_housing.data[:,np.newaxis,3]    #取第三个特征的x值，形成501行一列的矩阵
y=v_housing.target          #target指数据集的最后一列
lm=LinearRegression()       #引用线性回归类
lm.fit(x,y)                 #填入数据

print('方程的确定性系数R的平方：',lm.score(x,y))
print('线性回归算法w值:',lm.coef_)
print('线性回归算法b值：',lm.intercept_)
#绘图
plt.scatter(x,y,color='blue')   #实际数据用散点
plt.plot(x,lm.predict(x),color='green',linewidth=6) #预测结果用直线
plt.ylabel('房屋价格')
plt.title('线性回归住宅平均房间数RM与房屋价格PRICE的关系')
plt.show()

```

