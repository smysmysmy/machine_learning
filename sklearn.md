# <p align="center">sklearn </p>

<p align="center"> 2019年7月31日 </p>

### <p align="left"> sklearn概述 </p>
>**估计量（estimator）**：基于数据集而估计某些参数。
**可调用的方法：** fit()方法。对于非监督学习只需要一个数据集作为参数，而监督学习算法需要两个数据集，其中第二个数据集包含标签。所有估计量的超参数都可以通过公共实例变量直接访问，估计量学习到的参数可以通过公共实例变量添加下划线后缀访问。
**转换量（transformer）**：与一些估计量相关，转换数据集。
**可调用的方法：** transform()方法和fit_transform()方法。两者均需要被转换的数据集作为参数。返回的是经过转换的数据集。fit_transform()一般等同于fit()+transform()。
**预测量（predictor）**：根据数据集进行预测等操作。
**可调用的方法：** predict()方法和score()方法。分别实现预测和评价。

### <p align="left"> Pandas基本用法 </p>
1. Dataframe:
```
# 绘制所有数据的直方图
dataset.hist()
# 显示所有数据的相关统计值
dataset.describe()
# 以dataframe格式显示前五行数据或后五行数据
dataset.head()/tail()
# 将dataframe格式的数据转化为ndarray格式
dataset.values
# 设置索引
dataset.set_index()
# 当dataset满足condition时保持不变，否则以num替换dataset中不满足的值(inplace=True)
dataset.where(condition,num,inplace=True/False)
```
2. Series:

与dataframe类似。

### <p align="left"> 训练集与测试集划分 </p>
1. 纯随机采样
```
# data可以为dataframe格式或ndarray格式，返回值类型对应data类型
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
```
2. 分层采样（数据集较小时避免偏差）
```
# data可以为dataframe格式或ndarray格式，返回值类型对应data类型
# n_splits:训练集测试集划分次数 
# test_size:测试集大小 
# random_state:随机种子
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(iris, iris["virginica"]):
    strat_train_set = iris.loc[train_index]
    strat_test_set = iris.loc[test_index]
```

### <p align="left"> 数据缺失的处理 </p>

1. 抛弃该特征


2. 对缺省值进行填充
```
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(data)
X = imputer.transform(data)
```

### <p align="left"> 非数字特征映射为数字特征 </p>
1. 直接映射编码
```
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
# encoder.fit(data["attribute"])
# data["attribute"] = encoder.transform(data["attribute"])
data["attribute"] = encoder.fit_transform(data["attribute"])
# encoder.classes_:查看映射表
# encoder.inverse_transform(data["attribute"]):将编码映射回原值
```
存在问题:人为引入相似度，文本上不相似的两个取值转化为数字后可能被认为更相似
2. Onehot编码
```
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(data["attribute"].values.reshape(-1,1)).toarray()
# toarray()返回密集矩阵
```
存在问题：密集矩阵过于稀疏浪费存储空间

### <p align="left"> 自定义转换量 </p>
1. 自定义转换量：

自定义转换量步骤：
* 创建一个类并执行三个方法：fit()（返回self），transform()，和fit_transform()
* 添加TransformerMixin作为基类，获得fit_transform方法
* 添加BaseEstimator作为基类（且构造器中避免使用*args和**kargs）可得到两个额外的方法（get_params()和set_params()），可以进行超参数自动微调

获取数据（将dataframe型数据转化为ndarray型）
```
# 输入:
# __init__(attributes(list)):包含所有数值属性的列表
# transform(x(dataframe)):dataframe格式数据
# transform()输出:将dataframe格式的数值数据转化为ndarray
from sklearn.base import BaseEstimator, TransformerMixin

class GetData(BaseEstimator, TransformerMixin):
    # 初始化attributes
    def __init__(self, attributes):
        self.attributes=attributes
    # 返回对象本身
    def fit(self, x, y=None):
        return self
    # 将传入的属性attributes有关的数据转化为ndarray型，如果attributes为空则返回空
    def transform(self, x, y=None):
        if x.empty:
            return 0
        else:
            return x[self.attributes].values
```
增加额外属性
```
# 输入:
# __init__(add_attribute(boolean))
# transform(x(ndarray))
# transform()输出:原数据和新变量组合
from sklearn.base import BaseEstimator, TransformerMixin

class AddAttribute(BaseEstimator, TransformerMixin):
    # 初始化add_attribute（默认为否）
    def __init__(self, add_attribute=False):
        self.add_attribute=add_attribute
    # 
    def fit(self, x, y=None):
        return self
    # 将传入的数据
    def transform(self, x, y=None):
        if self.add_attribute:
            new_variable=[0,0,0,0,0]
            return np.c_[x, new_variable]
        else:
            return x
```
文本属性转化
```
# 默认返回密集矩阵
from sklearn.preprocessing import OneHotEncoder

class MyOneHotEncoder(BaseEstimator, TransformerMixin):
    #
    def __init__(self,sparse=False):
        self.encoder = OneHotEncoder(sparse=sparse)
    #
    def fit(self, x, y=None):
        if  x.size:
            self.encoder.fit(x)
        return self
    # 
    def transform(self, x, y=None):
        if not x.size:
            return np.zeros(x.shape)
        else:
            return self.encoder.transform(x)
```
2. 流水线使用：
```
from sklearn.pipeline import Pipeline
# 对数字特征进行处理
num_attributes = [item for item in attributes if data[item].dtype=='float64']
char_attributes = [item for item in attributes if data[item].dtype!='float64']
num_pipeline = Pipeline([
    ('get_num_data', GetData(num_attributes)),
    ('imputer', SimpleImputer(strategy='median')),
    ('add_attributes', AddAttribute(add_attribute=True)),
    ('std_scale', StandardScaler()),
])
```

```
# 对非数字特征进行处理
char_pipeline = Pipeline([
    ('get_string_data', GetData(char_attributes)),
    ('encode', MyOneHotEncoder()),
])
```

```
# 合并输出
from sklearn.pipeline import FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("char_pipeline", char_pipeline),
])
```
以上代码集成在[my_data_process.py](https://github.com/smysmysmy/machine_learning/blob/master/my_data_process.py)
function data_process:
输入：待转化的数据(dataframe格式)
输出：以中值填入缺省值、标准化并将文本特征OneHot编码后的数据矩阵


### <p align="left"> 训练 </p>
```
# 以LinearSVC为例
from sklearn.svm import LinearSVC
# 可传入参数：
loss：损失函数
C：惩罚项系数

```
寻找最优参数
```
from sklearn.model_selection import GridSearchCV

model = svm.SVR(kernel='rbf')
c_can = np.logspace(-2, 2, 10)
gamma_can = np.logspace(-2, 2, 10)
svr = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
```
>GridSearchCV()属性及可调用方法:
cv_results_：包含了'mean_test_score'(验证集平均得分)，'rank_test_score'(验证集得分排名)，'params'(dict形式存储所有待选params的组合)
best_params_ : dict：最佳参数组合
best_score_ : cv_results_属性中，'mean_test_score'里面的最高分。验证集得到的最高分数
best_estimator_ : 得到打分最高的超参组合对应的estimator
fit()/predict():用网格搜索得到的最佳超参所构建的estimator对数据集进行fit、predict
get_params():模型更多的参数

### <p align="left"> 评价 </p>
交叉验证
```
# K折交叉验证
from sklearn.model_selection import cross_val_score
# estimator(第一项)为模型，scoring为评价指标，cv为K折折数
scores = cross_val_score(svc,X_train,y_train,scoring="neg_mean_squared_error",cv=10)
```

测试集
```
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
# 利用X_train的均值与方差标准化X_test
X_test = scale.transform(X_test)
```

##### <p align="left"> 其它 </p>
1. 绘制超平面(两个特征)
```
svc = LinearSVC(C=1.0)
prediction = svc.fit(X_train, y_train)
# 创建超平面(根据SVC的参数，求出直线的斜率与截距)
w = svc.coef_[0]
a = - w[0]/w[1]
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (svc.intercept_[0])/w[1]
# Plot the hyperplane
plt.plot(xx, yy)
plt.axis("off")
plt.show()
```
2. 打乱训练集
```
# 打乱训练集
np.random.seed(42)
shuffle_index=np.random.permutation(100)
X_train, y_train=X_train[shuffle_index],y_train[shuffle_index]
```

