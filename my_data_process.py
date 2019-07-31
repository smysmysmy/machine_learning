import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# 获取数据
class GetData(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes=attributes

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        if x.empty:
            return 0
        else:
            return x[self.attributes].values

# 增加额外特征
class AddAttribute(BaseEstimator, TransformerMixin):
    def __init__(self, add_attribute=False):
        self.add_attribute=add_attribute

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        if self.add_attribute:
            new_variable=[0,0,0,0,0]
            return np.c_[x, new_variable]
        else:
            return x

# 文本特征OneHot编码
class MyOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,sparse=False):
        self.encoder = OneHotEncoder(sparse=sparse)
    def fit(self, x, y=None):
        if  x.size:
            self.encoder.fit(x)
        return self
    def transform(self, x, y=None):
        if not x.size:
            return np.zeros(x.shape)
        else:
            return self.encoder.transform(x)

def data_process(data):
    # 获取所有特征
    all_attributes = list(data)
    # 提取数字特征(float64)
    num_attributes = [item for item in all_attributes if data[item].dtype == 'float64']
    # 提取文本特征
    char_attributes = [item for item in all_attributes if data[item].dtype != 'float64']
    # 处理数值数据
    num_pipeline = Pipeline([
        ('get_num_data', GetData(num_attributes)),
        ('impute', SimpleImputer(strategy='median')),
        ('add_attributes', AddAttribute(add_attribute=True)),
        ('std_scale', StandardScaler()),
    ])
    # 处理非数值数据
    char_pipeline = Pipeline([
        ('get_string_data', GetData(char_attributes)),
        ('encode', MyOneHotEncoder()),
    ])
    # 合并输出
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("char_pipeline", char_pipeline),
    ])
    return full_pipeline.fit_transform(data)
