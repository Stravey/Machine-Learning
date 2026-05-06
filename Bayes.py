import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# 导入数据集
data = pd.read_csv('career_data.csv')

#