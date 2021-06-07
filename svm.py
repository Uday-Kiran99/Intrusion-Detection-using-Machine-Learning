import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

df = pd.read_csv("network_dataset.csv")


df.head()
print(df.head())
df.shape
print(df.shape)
df.isnull().sum()
print(df.isnull().sum())
num_cols = df._get_numeric_data().columns
cate_cols = list(set(df.columns) - set(num_cols))

cate_cols.remove('class')

cate_cols

print(cate_cols)

df = df.dropna('columns')  # drop columns with NaN

df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values

corr = df.corr()



# This variable is highly correlated with num_compromised and should be ignored for analysis.
# (Correlation = 0.9938277978738366)
df.drop('num_root', axis=1, inplace=True)

# This variable is highly correlated with serror_rate and should be ignored for analysis.
# (Correlation = 0.9983615072725952)
df.drop('srv_serror_rate', axis=1, inplace=True)

# This variable is highly correlated with rerror_rate and should be ignored for analysis.
# (Correlation = 0.9947309539817937)
df.drop('srv_rerror_rate', axis=1, inplace=True)

# This variable is highly correlated with srv_serror_rate and should be ignored for analysis.
# (Correlation = 0.9993041091850098)
df.drop('dst_host_srv_serror_rate', axis=1, inplace=True)

# This variable is highly correlated with rerror_rate and should be ignored for analysis.
# (Correlation = 0.9869947924956001)
df.drop('dst_host_serror_rate', axis=1, inplace=True)

# This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
# (Correlation = 0.9821663427308375)
df.drop('dst_host_rerror_rate', axis=1, inplace=True)

# This variable is highly correlated with rerror_rate and should be ignored for analysis.
# (Correlation = 0.9851995540751249)
df.drop('dst_host_srv_rerror_rate', axis=1, inplace=True)

# This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
# (Correlation = 0.9865705438845669)
df.drop('dst_host_same_srv_rate', axis=1, inplace=True)
# protocol_type feature mapping
pmap = {'icmp':0, 'tcp':1, 'udp':2}
df['protocol_type'] = df['protocol_type'].map(pmap)
# flag feature mapping
fmap = {'SF':0, 'S0':1, 'REJ':2, 'RSTR':3, 'RSTO':4, 'SH':5, 'S1':6, 'S2':7, 'RSTOS0':8, 'S3':9, 'OTH':10}
df['flag'] = df['flag'].map(fmap)
df.drop('service', axis = 1, inplace = True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Splitting the dataset


# Target variable and train set
y = df[['class']]
X = df.drop(['class', ], axis=1)

sc = MinMaxScaler()
X = sc.fit_transform(X)
print(X)
# Split test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

from sklearn.svm import SVC

clfs = SVC(gamma='scale')
start_time = time.time()
clfs.fit(X_train, y_train.values.ravel())
end_time = time.time()
print("Training time: ", end_time - start_time)

start_time = time.time()
y_test_pred = clfs.predict(X_train)
end_time = time.time()
print("Testing time: ", end_time-start_time)

print("Train score is:", clfs.score(X_train, y_train))
print("Test score is:", clfs.score(X_test, y_test))

plt.xlabel('Attack Prediction')
plt.ylabel('Intrusion')
plt.plot(clfs.predict(X_test),c='r')
plt.title('SVM')
plt.legend()
plt.show()