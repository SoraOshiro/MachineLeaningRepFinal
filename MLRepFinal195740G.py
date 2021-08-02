import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# データの読み込み
train_file = './datasets/adult.data'
test_file = './datasets/adult.test'

col_names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']

x_train = pd.read_csv(train_file,header=None,names=col_names, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])
y_train = pd.read_csv(train_file,header=None,names=['income'], usecols=[14])
x_test = pd.read_csv(test_file,header=None,names=col_names, skiprows=1,usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])
y_test = pd.read_csv(test_file,header=None,names=['income'],skiprows=1,usecols=[14])

# データを連続値とクラス値で分割
col_class = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
col_num = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']

# OneHotEncoding - X
x_train_c = x_train[col_class]
hot_x = OneHotEncoder(sparse=False,handle_unknown='ignore')
x_train_c_trans = hot_x.fit_transform(x_train_c)

col_class_trans = [cls for ary in hot_x.categories_ for cls in ary]
df_x_train_c_trans = DataFrame(x_train_c_trans, columns=col_class_trans)

df_x_train_trans = df_x_train_c_trans.copy()
df_x_train_trans[col_num] = x_train[col_num]

x_test_c = x_test[col_class]
x_test_c_trans = hot_x.transform(x_test_c)

df_x_test_c_trans = DataFrame(x_test_c_trans, columns=col_class_trans)
df_x_test_trans = df_x_test_c_trans.copy()
df_x_test_trans[col_num] = x_test[col_num]

x_train = df_x_train_trans.values
x_test = df_x_test_trans.values

# LabelEncoding - Y
y_test = np.ravel(y_test.values)
y_test_repl = [s.replace('.', '') for s in y_test]
y_test_repl = np.array(y_test_repl)

y_train = np.ravel(y_train.values)
le = LabelEncoder()
le.fit(y_train)
y_train_id = le.transform(y_train)
y_test_id = le.transform(y_test_repl)

# データのスケーリング
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

# 分類学習の実行
# SVC
svm = SVC(kernel='rbf',gamma='scale',C=1.0)
svm.fit(x_train, y_train_id)
svm_predict = svm.predict(x_test)
print('SupportVectorMachineClassifier accuracy:{} precision:{} recall:{}'.format(accuracy_score(y_test_id, svm_predict), precision_score(y_test_id, svm_predict), recall_score(y_test_id, svm_predict)))

# DecisionTreeCrassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(x_train, y_train_id)
dtc_predict = dtc.predict(x_test)
print('DecisionTreeCrassifier accuracy:{} precision:{} recall:{}'.format(accuracy_score(y_test_id, dtc_predict), precision_score(y_test_id, dtc_predict), recall_score(y_test_id, dtc_predict)))

# KNeighborsClassifier
knc = KNeighborsClassifier()
knc.fit(x_train, y_train_id)
knc_predict = knc.predict(x_test)
print('KNeighborsClassifier accuracy:{} precision:{} recall:{}'.format(accuracy_score(y_test_id, knc_predict), precision_score(y_test_id, knc_predict), recall_score(y_test_id, knc_predict)))

