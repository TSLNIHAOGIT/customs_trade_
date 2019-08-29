from sklearn.model_selection import train_test_split
from pandas import DataFrame
from sklearn import metrics
from sklearn.datasets import make_hastie_10_2
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import pandas as pd

# 准备数据，y本来是[-1:1],xgboost自带接口邀请标签是[0:1],把-1的转成1了。
X, y = make_hastie_10_2(random_state=0)
X = DataFrame(X)
y = DataFrame(y)
y.columns = {"label"}
label = {-1: 0, 1: 1}
y.label = y.label.map(label)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # 划分数据集

# XGBoost自带接口
params = {
    'eta': 0.3,
    'max_depth': 3,
    'min_child_weight': 1,
    'gamma': 0.3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'nthread': 12,
    'scale_pos_weight': 1,
    'lambda': 1,
    'seed': 27,
    'silent': 0,
    'eval_metric': 'auc'
}
d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_test, label=y_test)
d_test = xgb.DMatrix(X_test)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

# sklearn接口
clf = XGBClassifier(
    n_estimators=30,  # 三十棵树
    learning_rate=0.3,
    max_depth=3,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=12,
    scale_pos_weight=1,
    reg_lambda=1,
    seed=27,
    importance_type='gain'
)

# print("XGBoost_自带接口进行训练：")
# model_bst = xgb.train(params, d_train, 30, watchlist, early_stopping_rounds=500, verbose_eval=10)


print("XGBoost_sklearn接口进行训练：")
model_sklearn = clf.fit(X_train, y_train)

##########################
booster = model_sklearn.get_booster()
print('booster',booster)
'''
* 'weight': the number of times a feature is used to split the data across all trees.
* 'gain': the average gain across all splits the feature is used in.
* 'cover': the average coverage across all splits the feature is used in.
* 'total_gain': the total gain across all splits the feature is used in.
* 'total_cover': the total coverage across all splits the feature is used in
'''
#['weight', 'gain', 'cover', 'total_gain', 'total_cover']
import numpy as np
imp0=booster.get_score(importance_type='cover')
b=booster
score = b.get_score(importance_type='gain')
all_features = [score.get(f, 0.) for f in b.feature_names]
print('all_features',all_features)
all_features = np.array(all_features, dtype=np.float32)
print(all_features / all_features.sum())

imp0={key:np.float32(value) for key ,value in imp0.items()}
print('iiimmmp',imp0)
sum_values=sum(imp0.values())

#注意不是转为整形，不然出错
imp0={key:np.float32(value)/sum_values for key ,value in imp0.items()}
imp0=dict(sorted(imp0.items(),key=lambda item:item[1],reverse=True))
print('imp0',imp0)

imp = booster.get_score(importance_type='gain')#.feature_importance(importance_type='split')#importance_type='total_gain'
imp={key:np.float32(value) for key ,value in imp.items()}

sum_values=sum(imp.values())



imp={key:np.float32(value)/sum_values for key ,value in imp.items()}
#python3.6 dict是有序的
imp=dict(sorted(imp.items(),key=lambda item:item[1],reverse=True))
feature_name = booster.feature_names

print('imp',imp)





# for (feature_name,importance) in zip(feature_name,imp):
#
#     print ('featuress',feature_name,int(importance)/sum_values)


##########################################3


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# sorted(zip(clf.feature_importances_, X.columns), reverse=True)
#列明得是字符串显示才正常，纵坐标为特征名称，横坐标是重要性数值；否则显示不正常
X_train_columns=['f_{}'.format(col_name) for col_name in X_train.columns]

print('X_train_columns',X_train_columns)

fmp_sorted=sorted(zip(model_sklearn.feature_importances_,X_train_columns),reverse=True)
print('fmp_sorted',fmp_sorted)
# feature_imp = pd.DataFrame(fmp_sorted, columns=['Value','Feature'])


# plt.figure(figsize=(20, 10))
# sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
# plt.title('LightGBM Features (avg over folds)')
# plt.tight_layout()
# plt.savefig('xgb_importances-01.png')
# plt.show()



# y_bst = model_bst.predict(d_test)
y_sklearn = clf.predict_proba(X_test)[:, 1]

# print("XGBoost_自带接口    AUC Score : %f" % metrics.roc_auc_score(y_test, y_bst))
print("XGBoost_sklearn接口 AUC Score : %f" % metrics.roc_auc_score(y_test, y_sklearn))

# 将概率值转化为0和1
# y_bst = pd.DataFrame(y_bst).apply(lambda row: 1 if row[0] >= 0.5 else 0, axis=1)
y_sklearn = pd.DataFrame(y_sklearn).apply(lambda row: 1 if row[0] >= 0.5 else 0, axis=1)
# print("XGBoost_自带接口    AUC Score : %f" % metrics.accuracy_score(y_test, y_bst))
print("XGBoost_sklearn接口 AUC Score : %f" % metrics.accuracy_score(y_test, y_sklearn))
'''
XGBoost_自带接口    AUC Score : 0.970292
XGBoost_sklearn接口 AUC Score : 0.970292
XGBoost_自带接口    AUC Score : 0.897917
XGBoost_sklearn接口 AUC Score : 0.897917
'''
