import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
print("Loading Data ... ")

iris=datasets.load_iris()
y=iris.target
# print('y',y)

y=y[y<2]#只取前两个类别
# print('y2',y)
x=iris.data[0:len(y)]
# print('x2',x)
# 用sklearn.cross_validation进行训练数据集划分，这里训练集和交叉验证集比例为7：3，可以自己根据需要设置
X, val_X, y, val_y = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=1,
    stratify=y # 这里保证分割后y的比例分布与原数据一致
)

X_train = X
y_train = y
X_test = val_X
y_test = val_y

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': {'binary_logloss', 'auc'},  #二进制对数损失
#     'num_leaves': 5,
#     'max_depth': 6,
#     'min_data_in_leaf': 450,
#     'learning_rate': 0.1,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.95,
#     'bagging_freq': 5,
#     'lambda_l1': 1,
#     'lambda_l2': 0.001,  # 越小l2正则程度越高
#     'min_gain_to_split': 0.2,
#     'verbose': 5,
#     'is_unbalance': True
# }

params = {

    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},  #二进制对数损失

}



# train
print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                early_stopping_rounds=500)

###根据5折cv的平均值，找到最佳的树的个数；每棵树都对应一个5折cv的平均值，找到平均值最大的那个个数
cv_results = lgb.cv(params, lgb_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',early_stopping_rounds=50,seed=0)
print('cv_results',cv_results)
print('best n_estimators:', len(cv_results['auc-mean']))
print('best cv score:', pd.Series(cv_results['auc-mean']).max())



print('Start predicting...')

preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)  # 输出的是概率结果

# 导出结果
threshold = 0.5
for pred in preds:
    result = 1 if pred > threshold else 0

# 导出特征重要性
importance = gbm.feature_importance(importance_type='split')
names = gbm.feature_name()

print('split:',names,importance)

# 导出特征重要性
importance = gbm.feature_importance(importance_type='gain')
names = gbm.feature_name()

print('gain:',names,importance)



# with open('./feature_importance.txt', 'w+') as file:
#     for index, im in enumerate(importance):
#         string = names[index] + ', ' + str(im) + '\n'
#         file.write(string)