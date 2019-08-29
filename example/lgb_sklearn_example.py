import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification
# 加载数据
print('Load data...')

iris = load_iris()
data=iris.data
target = iris.target
X_train,X_test,y_train,y_test =train_test_split(data,target,test_size=0.2)

print('X_train',X_train[0:10])
print('y_train',y_train[0:10])
# df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
# df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')
# y_train = df_train[0].values
# y_test = df_test[0].values
# X_train = df_train.drop(0, axis=1).values
# X_test = df_test.drop(0, axis=1).values

print('Start training...')
# 创建模型，训练模型

###其实是分类数据
gbm = lgb.LGBMRegressor(objective='regression',num_leaves=31,learning_rate=0.05,n_estimators=20)

'''
gbm22 = lgb.LGBMClassifier(objective='binary',num_leaves=31,learning_rate=0.05,n_estimators=20)
gbm22.booster_.feature_importance(importance_type='gain')

[split gain]
importance_type : string, optional (default="split")
How the importance is calculated.
If "split", result contains numbers of times the feature is used in a model.
If "gain", result contains total gains of splits which use the feature.

'''



gbm.fit(X_train, y_train,eval_set=[(X_test, y_test)],eval_metric='l1',early_stopping_rounds=5)

print('importances',gbm.feature_importances_)



booster = gbm.booster_

importance = booster.feature_importance(importance_type='gain')

feature_name = booster.feature_name()

for (feature_name,importance) in zip(feature_name,importance):

    print ('importance ***',feature_name,importance)




import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)





# sorted(zip(clf.feature_importances_, X.columns), reverse=True)
X_train_columns=['f1','f2','f3','f4']
print('X_train_columns',X_train_columns)
feature_imp = pd.DataFrame(sorted(zip(gbm.feature_importances_,X_train_columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances-01.png')
plt.show()




print('Start predicting...')
# 测试机预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# 模型评估
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# feature importances
print('Feature importances:', list(gbm.feature_importances_))

# 网格搜索，参数优化
estimator = lgb.LGBMRegressor(num_leaves=31)

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}

gbm = GridSearchCV(estimator, param_grid)

gbm.fit(X_train, y_train)

print('Best parameters found by grid search are:', gbm.best_params_)
