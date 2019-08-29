from sklearn import datasets
print("Loading Data ... ")
iris=datasets.load_iris()
y=iris.target
y=y[y<2]
x=iris.data[0:len(y)]
print(x,y)

# X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3)