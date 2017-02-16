##4 SKlearn
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
irisX = iris.data
irisY = iris.target

#sep train test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(irisX, irisY,test_size = 0.3)
knn = KNeighborsClassifier()
knn.fit(Xtrain,Ytrain)
print(knn.predict(Xtest))
print(Ytest)



##5 Datasets Regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

load_data = datasets.load_boston()
bosX = load_data.data
bosY = load_data.target

lr = LinearRegression()
lr.fit(bosX,bosY)
print(lr.predict(bosX[:4,:]))
print(bosY[:4])

X,y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)
#用來生data
plt.scatter(X,y)
plt.show()



##6 Model
print(lr.coef_)   #斜率
print(lr.intercept_) #截距
print(lr.get_params()) #定義的參數
print(lr.score(bosX,bosY)) #R^2 看這個回歸多少多好



##7 Normalization
from sklearn import preprocessing
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC

a = np.array([[10,2.7,3.6],
              [-100, 5,-2],
              [120,20,40]],dtype=np.float64)
print(a)
print(preprocessing.scale(a))

X,y = make_classification(n_samples=300,n_features=2,n_redundant=0,n_informative=2,
                          n_clusters_per_class=1,
random_state=22,scale=100)
#用來生data
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

#X= preprocessing.minmax_scale(X,feature_range=(0,1)) #複雜板
X = preprocessing.scale(X)
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=.3)
clf=SVC()
clf.fit(Xtrain,Ytrain)
print(clf.score(Xtest,Ytest)) #有scale過0.93 眉0.56準確度



##8 CV1
iris = datasets.load_iris()
XX = iris.data
YY = iris.target
XXtrain,XXtest,YYtrain,YYtest = train_test_split(XX,YY,random_state=4)
knn1= KNeighborsClassifier(n_neighbors=5)
knn1.fit(XXtrain,YYtrain)
#y_pred=knn1.predict(XXtest)
print(knn1.score(XXtest,YYtest))

#加上分五組的功能 
from sklearn.cross_validation import cross_val_score
knn=KNeighborsClassifier(n_neighbors=5)
scores=cross_val_score(knn,XX,YY,cv=5,scoring='accuracy') #自動分五組
print(scores.mean()) #平均後比較準


#看不同neighbor大小

k_range=range(1,31)
k_scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    loss= -cross_val_score(knn,XX,YY,cv=10,scoring='mean_squared_error') #會是負值 圖反過來 誤差越小越好
    #scores=cross_val_score(knn,XX,YY,cv=10,scoring='accuracy') #精準度
    k_scores.append(loss.mean())

plt.plot(k_range,k_scores)
plt.xlabel('Value of k for knn')
plt.ylabel('CV accuracy')
plt.show()
#12-20個neighbor是最好的 超過會overfit



##9 CV2 overfit
from sklearn.learning_curve import learning_curve
from sklearn.datasets import load_digits

digits = load_digits()
X=digits.data
y=digits.target
train_sizes, train_loss, test_loss=learning_curve(
    SVC(gamma=0.001),X,y,cv=10,scoring='mean_squared_error',
    train_sizes=[.1,.25,.5,.75,1])   #五個點記錄 
train_loss_mean= -np.mean(train_loss,axis=1)
test_loss_mean=-np.mean(test_loss,axis=1)

plt.plot(train_sizes,train_loss_mean,'o-',color='r',label='training')
plt.plot(train_sizes,test_loss_mean,'o-',color='g',label='cv')
plt.xlabel('training example')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()
#如果gamma改成0.01會變成不好 遍成overfitting CV取線的loss會變大 要小心 隨時要看CV取線




##10 CV3 找出參數
from sklearn.learning_curve import validation_curve

digits = load_digits()
X=digits.data
y=digits.target
param_range=np.logspace(-6,-2.3,5)
train_loss, test_loss=validation_curve(
    SVC(),X,y,param_name='gamma',param_range=param_range,cv=10,
    scoring='mean_squared_error')   #改gamma而已
train_loss_mean= -np.mean(train_loss,axis=1)
test_loss_mean=-np.mean(test_loss,axis=1)

plt.plot(param_range,train_loss_mean,'o-',color='r',label='training')
plt.plot(param_range,test_loss_mean,'o-',color='g',label='cv')
plt.xlabel('gamma')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()
#gamma餐數要選在0.0005,0.0006中間比較好 loss最小




##11 保存model

from sklearn import svm

clf = svm.SVC()
iris = datasets.load_iris()
X,y = iris.data, iris.target
clf.fit(X,y)

#method1:保存pickle
import pickle
with open('save/clf.pickle','wb') as f:
    pickle.dump(clf,f)
#島出
with open('save/clf.pickle','rb') as f:
    clf2=pickle.load(f)
    print(clf2.predict(X[0:1]))



#method2: joblib 跑比較多.pkl文件
from sklearn.externals import joblib
joblib.dump(clf,'save/clf.pkl')
#島出
clf3 = joblib.load('save/clf.pkl')
print(clf3.predict(X[0:1]))




