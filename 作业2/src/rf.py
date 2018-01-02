from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import log_loss
import pandas as pd             

merge_userid2=pd.read_csv('All_merge_userid_all.csv')
merge_userid2=merge_userid2.dropna()

X=merge_userid2[['user_id','merchant_id','age_range','gender','item_id','cat_id','brand_id','time_stamp','action_type']]
#print(X)
Y=merge_userid2['label']
#print(Y)

min_max_scaler=preprocessing.MinMaxScaler()
min_max_X=min_max_scaler.fit_transform(X)
#min_max_Y=min_max_scaler.fit_transform(Y)
min_max_Y=Y
#print(min_max_Y)
#print(min_max_X)

X_train,X_test,Y_train,Y_test=train_test_split(min_max_X,min_max_Y,test_size=0.3)
lr=LogisticRegression()
lr.fit(X_train,Y_train)
print('LogisticRegression score:',lr.score(X_test,Y_test))
y_lr=lr.predict(X_test)
print('LogisticRegression log_loss:',log_loss(Y_test,y_lr))

RF = RandomForestClassifier()
RF = RF.fit(X_train,Y_train)
print('RandomForest score:',RF.score(X_test,Y_test))
y_rf=RF.predict(X_test)
print('RandomForest logloss:',log_loss(Y_test,y_rf))

#print(lr.predict(X_test)[:40])
#print(Y_test[:40])
#X_train,X_test,Y_train,Y_test=train_test_split(iris_X,iris_Y,test_size=0.3)
#knn=KNeighborsClassifier()
#knn.fit(X_train,Y_train)
#print(knn.score(X_test,Y_test))
#print(knn.predict(X_test))
#print(Y_test)
