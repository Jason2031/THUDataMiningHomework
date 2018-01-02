# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#train_format1=pd.read_csv('../train_format1.csv')
#user_info_format1=pd.read_csv('../user_info_format1.csv')

#user_log_format1=pd.read_csv('../user_log_format1.csv')

#purchased_user_log_format1=user_log_format1[user_log_format1['action_type']==2]

#purchased_user_log_format1.to_csv('purchased_user_log_format1')

merge_userid2=pd.read_csv('merge_userid2.csv')
merge_userid2_label1=merge_userid2[merge_userid2['label']==1]

merchant_count=merge_userid2_label1['merchant_id'].value_counts()

used_merchant=merchant_count[0:10]
dataSet=[]
#print(used_merchant.index)
for i in used_merchant.index:
	print (i)
	users=merge_userid2[merge_userid2['merchant_id']==i]
	X=users['age_range']
	Y=users['gender']
	dataSet=users[['age_range','gender']]
	#print(dataSet.head(5))
	fileName1=str(i)+"X.csv"
	fileName2=str(i)+"Y.csv"
	X.to_csv(fileName1)
	Y.to_csv(fileName2)
	clf=KMeans(n_clusters=8)
	s=clf.fit(dataSet.dropna())
	print (s)
	cluster_centers=clf.cluster_centers_
	#print(cluster_centers)
	#print(clf.labels_)
	#print (clf.inertia_)
	#plt.figure()
	#plt.scatter(X,Y)
	plt.xlabel('age_range')
	plt.ylabel('gender')
	plt.title('merchant_id:'+str(i))

	for j in cluster_centers:
		#print (i[0],i[1])
		plt.scatter(j[0],j[1])
	#plt.show()	
	picName=str(i)+".jpg"
	plt.savefig(picName)