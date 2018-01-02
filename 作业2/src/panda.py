import pandas as pd

user_log_format1=pd.read_csv('user_log_format1.csv')
user_info_format1=pd.read_csv('user_info_format1.csv')
train_format1=pd.read_csv('train_format1.csv')
merge_userid4=pd.merge(train_format1,user_info_format1,how='inner',on='user_id')
merge_userid4.to_csv('merge_userid4.csv')

favourite_user_log_format1=user_log_format1[user_log_format1['action_type']==3]
#favourite_user_log_format1.rename(columns={'seller_id':'merchant_id'},inplace=True)
favourite_user_log_format1.to_csv('favourite_user_log_format1.csv')

favourite_user_log_format1=pd.read_csv('user_log_format1.csv')
favourite_user_log_format1.rename(columns={'seller_id':'merchant_id'},inplace=True)
merge_userid4=pd.read_csv('merge_userid4.csv')
#purchase_user_log_format1=user_log_format1[user_log_format1['action_type']==2]
#purchase_user_log_format1.to_csv('purchase_user_log_format1.csv')

All_merge_userid=pd.merge(merge_userid4,favourite_user_log_format1,how='inner',on=['user_id','merchant_id'])
All_merge_userid.to_csv('All_merge_userid_all.csv')

#print(left)
#print(right)

#merge_userid=pd.merge(train_format1,user_info_format1,how='right',on='user_id')
#print(merge_userid)
#merge_userid.to_csv('merge_userid.csv')
#merge_userid3=pd.merge(train_format1,user_info_format1,how='outer',on='user_id')
#merge_userid3.to_csv('merge_userid3.csv')
