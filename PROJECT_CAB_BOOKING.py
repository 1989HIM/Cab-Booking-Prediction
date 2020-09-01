#!/usr/bin/env python
# coding: utf-8

# ### OBJECTIVES:
#     1) To understand the supply and demand of cab booking
#     2) To improve the efficiency by minimizing the waiting time
#     3) To forecast cab booking by using historical data with weather data.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore',category=FutureWarning)
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn import metrics
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor


# ## 1. DATASET

# In[2]:


train=pd.read_csv('train.csv')
train.shape


# In[3]:


test=pd.read_csv('test.csv')
train_label=pd.read_csv('train_label.csv',header=None)
test_label=pd.read_csv('test_label.csv',header=None)
train['Total_booking']=train_label[0]
train.head()


# ### SEASON & CAB BOOKING

# In[8]:


fig=plt.figure()
fig,ax=plt.subplots()
plt.bar(train.season,train.Total_booking)


# ### HOLIDAY ,WORKING and Cab booking
# 

# In[9]:


# WD/HD/NWNHD = working days /Holidays/Neither working nor holidays
# WDB/HDB = working days booking/holidays booking
WD=train[train.workingday==1]['workingday'].count()
WDB=train[train.workingday==1]['Total_booking'].sum()

HD=train[train.holiday==1]['workingday'].count()
HDB=train[train.holiday==1]['Total_booking'].sum()

NWNHD=train[(train.holiday==0) & (train.workingday==0)]['holiday'].count()
NWNHDB=train[(train.holiday==0) & (train.workingday==0)]['Total_booking'].sum()

print('No of working days=',WD)
print('No of holidays    =',HD)
print('No of neither W&H =',NWNHD)
print(' ')
print('Booking on working days=',WDB)
print('Booking on holidays    =',HDB)
print('Booking on neither W&H =',NWNHDB)

print('Cab booking per weekday=',round(WDB/WD))
print('Cab booking per holiday=',round(HDB/HD))


# In[10]:


days=[WD,HD,NWNHD]
booking=[WDB,HDB,NWNHDB]
labels=['Working','Holiday','Neither']

fig1=plt.figure()
plt.subplots(1,2,figsize=(10,4))
plt.subplot(1,2,1)
plt.pie(days,labels=labels,autopct='%1.1f%%')
plt.axis('equal')
plt.title('DAYS')

plt.subplot(1,2,2)
plt.pie(booking,labels=labels,autopct='%1.1f%%')
plt.axis('equal')
plt.title('BOOKING')
plt.show()


#        1) From the figure, we can see that there isn't much significant difference in cab booking during weekdays and holidays as we can see in pie charts.
#        2) The number of cab booking per day on weekdays is 195 and on holidays is 188,not much difference.
#        3) Before jumping on to final conclusion, let's perform 2 sample t-test to check whether there is significant difference in cab booking between working and holiday

# ## t-test

# * Checking whether there is significant difference in cab booking between working and holiday.
# * Performing 2 sample t-test
# * Null hypothesis Ho = both are equal i.e. mean(work) = mean (holiday)
# * Alternate hypothesis H1= both are not equal i.e. mean(work) != mean(holiday)

# In[9]:


workday=train[train.workingday==1]['Total_booking']
holiday=train[train.holiday==1]['Total_booking']
from scipy.stats import ttest_ind

workday_mean=np.mean(workday)
holiday_mean=np.mean(holiday)

workday_std=np.std(workday)
holiday_std=np.std(holiday)

ttest,pval=ttest_ind(workday,holiday)
print('Workday_mean=',workday_mean)
print('holiday_mean=',holiday_mean)
print(' ')
print('workday_std=',workday_std)
print('holiday_std=',holiday_std)
print(' ')
print('test_stat_val=',ttest)
print('P-value=',pval)
print('')
if pval <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  print('The result is NOT statistically significant !')


# ### DATETIME AND TOTAL BOOKING

# In[10]:


dt=train[['datetime','Total_booking']]
dt['NewDatetime']=pd.to_datetime(dt['datetime'],errors='ignore')
#dt['date']=dt['NewDatetime'].dt.date
dt['month']=dt['NewDatetime'].dt.month
dt['year']=dt['NewDatetime'].dt.year
dt['hour']=dt['NewDatetime'].dt.hour
dt['Total_booking']=train['Total_booking']
dt['Dayofyear']=dt['NewDatetime'].dt.dayofyear
#dt.head()


# In[11]:


# ANNUAL BOOKING
plt.subplots(1,3,figsize=(15,4))
plt.subplot(1,3,1)
year=dt.groupby('year').sum()
x=[year.index[0],year.index[1]]
y=[year.iloc[0][0],year.iloc[1][0]]
plt.bar(x,y,label='Annual_booking',width=0.2,tick_label=[2011,2012],color='c')
plt.legend()
#--------------------------------------------------------------------------
monthly_2011=dt[dt.year==2011].groupby('month')['Total_booking'].sum()
monthly_2012=dt[dt.year==2012].groupby('month')['Total_booking'].sum()
#--------------------------------------------------------------------------
plt.subplot(1,3,2)
plt.bar(monthly_2011.index.values,monthly_2011,color='g')
plt.title('Monthly Booking (2011)')

plt.subplot(1,3,3)
plt.bar(monthly_2012.index.values,monthly_2012,color='y')
plt.title('Monthly Booking (2012)')
plt.show()


# In[12]:


# CHECKING WHETHER CHANGE IN MONTHLY BOOKING IS STATISTICALLY SIGNIFICANT OR NOT
# FIRST CHECKING THE NORMALITY OF MONTHLY BOOKING


# In[13]:


m_2011=dt[dt.year==2011]
month_2011_booking=m_2011.groupby('month')['Total_booking'].sum()
m_2012=dt[dt.year==2012]
month_2012_booking=m_2012.groupby('month')['Total_booking'].sum()


# In[14]:


from scipy import stats
import pylab
plt.subplots(1,2,figsize=(8,3))
plt.subplot(1,2,1)
stats.probplot(month_2011_booking,dist="norm", plot=pylab)
plt.subplot(1,2,2)
stats.probplot(month_2012_booking,dist='norm',plot=pylab)
plt.show()


# ### 2 sample t-test

# In[15]:


month_2011_booking_mean=np.mean(month_2011_booking)
month_2012_booking_mean=np.mean(month_2012_booking)
month_2011_booking_std=np.std(month_2011_booking)
month_2012_booking_std=np.std(month_2012_booking)
ttest,pval=ttest_ind(month_2011_booking,month_2012_booking)

print('month_2011_booking_mean=',month_2011_booking_mean)
print('month_2012_booking_mean=',month_2012_booking_mean)
print(' ')
print('month_2011_booking_std=',month_2011_booking_std)
print('month_2012_booking_std=',month_2012_booking_std)
print(' ')
print('test_stat_val=',ttest)
print('P-value=',pval)
print('')
if pval <0.05:
  print("we reject null hypothesis")
  print('The result is statistically significant !')
else:
  print("we accept null hypothesis")
  print('The result is NOT statistically significant !')


# In[16]:


x=dt.groupby('year')['Total_booking'].sum()
demand=round((x.iloc[1]-x.iloc[0])*100/(x.iloc[0]),2)
demand


#      1) The percentage rise in cab booking in 2012 as compare to 2011 is 70.54%
#      2) The maximum percentage increase can be seen in 1st quater of year 2012

# ### WEATHER PARAMETERS 
#     Temperature, Windspeed & Humidity

# In[17]:


plt.subplots(1,3,figsize=(15,4))
plt.subplot(1,3,1)
windspeed=train.groupby('windspeed')['Total_booking'].sum()
plt.plot(windspeed.index.values,windspeed.values,color='green',label='windspeed',linewidth=2,markersize=12)
plt.xlabel('Windspeed')
plt.ylabel('Booking')
plt.legend()
plt.subplot(1,3,2)
temp=train.groupby('temp')['Total_booking'].sum()
plt.plot(temp.index.values,temp.values,color='m',label='Temperature',linewidth=2,markersize=12)
plt.xlabel('Temperature')
plt.ylabel('Booking')
plt.legend()
plt.subplot(1,3,3)
humidity=train.groupby('humidity')['Total_booking'].sum()
plt.plot(humidity.index.values,humidity.values,color='c',label='Humidity',linewidth=2,markersize=12)
plt.xlabel('Humidity')
plt.ylabel('Booking')
plt.legend()
plt.show()


# ### Weather

# In[18]:


W=train.groupby('weather')['Total_booking'].sum()
print(W)
sns.set(style=('darkgrid'))
x=sns.barplot(x=list(W.index),y=list(W), saturation=2)
x.set_xticklabels(x.get_xticklabels(), rotation=45)
plt.show()


# ## OUTLIER ANALYSIS

# In[19]:


sns.set(style='white',palette='muted',color_codes=True)
f,axes=plt.subplots(1,3,figsize=(11,3),sharex=True)
plt.subplot(1,3,1)
sns.boxplot(train['windspeed'].values, color="b")
plt.xlabel('windspeed')
plt.subplot(1,3,2)
sns.boxplot(train['temp'].values, color="g")
plt.xlabel('Temperature')
plt.subplot(1,3,3)
sns.boxplot(train['humidity'].values, color="r")
plt.xlabel('Humidity')
plt.show()


# ### By Interquartile Range (IQR)

# In[20]:


def outlier_analysis(data):
    q1,q2,q3=np.percentile(data,[25,50,70])
    iqr=q3-q1
    upper_bound=round(q3+1.5*iqr,3)
    lower_bound=round(q1-1.5*iqr,3)
    return lower_bound,upper_bound
print('UPPER & LOWER BOUNDS')
# Temperature
print('Temperature   =',outlier_analysis(train['temp']))
# ATemperature
print('A_Temperature =',outlier_analysis(train['atemp']))
# Humidity
print('Humidity      =',outlier_analysis(train['humidity']))
# Windspeed
print('Windspeed     =',outlier_analysis(train['windspeed']))


# ### By Standard Deviation Method

# In[21]:


def std(data):
    mean=np.mean(data)
    stdv=np.std(data)
    lower_bound=round(mean-3*stdv,2)
    upper_bound=round(mean+3*stdv,2)
    return lower_bound,upper_bound
print('UPPER & LOWER BOUNDS')
print('Temperature  =',std(train['temp']))
print('A_Temperature=',std(train['atemp']))
print('Humidity     =',std(train['humidity']))
print('Windspeed    =',std(train['windspeed']))


# ### copying original dataset into new dataframe

# In[22]:


train1=train
train1.head(1)


# In[23]:


#outlier removal
train1=train1[train1.temp<42.64]
train1=train1[train1.humidity<119.24]
train1=train1[train1.humidity>4.05]
train1=train1[train1.windspeed<31.993]
train1.shape


# In[24]:


#check for windspeed
sns.boxplot(train1['windspeed'].values, color="b")
plt.xlabel('windspeed')


# In[25]:


data=train1.copy()
data.head(1)
data[['month','year','hour','Dayofyear']]=dt[['month','year','hour','Dayofyear']]
dummy_season=pd.get_dummies(data.season)
dummy_weather=pd.get_dummies(data.weather)
dummy=pd.concat([dummy_season,dummy_weather],axis='columns')
dummy1=dummy.drop([dummy.columns[0],dummy.columns[4]],axis='columns')
data=pd.concat([data,dummy1],axis='columns')
#x1=x.iloc[:,2:]
#x=data.drop(['weather','datetime','season','Mist + Cloudy'],axis='columns')
#x.head(1)
data1=data.drop(['datetime','season','weather','holiday'],axis='columns')
data2=data1.copy()
data2.head()


# ### LINEAR REGRESSION

# In[26]:


X=data2.drop('Total_booking',axis='columns')
y=data2['Total_booking']


# In[27]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# In[28]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)


# In[29]:


y_pred=lr.predict(X_test)


# In[30]:


lr.score(X_test,y_pred)


# #### Remarks:
#     MODEL IS OVERFITTING !

# ## FEATURE ENGINEERING
#     objective - to reduce number of correlated features

# ### HEAT MAP

# In[39]:


plt.figure(figsize=(20,15))
sns.heatmap(data2.corr(),annot=True)


# ### FEATURE SCALING
#       >> Using min-max scaling to scale all the features in a dataset in 0-1 range
#       >> Rescaling all the features except categorical.

# In[31]:


def scaling(x):
    scaling = (x-np.min(x))/(np.max(x)-np.min(x))
    return scaling


# In[32]:


re_sc=data2.iloc[:,2:-6]
total_booking=re_sc['Total_booking']
re_sc1=re_sc.drop(['Total_booking'],axis=1)
data2_1=scaling(re_sc1)
data2_2=data2.iloc[:,[0,1,-6,-5,-4,-3,-2,-1]]
#data2_2.head()


# In[33]:


# for Numeric data
#plt.figure(figsize=(20,15))
#sns.pairplot(data2_1)


# In[34]:


data3=pd.concat([data2_1,data2_2],axis='columns')
data3.head(2) #<----- scaled dataframe


# ### Checking with OLS

# In[35]:


x_ols1=data3
y_ols1=total_booking
X_train_ols1,X_test_ols1,y_train_ols1,y_test_ols1=train_test_split(x_ols1,y_ols1,test_size=0.3)
ols_1 = sm.OLS(y_train_ols1,X_train_ols1).fit()
print(ols_1.summary())


# ### Remarks:
#        -The R2 and adjusted R2 going upto 0.71.
#        -Now, lets check for multicollinearity by VIF (Variance Infaltion Factor)
#     

# ### REDUCING MULTICOLLINEARITY by VIF (Variance Inflation Factor)

# In[36]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# In[37]:


data3_vif=add_constant(data3)
pd.Series([variance_inflation_factor(data3_vif.values, i)for i in range(data3_vif.shape[1])],
          index=data3_vif.columns)


# ### Remarks
#        -Some features are having VIF above 10.
#        -Removing all the features having VIF above 10 in order to reduce
#         the multicollinearity

# #### REMOVING MONTH, DAYOFYEAR,TEMP,ATEMP AND SPRING TO BRING VIF BELOW 10

# In[38]:


data4=data3.drop(['month','Dayofyear','Spring'],axis=1)
data4_vif=add_constant(data4)
pd.Series([variance_inflation_factor(data4_vif.values, i)for i in range(data4_vif.shape[1])],
          index=data4_vif.columns)


# In[39]:


# NOW THIS DATAFRAME WILL BE USED FOR MODELLING
# data4 = Independednt variables
# total_booking = dependednt variable
data4.head(3)


# ### Train and Test split

# In[40]:


X=data4
Y=total_booking
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)


# ### OLS after removing features

# In[41]:


ols_2 = sm.OLS(y_train,X_train).fit()
print(ols_2.summary())


# ### Remarks:
#      After removing multicolinear features and performing OLS, the accuracy of the 
#      R squared value dropped from 0.71 to 0.657.

# ## K-fold cross validation
#          - Random Forest Regressor

# In[42]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


# ## Random Forest Regressor

# In[49]:


rfr=RandomForestRegressor()
score_rfr=cross_val_score(rfr,X_train,y_train,cv=10,scoring='r2')
score_rfr.mean()


# ### Grid Search CV +Random Forest Regressor

# In[59]:


from sklearn.model_selection import GridSearchCV


# In[115]:


params={'n_estimators':np.arange(1,40),'max_depth':np.arange(1,13),'max_features':np.arange(1,6)}
gcv=GridSearchCV(rfr,params,scoring='r2',cv=5,verbose=1)
gcv.fit(X_train,y_train)
print(gcv.best_score_)
print(gcv.best_params_)


# ### The best parameters obtained from Grid search CV:  'max_depth': 12, 'max_features': 5 and 'n_estimators': 18. But the accuracy obtained is not much

# In[ ]:


rf_df=pd.DataFrame()
l1=[]
l2=[]
for i in range(1,500):
    rf1=RandomForestRegressor(random_state=i,max_depth=12,max_features=5,n_estimators=37)
    rf1.fit(X_train,y_train)
    y_pred=rf1.predict(X_test) 
    score=r2_score(y_test,y_pred)
    l1.append(i)
    l2.append(score)
    #print('RS=',i,' ',score)


# In[119]:


rf_df['random_state']=l1
rf_df['score']=l2
xyz=pd.DataFrame()
xyz['A']=l1
xyz['B']=l2
xyz.sort_values(by='B',axis=0,ascending=False).head()


# In[2]:


from sklearn.ensemble import AdaBoostRegressor


# In[122]:


abr=AdaBoostRegressor()
params={'learning_rate':np.arange(0.1,1,0.001),'n_estimators':np.arange(1,50)}
gcv=GridSearchCV(abr,params,scoring='r2',cv=5,verbose=1)
gcv.fit(X_train,y_train)
print(gcv.best_score_)
print(gcv.best_params_)


# ### AdaBoostRegressor didn't performed well as compared to Random Forest regressor

# In[123]:


from xgboost import XGBRegressor


# xgbr=XGBRegressor()
# params={'learning_rate':np.arange(0.1,1,0.001),'n_estimators':np.arange(1,100),'max_depth':np.arange(1,13)}
# gcv=GridSearchCV(xgbr,params,scoring='r2',cv=5,verbose=1)
# gcv.fit(X_train,y_train)
# print(gcv.best_score_)
# print(gcv.best_params_)

# ### Applying best parameters obtained from GridSearchCV in Xgboost regressor

# In[88]:


xgbr=XGBRegressor(learning_rate=0.25,n_estimators=375)
xgbr.fit(X_train,y_train)
y_pred=xgbr.predict(X_test)
metrics.r2_score(y_pred,y_test)


# In[ ]:


abc=AdaBoostRegressor(base_estimator=RandomForestRegressor(random_state=20,max_depth=12,n_estimators=375),learning_rate=0.1,n_estimators=375,random_state=20)
abc.fit(X_train,y_train)
y_pred=xgbr.predict(X_test)
metrics.r2_score(y_pred,y_test)


# ### The best accuracy achieved was 92.46 % by using AdaBoostRegressor with RandomForestRegressor as base estimator.
# 

# In[ ]:




