import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
#############################################################################################################################################
#DATA PREP, REMOVE Na's
df = pd.read_excel(r'C:\Users\mattc\OneDrive\Desktop\MyProjects\Census Data.xlsx')
#Show the head
print(df.head())
#drop the pay column (would defeat the point of the classifier to keep)
df.drop('Pay', axis = 1)
#Look at info
df.info()
#The head indicates that null is shown as question marks
#This loop removes spaces in columns holding data of type string
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.strip()
#Now all question marks have no spaces
df.replace(to_replace = '?', value = np.nan, inplace = True)
#Now count the null's and display
print('This shows the na values')
print(df.isna().sum())
#one row is completely null. just remove it 
df.drop(32561, axis = 0)
#Nan values are fixed
#Next goal is to understand distribution to see the nature of the na's and how to proceed
print('DISTRIBUTION OF WORK CLASS')
print(df['Work Class'].value_counts())
print('DISTRIBUTION OF OCCUPATION')
print(df['occupation'].value_counts())
print('DISTRIBUTION OF NATIVE-COUNTRY')
print(df['native-country'].value_counts())
#In 'Work class' we see outliers in 'without pay' and 'never workerd'. This suggests that people may underreport not working out of shame

#In 'occupation' the outlier is military, but there are not many military officers in the US, although this is still off the proportion

#In 'native-country' the only thing that comes to mind is the Honduras category might be underreported because many are here illigally.
#I will drop all na's. There may be some systematic patterns, but none firmly jump out
df.dropna(inplace = True)
#############################################################################################################################################
#DATA PREP, REMOVE CATEGORIES THAT ARE TOO GRANULAR
#Following loop prints uniques values of all categorical columns
for i in df.columns:
    if df[i].dtype == 'object':
        print('UNIQUE VALUES FOR:')
        print(i)
        print(df[i].unique())
#Categories can be cut from education. The following will grouped to No-HS

#11th, 9th, 7th-8th, 5th-6th, 10th, Preschool, 12th, 1st-4th
df['Education']=np.where(df['Education'] =='11th', 'No-HS', df['Education'])
df['Education']=np.where(df['Education'] =='9th', 'No-HS', df['Education'])
df['Education']=np.where(df['Education'] =='7th-8th', 'No-HS', df['Education'])
df['Education']=np.where(df['Education'] =='5th-6th', 'No-HS', df['Education'])
df['Education']=np.where(df['Education'] =='10th', 'No-HS', df['Education'])
df['Education']=np.where(df['Education'] =='Preschool', 'No-HS', df['Education'])
df['Education']=np.where(df['Education'] =='12th', 'No-HS', df['Education'])
df['Education']=np.where(df['Education'] =='1st-4th', 'No-HS', df['Education'])

#Group Some-college and HS-grad can be grouped to HS-grad
df['Education']=np.where(df['Education'] =='Some-college', 'HS-grad', df['Education'])

print(df['Education'].unique())

#Now have to make all string columns numeric to use logistic regression
model_data = pd.get_dummies(df)
#############################################################################################################################################
#DETERMINE RELEVANT COVARIATES 
#irrelevant covariates will be eliminated by examining the score of one variables used to explain the dependent variable
logreg = LogisticRegression(C = 10000000) #no regularization
score_list = []
y = model_data['Above?_>50K']
for i in model_data:
    if i != 'Above?_>50K' or i != 'Above?_<=50K':
         X_train, X_test, y_train, y_test = train_test_split(model_data[i].values.reshape(-1,1), y, test_size=0.30, random_state=42)
         score_list.append(logreg.fit(X_train, y_train).score(X_test, y_test))

print('The minimum accuracy is:', min(score_list))

#The minimum accuracy on a test set was 0.73. I assume this is reasonably high and choose to keep all features
#############################################################################################################################################
#THE MODEL SELECTION PROCESS
#Select test and train set
X_train, X_test, y_train, y_test = train_test_split(model_data.drop(['Above?_>50K','Above?_<=50K'], axis = 1), model_data['Above?_>50K'], test_size=0.30, random_state=42)
#train the model using grid search to find (1) proper regularization parameter (2) correct type of regularization (L1 OR L2)
logreg = LogisticRegression()
parameters = {'penalty':('l1', 'l2'), 'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
scores = ['precision', 'recall', 'f1']
best_param = pd.DataFrame(index=['score', 'C', 'penalty'], columns = scores)
for score in scores:
    gs = GridSearchCV(logreg, parameters, cv=5, scoring = score)
    gs.fit(X_train, y_train)
    best_param.loc['score', score] = gs.best_score_
    best_param.loc['C', score] = gs.best_params_['C']
    best_param.loc['penalty', score] = gs.best_params_['penalty']
#############################################################################################################################################
#TESTING OF THE BEST MODEL
#the best models found using the different scoring metrics
print(best_param)
#the best model is determined by f1 score
bestreg = LogisticRegression(C = best_param.loc['C','f1'], penalty = best_param.loc['penalty','f1'])
#fit on test data and get a score
print('Accuracy of the best model on test data:', bestreg.fit(X_test, y_test).score(X_test, y_test))

#Model features 85% accuracy on testing data