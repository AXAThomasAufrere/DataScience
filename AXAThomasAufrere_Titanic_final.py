# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:08:34 2017

@author: thomas.aufrere
"""

#Titanic main code

#Import section
import os
import pandas as pd
import numpy as np
import csv as csv
import pylab as plt


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot

# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---

%matplotlib inline
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

pd.options.display.max_rows = 100

#Directory
os.chdir('\\\NR5VFL01.hk.intraxa\\thomas.aufrere$\\Documents\\1. AARO\\Data Science\\')

# Load the train data file
train_data = pd.read_csv('train.csv', header=0)
train_data.head(n=10)
train_data = train_data.set_index('PassengerId')
train_data.head(n=10)
train_data.describe()

# Load the test data file
test_data = pd.read_csv('test.csv', header=0)
test_data.head(n=10)
test_data = test_data.set_index('PassengerId')
test_data.head(n=10)

# Size of the sample: 891 rows 
# Variables: 
#   - PassengerId: id of the passenger (key)
#   - Survived: target, 0 or 1
#
#   - Name: name of the passenger (text)
#   - Cabin: cabin number of the passenger (text)   
#   - Ticket: ticket reference contains info on Embarked (text). Examples: 
#       * when starts with C or W or STON or SOTON  then Embarked at S
#       * when contains PARIS                       then Embarked at C
#       * when contains S.O.*                       then Embarked at S
#       * when starts with PC                       then Embarked at S
#       * when contains A/*                         then Embarked at S
#
#   - Sex: sex of the passenger, male of female (discrete)
#   - Embarked: where the passenger embarked, Cherbourg, Queenstown or Southampton (discrete, 2 missing)
#   - Pclass: class of the passenger, 1: high, 2: medium, 3: low (discrete ordered)
# 
#   - Age: age of the passenger (quantitative 177 missing values, ~20%)
#   - SibSp: number of siblings and spouse (quantitative)
#   - Parch: number of parents and childs (quantitative)
#   - Fare: price of the ticket (quantitative)

# Age
print train_data['Age'].value_counts()
print len(train_data['Age'][ train_data['Age'].isnull()] )
pyplot.hist(train_data['Age'].dropna(), bins=np.linspace(0, 100, 100), color='green', alpha=0.5)

# Replace NA values by median
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)


# Pclass
print train_data['Pclass'].value_counts()
print len(train_data['Pclass'][ train_data['Pclass'].isnull()] )
pyplot.hist(train_data['Pclass'].dropna(), bins=np.linspace(0, 10, 100), color='green', alpha=0.5)

# Sex
print train_data['Sex'].value_counts()
print len(train_data['Sex'][ train_data['Sex'].isnull()] )

# Embarked, 2 missing values
print train_data['Embarked'].value_counts()
print len(train_data['Embarked'][ train_data['Embarked'].isnull()] )

# Cabin
print train_data['Cabin'].value_counts()
print len(train_data['Cabin'][ train_data['Cabin'].isnull()] )


# Plot histogram depending on the survival
def plot_histogram(data, variable, bins=20):
    survived = data[data.Survived == 1]
    dead = data[data.Survived == 0]
    
    x1 = dead[variable].dropna()
    x2 = survived[variable].dropna()
    plt.hist( [x1,x2], label=['Dead','Survived'], color=['red','blue'], bins=bins)
    plt.legend(loc='upper left')
    plt.show()


plot_histogram(data=train_data, variable='Pclass')
plot_histogram(data=train_data[train_data['Sex']=='male'], variable='Pclass')
plot_histogram(data=train_data[train_data['Sex']=='female'], variable='Pclass')

plot_histogram(data=train_data, variable='Age')
plot_histogram(data=train_data[train_data['Sex']=='male'], variable='Age')
plot_histogram(data=train_data[train_data['Sex']=='female'], variable='Age')

plot_histogram(data=train_data, variable='Fare')
plot_histogram(data=train_data[train_data['Sex']=='male'], variable='Fare')
plot_histogram(data=train_data[train_data['Sex']=='female'], variable='Fare')

#histogram
survived_sex = train_data[train_data['Survived']==1]['Sex'].value_counts()
dead_sex = train_data[train_data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))

#embark
survived_embark = train_data[train_data['Survived']==1]['Embarked'].value_counts()
dead_embark = train_data[train_data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))

figure = plt.figure(figsize=(15,8))
plt.hist([train_data[train_data['Survived']==1]['Age'],train_data[train_data['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()

figure = plt.figure(figsize=(15,8))
plt.hist([train_data[train_data['Survived']==1]['Fare'],train_data[train_data['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()

#combined info scatter plot
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(train_data[train_data['Survived']==1]['Age'],train_data[train_data['Survived']==1]['Fare'],c='green',s=40)
ax.scatter(train_data[train_data['Survived']==0]['Age'],train_data[train_data['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)

#average fare per class
ax = plt.subplot()
ax.set_ylabel('Average fare')
train_data.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(15,8), ax = ax)

# Main comments
# * PClass
#       Overall: 
#           High probability of survival for PClass 1, 
#           High probability of death PClass 3
#       Drilldown by sex:
#           Male: have a higher survival rate for PClass 1
#           Female: have a high proba of survival, except for PClass 3
# * Age
#       Overall: Young and old ages have a high probability of survival
#       Drilldown by sex:
#           Male: have a higher survival rate only for young ages
#           Female: have consistently a high survival rate for all ages




#feature engineering

def status(feature):

    print 'Processing',feature,': ok'

#combined data
def get_combined_data():
    # reading train data
    train = pd.read_csv('train.csv')

    # reading test data
    test = pd.read_csv('test.csv')

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop('Survived',1,inplace=True)
    

    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)
    
    return combined


combined = get_combined_data()
combined.shape
combined.head()

#get titles

def get_titles():

    global combined
    
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)
    
get_titles()
combined.head()

#processing ages
grouped = combined.groupby(['Sex','Pclass','Title'])
grouped.median()

def process_age():
    
    global combined
    
    # a function that fills the missing values of the Age variable
    
    def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26
    
    combined.Age = combined.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)
    
    status('age')
    
process_age()
combined.info()

#process names

def process_names():
    
    global combined
    # we clean the Name variable
    combined.drop('Name',axis=1,inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')
    combined = pd.concat([combined,titles_dummies],axis=1)
    
    # removing the title variable
    combined.drop('Title',axis=1,inplace=True)
    
    status('names')
process_names()
combined.head()


def process_fares():
    
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined['Fare'].fillna(combined.Fare.mean(),inplace=True)
    
    status('fare')

process_fares()

print train['Fare']
print len(train['Fare'][ train['Fare'].isnull()] )
 

def process_embarked():
    
    global combined
    # two missing embarked values - filling them with the most frequent one (S)
    combined['Embarked'].fillna('S',inplace=True)
    
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop('Embarked',axis=1,inplace=True)
    
    status('embarked')
process_embarked()

def process_cabin():
    
    global combined
    
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U',inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix='Cabin')
    
    combined = pd.concat([combined,cabin_dummies],axis=1)
    
    combined.drop('Cabin',axis=1,inplace=True)
    
    status('cabin')
process_cabin()
combined.info()
combined.head()

def process_sex():
    
    global combined
    # mapping string values to numerical one 
    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
    
    status('sex')

process_sex()


def process_pclass():
    
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")
    
    # adding dummy variables
    combined = pd.concat([combined,pclass_dummies],axis=1)
    
    # removing "Pclass"
    
    combined.drop('Pclass',axis=1,inplace=True)
    
    status('pclass')

process_pclass()

def process_ticket():
    
    global combined
    

    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies],axis=1)
    combined.drop('Ticket',inplace=True,axis=1)

    status('ticket')
process_ticket()

combined.info()
combined.head()

def process_family():
    
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    
    status('family')
    
process_family()
combined.shape
combined.head()

def scale_all_features():
    
    global combined
    
    features = list(combined.columns)
    features.remove('PassengerId')
    combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)
    
    print 'Features scaled successfully !'

scale_all_features()


from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

def compute_score(clf, X, y,scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5,scoring=scoring)
    return np.mean(xval)


#split dataset into original train / test DB 

def recover_train_test_target():
    global combined
    
    train0 = pd.read_csv('train.csv')
    
    targets = train0.Survived
    train = combined.ix[0:890]
    test = combined.ix[891:]
    
    return train,test,targets

train,test,targets = recover_train_test_target()

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

#importance features through tree based estimator
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train, targets)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_

print features
features.sort(['importance'],ascending=False)

#transform dataset, dimension reduction
model = SelectFromModel(clf, prefit=True)
train_new = model.transform(train)
train_new.shape


test_new = model.transform(test)
test_new.shape

#random forest modelling
forest = RandomForestClassifier(max_features='sqrt')

parameter_grid = {
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': [200,210,240,250],
                 'criterion': ['gini','entropy']
                 }
#n_fold = 10 ==> risk of over fitting ?
cross_validation = StratifiedKFold(targets, n_folds=10)

grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(train_new, targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


# output 
output = grid_search.predict(test_new).astype(int)
df_output = pd.DataFrame()

print df_output
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
print df_output         
         
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)












































