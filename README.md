# EXNO:4-DS
## AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

## ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

## FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

## FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```py

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```

![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/85a89809-3575-48a0-b481-e74f182aa644)


```py
data.isnull().sum()
```
![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/b666128b-c521-4c26-8f1b-b68b3fd9a2e6)

```py
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/c6c68d16-3ed3-4632-ab8d-0d8e06dfaeaa)

```py
data2=data.dropna(axis=0)
data2
```

![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/7cf2faae-a9b5-4bd1-86f9-7fb1024c634d)


```py
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/fbb85372-cd78-4211-8236-b525b0e4386b)

```py
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/869c4c16-5ca1-4805-b2fb-d133e03faa30)


```py
data2
```
![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/4cbd7f01-88ef-4fa9-831c-364261b589cf)

```py
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/f88a7595-5925-4959-beac-254a652a12ce)


```py
columns_list=list(new_data.columns)
print(columns_list)
```

![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/70564c9b-e2ff-4069-a34f-08703d86e2aa)


```py
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/dc2b5d92-560a-461d-8ee1-b080e33f06d1)


```py
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/dc1faedd-7175-4c05-95fe-0a4e56593ad7)


```py
x=new_data[features].values
print(x)
```

![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/79bc69b5-d55d-447a-9532-a1170406b78e)


```py
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/60079b52-29e1-472d-8372-e8dba50889cb)


```py
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/d78d3964-8004-4f92-af3f-f2283a005781)


```py
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/f0f215d1-1093-4cb3-96d2-e3813489ede4)


```py
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/2d996958-0402-4c43-86ca-8ae8e63c893d)

```py
data.shape
```
![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/34da301a-0172-4a59-bec4-c66ff85e1521)

```py
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

```
![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/60eda8cd-0e8c-4678-91ed-7b573d48c95e)


```py
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/8195e759-6fcf-4b01-b0e8-98ea2a79a0ab)


```py
tips.time.unique()
```
![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/6d75f9e9-6a1f-4cb2-885b-8e93d7dfcd72)

```py
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/ad06ab22-e701-467a-86e8-616519b226dd)


```py
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

![image](https://github.com/PSriVarshan/EXNO-4-DS/assets/114944059/51ae89c5-e6b3-49c3-9de2-e06f0922c100)


# RESULT:

### Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
