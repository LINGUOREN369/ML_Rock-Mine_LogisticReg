# Machine Learning Project Youtube
**#ML1: Mine and Rock:**
1. Logistic Regression
2. Split Data
3. Accuracy Test
4. Input Data

**#ML2: ClassifyDiabetes:**
1. SVM(Supporting Vector Machine) Classification Regression
2. Scalaring the data using the standard scaler
3. Accuracy_score
4. Groupby
5. Value_counts()

**#ML3: House Price Prediction** 
1. XGB regresson
2. Using R sqaure error and mean absolute error to test
3. Plotting data
4. Correlation of data
5. Heatmap

**#ML4: Fake news**
1. Logistic Regression to Predict Fake/True
2. Regular Expression
3. Stopwords
4. PorterStemmer
5. Find the sum of null: news_dataset.isnull().sum()
6. #replacing the null values with empty string news_dataset = news_dataset.fillna('')

**#ML5 LOAN**
1. SVM
2. Replace cat value to numerical
3. Drop columns

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import accuracy_score

loan_dataset = loan_dataset.dropna()
loan_dataset.replace({'Loan_Status':{'N':0, 'Y':1}}, inplace = True)
loan_dataset.replace({'Married':{'No':0, 'Yes':1},\
                     'Gender':{'Male':1, 'Female':0},\
                     'Self_Employed':{'No':0, 'Yes':1},\
                     'Property_Area':{'Rural': 0, 'Semiurban' : 1, 'Urban' : 2},\
                     'Education':{'Graduate':1, 'Not Graduate':0}},
                     inplace = True)

**#ML6 WINE**
1. RANDOM_FOREST
2. Plotting
3. heatmap
4. Lamda--> Y = wine_dataset['quality'].apply(lambda y_value:1 if y_value>=7 else 0)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

**#7 Car Price**
1. Use multiple regression on one probelm
2. Lasso regression

**#ML8 Gold**
1. Random Forest Regressor
2. distplot
3. error_score
4. 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

**#ML10 Credit Card**
clustering:
legit = credit_dataset[credit_dataset.Class == 0]
fraud = credit_dataset[credit_dataset.Class == 1]
Concat:
new_dataset = pd.concat([legit_sample, fraud], axis = 0)
Set sample size:
legit_sample = legit.sample(n=85)

**ML12 Big Mart Sales Prediction**
XGB regression
Catogorized data based on another column:
  mode_of_outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
  missing_values = big_mart_data['Outlet_Size'].isnull()
  big_mart_data.loc[missing_values, 'Outlet_Size'] = big_mart_data.loc[missing_values, "Outlet_Type"].apply(lambda x: mode_of_outlet_size[x])

replace null with mean values:
  weight_mean = big_mart_data['Item_Weight'].mean()
  big_mart_data['Item_Weight']. fillna(weight_mean, inplace = True)

encoder = LabelEncoder()
big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

big_mart_data.replace({'Item_Fat_Content': {'low fat':"Low Fat", 'LF':"Low Fat",'reg':'Regular'}}, inplace = True)

**ML13_Customer_Segmentation_K_mean**
1. K-means
2. #WCSS -> Within Clusters Sum of Squares
3. #PLOTING ALL THE CLUSTERS AND THEIR CENTROIDS
