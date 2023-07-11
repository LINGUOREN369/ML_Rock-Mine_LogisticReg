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

**#ML7 Gold**
1. Random Forest Regressor
2. distplot
3. error_score
4. 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
