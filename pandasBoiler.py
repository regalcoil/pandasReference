#pip install numpy
#pip install pandas
#pip install sklearn
#pip install matplotlib
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import statsmodels.api as sm

#importing csv file
df = pd.read_csv("cleaned.csv");

df2 = pd.DataFrame(df)

#setting independent and dependent variables
X = df[['income', 'metro_status', 'education']]
y = df['numvisits']

#perform a covariate analysis 
X, y = np.array(X), np.array(y)
#add a constant to the independent variables
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

#scatter chart using pyplot
#plt.scatter(df['education'], df['numvisits'], color='red')
#plt.title('chart', fontsize=14)
#plt.xlabel('education', fontsize=14)
#plt.ylabel('numvisits', fontsize=14)
#plt.grid(True)
#plt.show()

#predictive output from a given input
#regr = linear_model.LinearRegression()
#regr.fit(X, y)
#predictedV = regr.predict([[16, 2, 45, 40]])
#print(predictedV)

exit()

