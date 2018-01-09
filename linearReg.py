import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

data = pd.read_fwf('Brain_Body.txt')
x_values = data[['Brain']]
y_values = data[['Body']]

#train model on dataset
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)
 
#visualizations using matplotlib
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()