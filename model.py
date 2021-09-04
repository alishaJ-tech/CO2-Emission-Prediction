import numpy as np
import pandas as pd
import pickle

data=pd.read_csv('CO2 Emissions_Canada.csv')
Data= data[['Engine Size(L)',
 'Cylinders',
 'Fuel Consumption City (L/100 km)',
 'Fuel Consumption Hwy (L/100 km)',
 'Fuel Consumption Comb (L/100 km)',
 'Fuel Consumption Comb (mpg)',
 'CO2 Emissions(g/km)']]

X = Data.drop('CO2 Emissions(g/km)', axis = 1)
y = Data['CO2 Emissions(g/km)']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(X_train, y_train)


pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))


