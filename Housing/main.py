import pandas as pd
melbourne_file_path = 'melb_data.csv' # Location of the nelb_data file is currently in the same place
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data = melbourne_data.dropna(axis=0) # Drops Missing Values from DF
# print(melbourne_data)
melbourne_predicted = ['Price']
y = melbourne_data[melbourne_predicted] # y is equaled to the prediction target (whatmodel is trying topredict)
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']  # Creating a lost of features that are going to be used
x = melbourne_data[melbourne_features] #fitting features to df
#print(x.head())
from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor(random_state=0)
clf.fit(x, y) #Trains using Decision Tree Regressor
print("Making predictions for the following 5 houses:")
print(x.tail())
print("The predictions are")
print(clf.predict(x.tail()))
print(y.tail()) # comparing prediction to actual
