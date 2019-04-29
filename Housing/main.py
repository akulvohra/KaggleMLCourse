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
clf = DecisionTreeRegressor(random_state=1)
clf.fit(x, y) #Trains using Decision Tree Regressor
from sklearn.metrics import mean_absolute_error
predicted_home_prices = clf.predict(x) #these are the predicted values
print("Train data accuracy: ", mean_absolute_error(y, predicted_home_prices)) # gives the MAE between two versions

from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 1) # creates new values for a train test train_test_split
clf.fit(train_x, train_y)
pred_y = clf.predict(val_x)
print("Test data accuracy: ", mean_absolute_error(pred_y, val_y))