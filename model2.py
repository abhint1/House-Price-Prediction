import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# Load the dataset
df = pd.read_csv('Housing.csv')

# Preprocess the data
numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', LinearRegression())])

model.fit(X_train, y_train)
# 6000,4,1,2,yes,no,yes,no,no,2,no,semi-furnished
# Predict the price for the input
input_data = [[8000, 4, 1, 2, 'yes', 'yes', 'yes', 'yes', 'yes', 2, 'yes', 'semi-furnished']]
input_df = pd.DataFrame(input_data, columns=X.columns)
y_pred = model.predict(input_df)

print('Predicted price:', y_pred[0])
# Take input from the user
# area = int(input("Enter the area of the house: "))
# bedrooms = int(input("Enter the number of bedrooms: "))
# bathrooms = int(input("Enter the number of bathrooms: "))
# stories = int(input("Enter the number of stories: "))
# mainroad = input("Is the house located near a main road? (yes/no): ")
# guestroom = input("Does the house have a guest room? (yes/no): ")
# basement = input("Does the house have a basement? (yes/no): ")
# hotwaterheating = input("Does the house have hot water heating? (yes/no): ")
# airconditioning = input("Does the house have air conditioning? (yes/no): ")
# parking = int(input("Enter the number of parking spots: "))
# prefarea = input("Is the house located in a preferred area? (yes/no): ")
# furnishingstatus = input("Enter the furnishing status (furnished, semifurnished, unfurnished): ")

# # Create a DataFrame with the user input
# user_input = [[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]]

# userinput = pd.DataFrame(user_input, columns=X.columns)
# y_pred = model.predict(userinput)



























































































#MODEL 2








# import pandas as pd
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# # Load the dataset
# df = pd.read_csv('Housing.csv')

# # Preprocess the data
# numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
# numeric_transformer = Pipeline(steps=[
#     ('scaler', StandardScaler())])

# categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)])

# X = df.drop('price', axis=1)
# y = df['price']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('regressor', LinearRegression())])

# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # Evaluate the performance of the model
# from sklearn.metrics import mean_squared_error, r2_score
# print('MSE:', mean_squared_error(y_test, y_pred))
# print('R^2:', r2_score(y_test, y_pred))
