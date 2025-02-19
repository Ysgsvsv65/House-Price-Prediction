# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the Dataset
data = pd.read_csv('house_prices.csv')
print(data.head())

# Step 3: Exploratory Data Analysis (EDA)
print(data.isnull().sum())
print(data.describe())
sns.histplot(data['Price'], kde=True)
plt.title('Distribution of House Prices')
plt.show()

# Step 4: Data Preprocessing
data.fillna(data.median(), inplace=True)
data = pd.get_dummies(data, drop_first=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['Area', 'Bedrooms']] = scaler.fit_transform(data[['Area', 'Bedrooms']])

# Step 5: Feature Engineering
data['PricePerSqFt'] = data['Price'] / data['Area']

# Step 6: Split the Data
X = data.drop('Price', axis=1)
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Build and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 9: Visualize Results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()

# Step 10: Save the Model (Optional)
import pickle
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)