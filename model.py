import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


path = kagglehub.dataset_download("mrsimple07/stock-price-prediction")
print("Path to dataset files:", path)

file_path = path + "/stock_prices.csv"
df = pd.read_csv(file_path)


print(df.head())


# Let's predict 'Close' price using ['Open', 'High', 'Low', 'Volume']
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'


df.dropna(subset=features + [target], inplace=True)

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)


dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)


def evaluate_model(name, y_test, preds):
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name} Model:")
    print(f"  - Mean Squared Error: {mse:.2f}")
    print(f"  - R² Score: {r2:.4f}\n")

evaluate_model("Linear Regression", y_test, lr_preds)
evaluate_model("Decision Tree", y_test, dt_preds)
evaluate_model("Random Forest", y_test, rf_preds)


plt.figure(figsize=(14,5))

plt.plot(y_test.values[:50], label='Actual', marker='o')
plt.plot(lr_preds[:50], label='Linear Regression', linestyle='--')
plt.plot(dt_preds[:50], label='Decision Tree', linestyle='--')
plt.plot(rf_preds[:50], label='Random Forest', linestyle='--')

plt.title('Stock Price Prediction Comparison (First 50 test samples)')
plt.xlabel('Sample Index')
plt.ylabel('Stock Close Price')
plt.legend()
plt.show()
