import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


 
# 1. LOAD THE REAL DATASET
 

data = pd.read_csv(
    r"C:\Users\msi\Desktop\user.dx\MONKE\solarenergy.csv"
)


 
# 2. KEEP THE PART OF THE DATASET WITH WEATHER DATA
 

data = data.iloc[:2921].copy()


 
# 3. CONVERT DATETIME
 

data["Datetime"] = pd.to_datetime(
    data["Datetime"],
    dayfirst=True,
    errors="coerce"
)


 
# 4. SELECT WEATHER FEATURES
 

weather_features = [
    "wind-direction",
    "wind-speed",
    "humidity",
    "average-wind-speed-(period)",
    "average-pressure-(period)",
    "temperature"
]

target_name = "solar_mw"


 
# 5. CONVERT DATA TO NUMBERS
 

for column in weather_features + [target_name]:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


 
# 6. REMOVE BAD ROWS
 

data = data.dropna(
    subset=["Datetime"] + weather_features + [target_name]
).copy()


 
# 7. SORT CHRONOLOGICALLY
 

data = data.sort_values(
    "Datetime"
).reset_index(drop=True)



print("\nTime range:")
print(data["Datetime"].min())
print("to")
print(data["Datetime"].max())


 
# 8. CREATE PREVIOUS-HOUR SOLAR OUTPUT


data["solar_previous_hour"] = data[
    target_name
].shift(1)


 
# 9. EXTRACT TIME INFORMATION
 

data["hour"] = data["Datetime"].dt.hour

data["day_of_year"] = data["Datetime"].dt.dayofyear


 
# 10. CREATE CYCLICAL TIME FEATURES
 

# Daily cycle

data["hour_sin"] = np.sin(
    2 * np.pi * data["hour"] / 24
)

data["hour_cos"] = np.cos(
    2 * np.pi * data["hour"] / 24
)


# Seasonal cycle

data["day_sin"] = np.sin(
    2 * np.pi * data["day_of_year"] / 365.25
)

data["day_cos"] = np.cos(
    2 * np.pi * data["day_of_year"] / 365.25
)


 
# 11. ADD DAILY HARMONICS
 

for k in range(2, 5):

    data[f"hour_sin_{k}"] = np.sin(
        2 * np.pi * k * data["hour"] / 24
    )

    data[f"hour_cos_{k}"] = np.cos(
        2 * np.pi * k * data["hour"] / 24
    )


 
# 12. REMOVE ROWS CREATED AS INVALID BY THE LAG
 

data = data.dropna(
    subset=["solar_previous_hour"]
).reset_index(drop=True)




 
# 13. DEFINE MODEL FEATURES
 

feature_names = weather_features + [

    "hour_sin",
    "hour_cos",

    "day_sin",
    "day_cos",

    "solar_previous_hour"
]


# Add daily harmonics

for k in range(2, 5):

    feature_names.append(
        f"hour_sin_{k}"
    )

    feature_names.append(
        f"hour_cos_{k}"
    )


print("\nFeatures used by the model:")

for feature in feature_names:

    print("-", feature)


 
# 14. CREATE X AND y
 

X = data[
    feature_names
].to_numpy(dtype=float)

y = data[
    target_name
].to_numpy(dtype=float)


print("\nX shape:", X.shape)
print("y shape:", y.shape)


 
# 15. CHRONOLOGICAL TRAIN / TEST SPLIT
 

#
# First 80% = training
# Last 20%  = testing
#


split_index = int(
    len(data) * 0.80
)


X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))



# 16. TRAIN MULTIPLE LINEAR REGRESSION


model = LinearRegression()

model.fit(
    X_train,
    y_train
)



# 17. MAKE PREDICTIONS


y_train_pred = model.predict(
    X_train
)

y_test_pred = model.predict(
    X_test
)



# 18. CALCULATE PERFORMANCE


train_r2 = r2_score(
    y_train,
    y_train_pred
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_test_pred
    )
)




# 19. PRINT RESULTS


print("\n==========================================")
print("MULTIPLE LINEAR REGRESSION RESULTS")
print("==========================================")

print(
    f"Training R²: {train_r2:.4f}"
)

print(
    f"Testing R²:  {test_r2:.4f}"
)

print(
    f"Testing MAE: {test_mae:.2f}"
)

print(
    f"Testing RMSE: {test_rmse:.2f}"
)



# 20. PRINT LEARNED COEFFICIENTS

print("\n==========================================")
print("LEARNED COEFFICIENTS")
print("==========================================")

for name, coefficient in zip(
    feature_names,
    model.coef_
):

    print(
        f"{name}: {coefficient:.4f}"
    )


print("\nIntercept:")
print(
    f"{model.intercept_:.4f}"
)



# 21. ACTUAL VS PREDICTED OVER TIME

test_dates = data[
    "Datetime"
].iloc[split_index:]


plt.figure(
    figsize=(14, 6)
)


plt.plot(
    test_dates,
    y_test,
    label="Actual Solar Output",
    linewidth=2
)


plt.plot(
    test_dates,
    y_test_pred,
    label="Predicted Solar Output",
    linewidth=2
)


plt.xlabel(
    "Date and Time"
)

plt.ylabel(
    "Solar Output"
)


plt.title(
    f"Actual vs Predicted Solar Output "
    f"(Test R² = {test_r2:.3f})"
)


plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()



# 22. ACTUAL VS PREDICTED SCATTER PLOT

plt.figure(
    figsize=(7, 7)
)


plt.scatter(
    y_test,
    y_test_pred,
    alpha=0.5
)


minimum = min(
    y_test.min(),
    y_test_pred.min()
)

maximum = max(
    y_test.max(),
    y_test_pred.max()
)


plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--",
    linewidth=2,
    label="Perfect Prediction"
)


plt.xlabel(
    "Actual Solar Output"
)

plt.ylabel(
    "Predicted Solar Output"
)


plt.title(
    f"Actual vs Predicted Solar Output\n"
    f"R² = {test_r2:.3f}"
)


plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


 
# 23. CALCULATE ERRORS
 

errors = (
    y_test - y_test_pred
)


 
# 24. ERROR / RESIDUAL PLOT
 

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    test_dates,
    errors,
    linewidth=1.5
)


plt.axhline(
    0,
    linestyle="--",
    linewidth=2
)


plt.xlabel(
    "Date and Time"
)

plt.ylabel(
    "Prediction Error"
)


plt.title(
    "Prediction Errors Over Time"
)


plt.grid(
    True,
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()
