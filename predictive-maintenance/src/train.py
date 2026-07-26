# Training script for predictive maintenance model

if __name__ == '__main__':
    print('Train script placeholder')


import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load dataset (NASA turbofan dataset or any CSV)
data = pd.read_csv("predictive-maintenance/data/engine_data.csv")  # replace with your dataset

# Step 2: Split features and labels
X = data.drop("label", axis=1)   # sensor readings
y = data["label"]                # failure/no failure

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Train XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Save model
joblib.dump(model, "predictive-maintenance/models/xgb_v1.pkl")

