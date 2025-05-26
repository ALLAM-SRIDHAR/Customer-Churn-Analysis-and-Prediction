import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN
import joblib

# Load clean data
df = pd.read_csv("tel_churn_clean.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Resample
X_resampled, y_resampled = SMOTEENN().fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Model
model = GradientBoostingClassifier(loss="log_loss", n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Save
joblib.dump(model, "Bi_gradient_boosting_model.joblib")
print("✅ Model saved successfully.")
