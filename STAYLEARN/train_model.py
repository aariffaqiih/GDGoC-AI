import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
df = pd.read_csv("dropout_dataset_clean.csv")
X = df.drop("dropout", axis=1)
y = df["dropout"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
log_features = ["family_income", "distance_to_institute"]
categorical_features = ["location_type"]
numeric_features = [
    "financial_aid_status",
    "internet_connectivity_issues",
    "motivation_score",
    "career_alignment",
    "stress_levels",
    "family_support",
    "attendance_rate",
    "test_scores_avg",
    "backlogs",
    "teaching_quality_rating"
]
log_transformer = Pipeline(steps=[
    ("log", FunctionTransformer(np.log1p, validate=False)),
    ("scaler", StandardScaler())
])
preprocessor = ColumnTransformer(
    transformers=[
        ("log", log_transformer, log_features),
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
joblib.dump(pipeline, "models/staylearn_model.joblib")
print("Model saved to models/staylearn_model.joblib")