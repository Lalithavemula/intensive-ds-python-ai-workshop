import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Setup
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# ------------------------------------
# 🔹 1. Load and Preprocess Data
# ------------------------------------
def load_heart_data():
    url = "https://raw.githubusercontent.com/aqlopes/Kaggle-s-Heart-Disease-UCI-Dataset-Project/master/heart.csv"
    df = pd.read_csv(url)
    print("✅ Dataset loaded successfully.\n")
    return df

# ------------------------------------
# 🔹 2. Descriptive Statistics
# ------------------------------------
def descriptive_analysis(df):
    print("\n====== DESCRIPTIVE ANALYSIS ======\n")
    print(df.describe(include='all').T)
    print("\n🔹 Value Counts:")
    for col in df.columns:
        print(f"\n{col}:\n{df[col].value_counts()}")

# ------------------------------------
# 🔹 3. EDA Charts
# ------------------------------------
def eda_charts(df, output_dir="heart_eda_charts"):
    os.makedirs(output_dir, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], kde=True, bins=30, color='salmon')
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_hist.png")
        plt.close()

    if len(num_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
        plt.close()

    print(f"📊 EDA charts saved in: {output_dir}")

# ------------------------------------
# 🔹 4. Predictive Modeling
# ------------------------------------
def predictive_analysis(df):
    print("\n====== PREDICTIVE ANALYSIS ======\n")
    target_col = "target"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Evaluation
    print(f"\n✅ Accuracy: {accuracy_score(y_test, preds):.2f}")
    print("\n🔍 Classification Report:\n", classification_report(y_test, preds))
    print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, preds))

    # Feature Importance
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print("\n🏆 Top 5 Important Features:\n", importances.sort_values(ascending=False).head(5))

    return importances

# ------------------------------------
# 🔹 5. Prescriptive Analysis
# ------------------------------------
def prescriptive_analysis(importances):
    print("\n====== PRESCRIPTIVE ANALYSIS ======\n")
    top_features = importances.sort_values(ascending=False).head(3).index.tolist()
    print("📌 Focus on these key features for predicting heart disease:")
    for feat in top_features:
        print(f"🔹 {feat}")

# ------------------------------------
# 🔹 6. Main Function
# ------------------------------------
def main():
    df = load_heart_data()
    descriptive_analysis(df)
    eda_charts(df)
    importances = predictive_analysis(df)
    prescriptive_analysis(importances)

# ------------------------------------
# 🔹 7. Run Script
# ------------------------------------
if __name__ == "__main__":
    main()
