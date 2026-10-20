import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# ==============================
# Load Dataset
# ==============================

data = pd.read_csv("../dataset/phishing.csv")

print("Dataset Loaded Successfully")

print(data.head())


# ==============================
# Separate Features and Label
# ==============================

X = data.drop("Result", axis=1)

y = data["Result"]


# Convert labels
# -1 = Phishing
#  1 = Legitimate

y = y.replace(-1, 0)


# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==============================
# Create Models
# ==============================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),


    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),


    "SVM": SVC(
        kernel="linear",
        probability=True
    )

}


# Create models folder

os.makedirs("../model", exist_ok=True)


results = {}


# ==============================
# Training and Saving Models
# ==============================

for name, model in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")


    # Train model
    model.fit(
        X_train,
        y_train
    )


    # Prediction

    y_pred = model.predict(
        X_test
    )


    # Accuracy

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    results[name] = accuracy


    print(
        "Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )


    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )


    # Save model

    file_name = (
        name.lower()
        .replace(" ", "_")
        + "_url_model.pkl"
    )


    path = "../model/" + file_name


    joblib.dump(
        model,
        path
    )


    print(
        "Model Saved:",
        path
    )


# ==============================
# Model Comparison
# ==============================

print("\n\n========== MODEL COMPARISON ==========")


for model_name, accuracy in results.items():

    print(
        model_name,
        ":",
        round(accuracy * 100, 2),
        "%"
    )


# ==============================
# Best Model Selection
# ==============================

best_algorithm = max(
    results,
    key=results.get
)


best_accuracy = results[best_algorithm]


print("\n========== BEST MODEL ==========")

print(
    "Best Algorithm:",
    best_algorithm
)

print(
    "Accuracy:",
    round(best_accuracy * 100, 2),
    "%"
)