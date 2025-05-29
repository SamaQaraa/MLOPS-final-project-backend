import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import mlflow
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

path = "data/hand_landmarks_data.csv"


def preprocess_data(path):
    df = pd.read_csv(path)
    df_scaling = df.copy()

    wrist_x = df_scaling['x1'].copy()
    wrist_y = df_scaling['y1'].copy()

    # Recenter the hand landmarks (x,y) to make the origin the wrist point
    for i in range(1, 22):
        df_scaling[f'x{i}'] = df_scaling[f'x{i}'] - wrist_x
        df_scaling[f'y{i}'] = df_scaling[f'y{i}'] - wrist_y

    # Divide all the landmarks by the mid-finger tip position.
    mid_finger_tip_position = np.sqrt(df_scaling['x13']**2 + df_scaling['y13']**2)
    for i in range(1, 22):
        df_scaling[f'x{i}'] = df_scaling[f'x{i}'] / mid_finger_tip_position
        df_scaling[f'y{i}'] = df_scaling[f'y{i}'] / mid_finger_tip_position

    features = df_scaling.drop("label", axis=1)
    labels = df_scaling["label"]
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, random_state=100, train_size=.8, test_size=.2
    )
    return features_train, features_test, labels_train, labels_test


def train_and_log_model(
    model_name, model_instance, params, X_train, y_train, X_test, y_test
):
    """
    Trains a model (with GridSearchCV if parameters are provided), logs it,
    and records metrics and a confusion matrix in MLflow.
    """
    with mlflow.start_run(run_name=model_name):
        if params:
            grid_search = GridSearchCV(
                estimator=model_instance, param_grid=params, cv=3, scoring='f1_macro', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            mlflow.log_params(grid_search.best_params_)
        else:
            model = model_instance
            model.fit(X_train, y_train)

        # Log the model with the input and output schema
        signature = mlflow.models.infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(
            model, model_name.replace(" ", "_"), signature=signature, input_example=X_train.iloc[0:1]
        )

        # Log the data artifact
        mlflow.log_artifact("hand_landmarks_data.csv")

        y_pred = model.predict(X_test)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average='macro', zero_division=0))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average='macro'))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='macro'))

        # Log tag
        mlflow.set_tag("model_type", model_name)

        # Plot and log confusion matrix
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Create the 'img' directory if it doesn't exist
        os.makedirs("./img", exist_ok=True)
        img_path = f"./img/confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(img_path)
        mlflow.log_artifact(img_path)
        plt.show() # Display the plot

def main():
    # Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    # Set the experiment name
    mlflow.set_experiment("hand_landmarks_prediction")

    X_train, X_test, y_train, y_test = preprocess_data(path)

    # Define models and their hyperparameters for GridSearchCV
    models_to_train = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {'max_iter': [500, 1000, 2000]} # Example param for LR
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        },
        "Support Vector Machine": {
            "model": SVC(class_weight='balanced', random_state=42),
            "params": {
                'kernel': ['rbf'],
                'C': [0.01, 0.1, 1, 10, 100],
                'gamma': [0.01, 0.1, 1, 10, 100]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(class_weight='balanced', random_state=42),
            "params": {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        },

    }

    for name, config in models_to_train.items():
        train_and_log_model(name, config["model"], config["params"], X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()