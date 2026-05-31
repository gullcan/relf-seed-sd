import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.baseline.feature_loader import load_features_for_split

def train_logistic_regression_on_fold(
        fold: int,
        max_train_samples: int | None = None,
        max_test_samples: int | None = None,
):
    split_index_path = "D:/ReLF/data/reports/subject_dependent_split_index.csv"

    print(f"Loading train features for fold {fold}...")

    X_train, y_train = load_features_for_split(
        split_index_csv_path=split_index_path,
        fold=fold,
        split="train",
        max_samples=max_train_samples,
    )
    print(f"Loading test features for fold {fold}...")

    X_test, y_test = load_features_for_split(
        split_index_csv_path=split_index_path,
        fold=fold,
        split="test",
        max_samples=max_test_samples,
    )
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=100,
                n_jobs=-1,
                verbose=1,
            )),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    weighted_precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    results = {
        "fold": fold,
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "weighted_precision": weighted_precision,
    }

    return results


if __name__ == "__main__":

    results = train_logistic_regression_on_fold(
        fold=0,
        max_train_samples=5000,
        max_test_samples=2000,
    )
    print("\nResults")
    print(results)
