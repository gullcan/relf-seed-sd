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

    all_results = []
    import pandas as pd
    from pathlib import Path

    for fold in [0,1,2]:
        print(f"\n========================")
        print(f"Training fold {fold}")
        print(f"==========================")

        results = train_logistic_regression_on_fold(
            fold=fold,
            max_train_samples=5000,
            max_test_samples=2000,
        )
        all_results.append(results)
        print(results)

    accuracies = [result["accuracy"] for result in all_results]
    f1_scores = [result["weighted_f1"] for result in all_results]
    precisions = [result["weighted_precision"] for result in all_results]    

    print("\nFinal 3-Fold Results")
    print(f"Accuracy: {np.mean(accuracies):4f} ± {np.std(accuracies):.4f}")
    print(f"Weighted F1: {np.mean(f1_scores):4f} ± {np.std(f1_scores):.4f}")
    print(f"Weighted Precision: {np.mean(precisions):4f} ± {np.std(precisions):.4f}")

    results_df = pd.DataFrame(all_results)

    output_dir = Path("D:/ReLF/reports/baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "logistic_regression_subject_dependent_subset.csv"

    results_df.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")
    

    
