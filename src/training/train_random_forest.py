import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import confusion_matrix

from src.models.baseline.feature_loader import load_features_for_split

def train_random_forest_on_fold(
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

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        verbose=1,
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
    print("\nClass Distribution (Test)")
    print(np.bincount(y_test))

    print("\nClass Distribution (Prediction)")
    print(np.bincount(y_pred))
    return results


if __name__ == "__main__":

    all_results = []

    for fold in [0,1,2]:
        print("\n======================")
        print(f"Training fold {fold}")
        print("========================")

        results = train_random_forest_on_fold(
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
    print(f"Accuracy: {np.mean(accuracies):.6f} ± {np.std(accuracies):.4f}")
    print(f"Weighted F1: {np.mean(f1_scores):.6f} ± {np.std(f1_scores):.4f}")
    print(f"Weighted Precision: {np.mean(precisions):.6f} ± {np.std(precisions):.4f}")    

    results_df = pd.DataFrame(all_results)

    output_dir = Path("D:/ReLF/reports/baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "random_forest_subject_dependent_subset.csv"

    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")