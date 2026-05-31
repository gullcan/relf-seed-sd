import numpy as np
import pandas as pd

from src.data.dataset import SEEDSDDataset

def flatten_eeg_feature(
        eeg_feature: np.ndarray
) -> np.ndarray:
    return eeg_feature.reshape(-1)    

def build_multimodal_feature(
        eeg_feature: np.ndarray,
        eye_feature: np.ndarray
) -> np.ndarray:
    eeg_flat = flatten_eeg_feature(eeg_feature)
    multimodal_feature = np.concatenate(
        [eeg_flat, eye_feature],
        axis=0
    )

    return multimodal_feature

def load_features_for_splits(
        split_index_csv_path: str,
        fold: int,
        split: str,
        max_samples: int | None = None
):
    split_index_df = pd.read_csv(split_index_csv_path)
    selected_df = split_index_df[
        (split_index_df["fold"] == fold)
         & (split_index_df["split"] == split)
        ].copy()
    
    if max_samples is not None:
        selected_df = selected_df.head(max_samples)

    dataset = SEEDSDDataset(
        index_csv_path="D:/ReLF/data/reports/full_dataset_index.csv"
    )
    features = []
    labels = []

    for _, row in selected_df.iterrows():
        sample_idx = row.name % len(dataset)

        sample = dataset[sample_idx]

        feature_vector = build_multimodal_feature(
            sample["eeg"].numpy(),
            sample["eye"].numpy()
        )
        features.append(feature_vector)
        labels.append(int(sample["label"]))
    X = np.stack(features)
    y = np.array(labels, dtype=np.int64)

    return X, y        


if __name__ == "__main__":

    dummy_eeg = np.random.rand(5,62)
    flattened = flatten_eeg_feature(dummy_eeg)

    print(dummy_eeg.shape)
    print(flattened.shape)

    dummy_eye = np.random.rand(50)
    multimodal = build_multimodal_feature(
        dummy_eeg,
        dummy_eye
    )
    print(dummy_eye.shape)
    print(multimodal.shape)

    print("\n--- Load Features For Split Test ---")

    X, y = load_features_for_splits(
        split_index_csv_path="D:/ReLF/data/reports/subject_dependent_split_index.csv",
        fold=0,
        split="train",
        max_samples=1000
    )

    print(X.shape)
    print(y.shape)
    print(np.unique(y, return_counts=True))