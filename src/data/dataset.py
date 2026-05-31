from pathlib import Path
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from src.utils.paths import get_dataset_root

class SEEDSDDataset(Dataset):

    def __init__(
            self,
            index_csv_path: str
    ):
        self.index_df = pd.read_csv(index_csv_path)
        self.dataset_root = get_dataset_root()


    def __len__(self):
        return len(self.index_df)
    

    def __getitem__(self, idx: int):
        row = self.index_df.iloc[idx]

        session = row["session"]
        subject_file = row["subject_file"]
        clip = row["clip"]
        window_index = int(row["window_index"])
        label = int(row["label"])

        eeg_path = self.dataset_root / "eeg_features" / session / subject_file
        eye_path = self.dataset_root / "eye_features" / session / subject_file

        eeg_data = np.load(eeg_path, allow_pickle=True).item()
        eye_data = np.load(eye_path, allow_pickle=True).item()

        eeg = eeg_data[clip][window_index]
        eye = eye_data[clip][window_index]

        sample = {
            "eeg": eeg,
            "eye": eye,
            "label": label,
            "session": session,
            "subject_file": subject_file,
            "clip": clip,
            "window_index": window_index,
        }
        return sample

        


if __name__ == "__main__":

    dataset = SEEDSDDataset(
        index_csv_path="D:/ReLF/data/reports/full_dataset_index.csv"
    )
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]

    print(f"EEG shape: {sample['eeg'].shape}")
    print(f"Eye shape: {sample['eye'].shape}")
    print(f"Label: {sample['label']}")
    print(f"Session: {sample['session']}")
    print(f"Subject file: {sample['subject_file']}")
    print(f"Clip: {sample['clip']}")
    print(f"Window index: {sample['window_index']}")