from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.paths import get_dataset_root

def load_clip_labels() -> pd.DataFrame:
    dataset_root = get_dataset_root()
    stimulus_path = dataset_root / "SEED-SD_stimulation.xlsx"

    stimulus_df = pd.read_excel(stimulus_path)

    stimulus_clean = stimulus_df.dropna(subset=["Label"]).copy()
    stimulus_clean = stimulus_clean.reset_index(drop=True)

    label_map = {
        0: "Neutral",
        1: "Sad",
        2: "Fear",
        3: "Happy",
    }

    stimulus_clean["Label"] = stimulus_clean["Label"].astype(int)
    stimulus_clean["Emotion"] = stimulus_clean["Label"].map(label_map)

    stimulus_clean["clip"] = [
        f"clip_{i}"
        for i in range(1, len(stimulus_clean) + 1)
    ]

    return stimulus_clean[["clip", "Label", "Emotion", "Name of the clip"]]





def load_subject_features(session: str,subject_file: str):

    dataset_root = get_dataset_root()

    eeg_path = dataset_root / "eeg_features" / session / subject_file
    eye_path = dataset_root / "eye_features" / session / subject_file

    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")
    
    if not eye_path.exists():
        raise FileNotFoundError(f"Eye file not found: {eye_path}")
    
    eeg_data = np.load(eeg_path, allow_pickle=True).item()
    eye_data = np.load(eye_path, allow_pickle=True).item()

    return eeg_data, eye_data

# Bu fonksiyon tek bir katılımcı için EEG feature dictionary, Eye feature dictionary yükler. 


def build_subject_samples(session: str, subject_file: str) -> list[dict]:

    labels_df = load_clip_labels()
    label_lookup = labels_df.set_index("clip").to_dict(orient="index")

    eeg_data, eye_data = load_subject_features(session, subject_file)

    samples = []

    for clip_name in eeg_data.keys():
        eeg_clip = eeg_data[clip_name]
        eye_clip = eye_data[clip_name]

        if eeg_clip.shape[0] != eye_clip.shape[0]:
            raise ValueError(
                f"Window count mismatch for {session}/{subject_file}/{clip_name}:"
                f"EEG={eeg_clip.shape[0]}, Eye={eye_clip.shape[0]}"
            )
        label_info = label_lookup[clip_name]

        for window_index in range(eeg_clip.shape[0]):
            sample = {
                "eeg": eeg_clip[window_index],
                "eye": eye_clip[window_index],
                "label": label_info["Label"],
                "emotion": label_info["Emotion"],
                "clip": clip_name,
                "session": session,
                "subject_file": subject_file,
                "window_index": window_index,
            }
            samples.append(sample)

    return samples        

def build_full_dataset_index() -> pd.DataFrame:

    labels_df = load_clip_labels()
    label_lookup = labels_df.set_index("clip").to_dict(orient="index")

    dataset_root = get_dataset_root()

    sessions = ["session_1", "session_2", "session_3"]

    rows = []

    for session in sessions:
        eeg_dir = dataset_root / "eeg_features" / session
        eeg_files = sorted(eeg_dir.glob("*.npy"))

        for eeg_file in eeg_files:
            subject_file = eeg_file.name
            subject_id = subject_file.split("_")[0]

            eeg_data, eye_data = load_subject_features(
                session=session,
                subject_file=subject_file
            )

            for clip_name in eeg_data.keys():
                eeg_clip = eeg_data[clip_name]
                eye_clip = eye_data[clip_name]

                if eeg_clip.shape[0] != eye_clip.shape[0]:
                    raise ValueError(
                        f"Window count mismatch for {session}/{subject_file}/{clip_name}: "
                        f"EEG={eeg_clip.shape[0]}, Eye={eye_clip.shape[0]}"
                    )
                
                label_info = label_lookup[clip_name]

                for window_index in range(eeg_clip.shape[0]):
                    rows.append({
                        "session": session,
                        "subject_file": subject_file,
                        "subject_id": subject_id,
                        "clip": clip_name,
                        "window_index": window_index,
                        "label": label_info["Label"],
                        "emotion": label_info["Emotion"],
                    })
    index_df = pd.DataFrame(rows)
    return index_df                  

def save_full_dataset_index() -> Path:
    dataset_root = get_dataset_root()

    index_df = build_full_dataset_index()

    outputh_path = dataset_root.parent / "ReLF" / "data" / "reports" / "full_dataset_index.csv"
    outputh_path.parent.mkdir(parents=True, exist_ok=True)

    index_df.to_csv(outputh_path, index=False)

    print(f"Saved full dataset index to: {outputh_path}")
    print(f"Total samples: {len(index_df)}")

    return outputh_path


if __name__ == "__main__":
    labels= load_clip_labels()

    print(labels.head())
    print(labels["Emotion"].value_counts())

    print("\n--- Subject Feature Loading Test---")

    eeg_data, eye_data = load_subject_features(
        session="session_1",
        subject_file="sub10_20200903.npy"
    )

    print(f"EEG clips: {len(eeg_data)}")
    print(f"Eye clips: {len(eye_data)}")

    print(f"EEG clip_1 shape: {eeg_data['clip_1'].shape}")
    print(f"Eye clip_1 shape: {eye_data['clip_1'].shape}")

    print("\n--- Subject Sample Building Test ---")

    samples = build_subject_samples(
        session="session_1",
        subject_file="sub10_20200903.npy"
    )
    print(f"Number of samples: {len(samples)}")

    first_sample = samples[0]

    print(f"First EEG shape: {first_sample['eeg'].shape}")
    print(f"First Eye shape: {first_sample['eye'].shape}")
    print(f"First label: {first_sample['label']}")
    print(f"First emotion: {first_sample['emotion']}")
    print(f"First clip: {first_sample['clip']}")
    print(f"First window index: {first_sample['window_index']}")


    print("\n--- Full Dataset Index Test ---")

    index_df = build_full_dataset_index()

    print(index_df.head())
    print(index_df.shape)
    print(index_df["emotion"].value_counts())

    print("\n--- Save Full Dataset Index ---")
    save_full_dataset_index()


