import pandas as pd

def create_subject_dependent_clip_folds(
        clip_label_df: pd.DataFrame,
        n_folds: int = 3
) -> dict:
    
    folds = {}

    for fold_idx in range(n_folds):
        folds[fold_idx] = {
            "train_clips": [],
            "test_clips": [],
        }

    for emotion in sorted(clip_label_df["Emotion"].unique()):
        emotion_clips = (
            clip_label_df[clip_label_df["Emotion"] == emotion]
            ["clip"]
            .tolist()
        )    
        for fold_idx in range(n_folds):
            start = fold_idx * 2
            end = start + 2

            test_clips = emotion_clips[start:end]
            train_clips =[
                clip for clip in emotion_clips
                if clip not in test_clips
            ]

            folds[fold_idx]["test_clips"].extend(test_clips)
            folds[fold_idx]["train_clips"].extend(train_clips)
    return folds        


def summarize_fold_emotions(
        fold: dict,
        clip_label_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    clip_to_emotion = (
        clip_label_df
        .set_index("clip")["Emotion"]
        .to_dict()
    )

    for fold_idx, split in folds.items():
        for clip in split["test_clips"]:
            rows.append({
                "fold": fold_idx,
                "clip": clip,
                "emotion": clip_to_emotion[clip],
            })
    summary_df = pd.DataFrame(rows)
    return summary_df        


def apply_clip_folds_to_index(
        index_df: pd.DataFrame,
        folds: dict
) -> pd.DataFrame:

    split_rows = []
    for fold_idx, split in folds.items():
        train_clips = set(split["train_clips"])
        test_clips = set(split["test_clips"])

        fold_df = index_df.copy()
        fold_df["fold"] = fold_idx

        fold_df["split"] = fold_df["clip"].apply(
            lambda clip: "test" if clip in test_clips else "train"
        )
        split_rows.append(fold_df)

    split_index_df = pd.concat(
        split_rows,
        ignore_index=True
    )

    return split_index_df    


def save_subject_dependent_split_index(
        output_path: str = "D:/ReLF/data/reports/subject_dependent_split_index.csv"
):
    from src.data.build_dataset import load_clip_labels
    clip_label_df = load_clip_labels()
    folds = create_subject_dependent_clip_folds(
        clip_label_df=clip_label_df
    )
    index_df = pd.read_csv("D:/ReLF/data/reports/full_dataset_index.csv")

    split_index_df = apply_clip_folds_to_index(
        index_df=index_df,
        folds=folds
    )
    split_index_df.to_csv(output_path, index=False)

    print(f"Saved subject-dependent split index to: {output_path}")
    print(f"Rows: {len(split_index_df)}")





if __name__ == "__main__":
    from src.data.build_dataset import load_clip_labels
    clip_label_df = load_clip_labels()
    folds= create_subject_dependent_clip_folds(clip_label_df)

    for fold_idx, split in folds.items():
        print(f"\nFold {fold_idx}")
        print(f"Train clips: {len(split['train_clips'])}")
        print(f"Test clips: {len(split['test_clips'])}")
        print(f"Test clips: {split['test_clips']}")

    print("\nFold emotion summary:")
    summary_df = summarize_fold_emotions(folds, clip_label_df)
    print(summary_df)

    print("\nEmotion counts per fold:")
    print(
        summary_df
        .groupby(["fold","emotion"])
        .size()
    )
    print("\nApply folds to full dataset index:")

    index_df = pd.read_csv("D:/ReLF/data/reports/full_dataset_index.csv")
    split_index_df = apply_clip_folds_to_index(
        index_df=index_df,
        folds=folds
    )
    print(split_index_df.head())
    print(split_index_df.shape)

    print("\nSplit counts:")
    print(
        split_index_df
        .groupby(["fold", "split"])
        .size()
    )
    print("\nEmotion counts by fold and split:")
    print(
        split_index_df
        .groupby(["fold", "split", "emotion"])
        .size()
    )    

    print("\nSave subject-dependent split index:")
    save_subject_dependent_split_index()