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