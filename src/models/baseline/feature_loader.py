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