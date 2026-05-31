from torch.utils.data import DataLoader

from src.data.dataset import SEEDSDDataset

dataset = SEEDSDDataset(
    index_csv_path="D:/ReLF/data/reports/full_dataset_index.csv"
    )

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

first_batch = next(iter(dataloader))

print(type(first_batch))
print(first_batch.keys())

print(first_batch["eeg"].shape)
print(first_batch["eye"].shape)
print(first_batch["label"].shape)