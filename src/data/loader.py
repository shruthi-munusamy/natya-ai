"""Dataset loading for the Samarth Bharatanatyam mudra dataset."""
from pathlib import Path
import yaml
from datasets import load_dataset


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_mudra_dataset(cache_dir: str | None = None):
    """Load the Samarth Bharatanatyam mudra dataset from HuggingFace.

    Returns a DatasetDict with a 'train' split. Each sample is
    {'image', 'label', 'gesture_type', 'label_id'}.
    """
    config = load_config()
    cache_dir = cache_dir or config["data"]["cache_dir"]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    return load_dataset(config["data"]["dataset_id"], cache_dir=cache_dir)


if __name__ == "__main__":
    ds = load_mudra_dataset()
    train = ds["train"]
    print(f"Splits: {list(ds.keys())}")
    print(f"Num samples: {len(train)}")
    print(f"Features: {train.features}")
    print(f"First sample: label={train[0]['label']}, "
          f"gesture_type={train[0]['gesture_type']}, "
          f"label_id={train[0]['label_id']}")

    unique_labels = set(train["label"])
    unique_types = set(train["gesture_type"])
    print(f"Num unique labels: {len(unique_labels)}")
    print(f"Unique gesture_types: {unique_types}")
