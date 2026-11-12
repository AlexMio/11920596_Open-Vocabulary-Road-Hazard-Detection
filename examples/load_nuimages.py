# Code modified from https://huggingface.co/datasets/finedet/nuimages

from datasets import load_dataset
from transformers import AutoProcessor
import sys
import getopt
from torch.utils.data import DataLoader


if __name__ == "__main__":
    hf_access_token = False

    shortopts = "t:"
    longopts = ["hf_token="]

    # Get command-line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], shortopts, longopts)
    except getopt.GetoptError as err:
        print(f"Error: {err}")
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-t", "--hf_token"):
            # Passes the HuggingFace access token. This enables faster download speeds.
            hf_access_token = arg
        else:
            Warning(f"Option {opt} invalid")

    # Load the dataset
    ds = load_dataset("finedet/nuimages", token=hf_access_token)

    # Get example processor for data transformation
    checkpoint = "google/owlv2-base-patch16"
    processor = AutoProcessor.from_pretrained(checkpoint)  # By default, the processor expects PIL images

    def transform(batch):
        # Transform a batch from the nuImages dataset to a format the checkpoint can read
        texts = [
            [
                f"a photo of a {category}" for category in objs["category"]
            ] for objs in batch["objects"]
        ]

        images = [img.convert("RGB") for img in batch["image"]]

        return processor(text=texts, images=images, return_tensors="pt")

    ds_transform = ds["validation"].with_transform(transform)

    # Print the keys in a slice of the transformed dataset
    print(f"Keys in a slice: {list(ds_transform[0].keys())}")

    ids = ds_transform.to_iterable_dataset(num_shards=64)
    dataloader = DataLoader(ids, num_workers=4)
