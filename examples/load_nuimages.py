# Code modified from https://huggingface.co/datasets/finedet/nuimages

from datasets import load_dataset
from transformers import AutoProcessor
import sys
import getopt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


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

    # Get label names
    names = ds["train"].features["objects"]["category"].feature.names
    label_names = [name.split('.')[-1] for name in names]
    texts = [f"a photo of a {category}" for category in label_names]


    def transform(batch):
        """
        Transform a batch from the nuImages dataset to a format the checkpoint and the loss function can read.

        Args:
            batch (`dict`): A dictionary containing keys:
                - "image" (`list`):
                    List of PIL images.
                - "width" (`list`):
                    List of image widths.
                - "height" (`list`):
                    List of image heights.
                - "objects" (`list`): L
                    ist of object annotations containing bounding-boxes and query_ids.
        Returns:
            `BatchFeature`: A dictionary containing keys:
                - "input_ids" (`torch.Tensor`):
                    Input query sequence. Padded for equal length across batch.
                - "attention_mask" (`torch.Tensor`):
                    Mask, representing which tokens in input_ids are not padding.
                - "pixel_values" (`torch.Tensor`):
                    Pixel values of the processed image.
                - "bbox" (`torch.Tensor`):
                    Ground-truth bounding-boxes.
                - "category" (`torch.Tensor`):
                    Ground-truth labels.
                - "num_objects" (`int`):
                    Number of annotated objects.
        """

        batch_size = len(batch["image"])

        bbox_list = []
        category_list = []
        num_objects_list = []

        # Process ground-truth
        for j, objs in enumerate(batch["objects"]):
            # Get bounding-boxes and categories as tensors
            category = torch.tensor(objs["category"], dtype=torch.long)
            bbox = torch.tensor(objs["bbox"], dtype=torch.float32)

            x_dir_idc = torch.tensor((0, 2))
            y_dir_idc = torch.tensor((1, 3))

            # Normalize bounding-boxes with width and height of image
            bbox[:, x_dir_idc] = bbox[:, x_dir_idc] / batch["width"][j]
            bbox[:, y_dir_idc] = bbox[:, y_dir_idc] / batch["height"][j]

            category_list.append(category)
            bbox_list.append(bbox)

            num_objects_list.append(len(objs["category"]))

        # Process input and add ground-truth
        processed_batch = processor(text=texts, images=batch["image"], return_tensors="pt")
        processed_batch.update(bbox=bbox_list, category=category_list, num_objects=num_objects_list)

        # Expand dimension of input_ids and attention_masks for DataLoader to be able to load them
        processed_batch["input_ids"] = torch.broadcast_to(processed_batch["input_ids"], size=(batch_size, -1, -1))
        processed_batch["attention_mask"] = torch.broadcast_to(processed_batch["attention_mask"], size=(batch_size, -1, -1))
        return processed_batch


    def collate_fn(batch):
        max_num_objects = max([item["num_objects"] for item in batch])

        pixel_values = torch.stack([item["pixel_values"] for item in batch])

        # Pad category and bounding-box to the maximum number of objects in the batch
        category = torch.stack(
            [
                nn.functional.pad(
                    item["category"],
                    pad=(0, max_num_objects - item["num_objects"]),
                    value=-1
                )
                for item in batch
            ]
        )

        bbox = torch.stack(
            [
                nn.functional.pad(
                    item["bbox"],
                    pad=(0, 0, 0, max_num_objects - item["num_objects"]),
                    value=-1.0
                )
                for item in batch
            ]
        )

        return {
            "input_ids": batch[0]["input_ids"],
            "attention_mask": batch[0]["attention_mask"],
            "pixel_values": pixel_values,
            "gt_query_idx": category,
            "gt_bbox": bbox
        }


    val_ds = ds["validation"].with_transform(transform)

    # Print the keys in a slice of the dataset
    print(f"Keys in a slice: {list(val_ds[0:5].keys())}")

    dataloader = DataLoader(val_ds, batch_size=4, collate_fn=collate_fn)

    for i, item in enumerate(dataloader):
        # Print keys in a batch
        print(f"Keys in a batch: {list(item.keys())}")
        break
