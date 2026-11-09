# Inference code copied from https://huggingface.co/google/owlv2-base-patch16

import requests
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import sys
import getopt
import cv2

if __name__ == '__main__':
    pretrained_model_name = "google/owlv2-base-patch16"
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    labels = ["cat", "dog"]

    image_path = None
    hf_access_token = False

    shortopts = "m:u:i:l:t:"
    longopts = ["model=", "image_url", "image_path=", "labels=", "hf_token="]

    try:
        opts, args = getopt.getopt(sys.argv[1:], shortopts, longopts)
    except getopt.GetoptError as err:
        print(f"Error: {err}")
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-m", "--model"):
            # Passes which of these models to use:
            # - google/owlv2-base-patch16
            # - google/owlv2-base-patch16-ensemble
            # - google/owlv2-base-patch16-finetuned
            # - google/owlv2-large-patch14
            # - google/owlv2-large-patch14-ensemble
            # - google/owlv2-large-patch14-finetuned
            # If model parameters need to be downloaded for the first time, run this script as admin

            assert arg.startswith("google/owlv2"), "The model needs to be a variant of OWLv2"
            pretrained_model_name = arg

        elif opt in ("-u", "--image_url"):
            # Passes the url of an image. Is ignored, if -i/--image_path is specified
            url = arg

        elif opt in ("-i", "--image_path"):
            # Passes the path to an image
            image_path = arg

        elif opt in ("-l", "--labels"):
            # Passes the labels to detect. Has to be n-grams (without spaces) separated by spaces.
            labels = arg.split()

        elif opt in ("-t", "--hf_token"):
            # Passes the HuggingFace access token. This enables faster download speeds of model parameters.
            hf_access_token = arg
        else:
            Warning(f"Option {opt} invalid")

    print(f"Model: {pretrained_model_name}")

    if image_path is not None:
        print(f"Image: {image_path}")
        image = Image.open(image_path)
    else:
        print(f"Image: {url}")
        image = Image.open(requests.get(url, stream=True).raw)

    print(f"Labels: {labels}")

    processor = AutoProcessor.from_pretrained(pretrained_model_name, token=hf_access_token)
    model = Owlv2ForObjectDetection.from_pretrained(pretrained_model_name, token=hf_access_token)

    # The text encoder of OWLv2 requires the prefix "a photo of a {label}" for a given label
    texts = [[f"a photo of a {label}" for label in labels]]
    inputs = processor(text=texts, images=image, return_tensors="pt")

    # Perform forward pass
    with torch.no_grad():
        outputs = model(**inputs)


    # Note: boxes need to be visualized on the padded, unnormalized image
    # hence we'll set the target image sizes (height, width) based on that

    def get_preprocessed_image(pixel_values):
        pixel_values = pixel_values.squeeze().numpy()
        unnormalized_image = (
                (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None])
                + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        )
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        unnormalized_image = Image.fromarray(unnormalized_image)
        return unnormalized_image


    unnormalized_image = get_preprocessed_image(inputs.pixel_values)

    target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, threshold=0.2, target_sizes=target_sizes
    )

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    image_np = np.array(image)

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

        # Draw bounding boxes and labels on the image
        cv2.rectangle(image_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        cv2.putText(
            image_np,
            text[label],
            (int(box[0]), int(box[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    cv2.imshow("Bounding Boxes", image_np)
    cv2.waitKey(0)
