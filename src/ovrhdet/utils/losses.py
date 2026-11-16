import torch.nn as nn
import torch


# --- Not yet implemented ---
class SetPredictionLoss(nn.Module):
    def __init__(self, confidence_threshold):
        super().__init__()
        # raise NotImplementedError("Not yet implemented!")

    def forward(self, prediction, target):
        """
        Args:
            prediction (`dict`): A dictionary containing bounding-box and class predictions.
            Must contain keys:
                - "pred_logits" (`torch.FloatTensor` of shape `(batch_size, N, num_queries)`):
                    Logits for the likelihood of classes. Here, N is the number of predictions and num_queries is the
                    number of queries representing the classes to predict.
                - "pred_bbox" (`torch.FloatTensor` of shape `(batch_size, N, 4)`):
                    Centered bounding boxes [cx, cy, w, h] for all N predictions, with cx/cy being the center within
                    [0, 1] coordinates and w/h being the width and height of a bounding box.

            target (`dict`): A dictionary containing ground-truth bounding-boxes and classes.
            Must contain keys:
                - "gt_query_idx" (`torch.LongTensor` of shape `(batch_size, K)`):
                    Index of the ground-truth query at a corresponding bounding box. Here, K is the number of
                    ground-truth bounding boxes and must be at most N.
                - "gt_bbox" (`torch.FloatTensor` of shape `(batch_size, K, 4)`):
                    Centered bounding boxes [cx, cy, w, h], with cx/cy being the center coordinates within [0, 1] and
                    w/h being the width and height of a bounding box.

        Returns:
             `torch.FloatTensor` of shape `torch.Size([])`: Set Prediction Loss
        """
        return torch.tensor(0.0, requires_grad=True)
