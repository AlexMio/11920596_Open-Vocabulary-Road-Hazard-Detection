import pytest
import torch

from ..utils.losses import SetPredictionLoss

devices = [torch.device("cpu")]
devices.extend([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])


def to_device(data: tuple, device):
    out = {k: v.to(device).detach() for k, v in data[0].items()}, {k: v.to(device).detach() for k, v in data[1].items()}
    return out


# --- ARRANGE ---

@pytest.fixture
def confidence():
    return 0.1


@pytest.fixture
def loss_fcn(confidence):
    return SetPredictionLoss(confidence_threshold=confidence)


def generate_random_valid(params):
    """
    Generates random valid samples for SetPredictionLoss.
    :param params: Parameter tuple of (batch_size, num_predictions, num_classes, num_targets)
    :return: (prediction, target)
    """
    batch_size, num_predictions, num_classes, num_targets = params

    prediction = {
        "pred_logits": torch.rand((batch_size, num_predictions, num_classes), requires_grad=True),
        "pred_bbox": torch.rand((batch_size, num_predictions, 4), requires_grad=True)
    }
    target = {
        "gt_query_idx": torch.randint(0, num_classes, (batch_size, num_targets,)),
        "gt_bbox": torch.rand((batch_size, num_targets, 4))
    }

    return prediction, target


def generate_zero_loss(params):
    """
    Generates valid samples that must return a loss of 0 in SetPredictionLoss. SetPredictionLoss should ignore that more
    predictions are available than targets.
    :param params: Parameter tuple of (batch_size, num_predictions, num_classes, num_targets)
    :return: (prediction, target)
    """
    batch_size, num_predictions, num_classes, num_targets = params

    prediction = {
        "pred_logits": torch.rand((batch_size, num_predictions, num_classes), requires_grad=True),
        "pred_bbox": torch.rand((batch_size, num_predictions, 4), requires_grad=True)
    }

    pred_labels = torch.max(prediction["pred_logits"], dim=-1).indices

    keep_idc = torch.randperm(num_predictions)[0:num_targets]

    gt_query_idx = pred_labels[:, keep_idc]
    gt_query_idx = gt_query_idx.detach().requires_grad_(False)
    gt_bbox = prediction["pred_bbox"][:, keep_idc, :]
    gt_bbox = gt_bbox.detach().requires_grad_(False)

    target = {"gt_query_idx": gt_query_idx, "gt_bbox": gt_bbox}
    return prediction, target


def generate_non_zero_loss(params):
    """
    Generates valid samples that must return a loss strictly greater than 0 in SetPredictionLoss.
    :param params: Parameter tuple of (batch_size, num_predictions, num_classes, num_targets)
    :return: (prediction, target)
    """
    batch_size, num_predictions, num_classes, num_targets = params

    prediction = {
        "pred_logits": torch.rand((batch_size, num_predictions, num_classes), requires_grad=True),
        "pred_bbox": torch.rand((batch_size, num_predictions, 4), requires_grad=True)
    }

    gt_labels = torch.max(prediction["pred_logits"], dim=-1).indices

    signs = torch.tensor([-1, 1])

    # Force ground-truth labels and bounding-boxes to be different from predicted ones

    # Get direction to shift the label index. If an index is below half, shift upwards and vice versa.
    label_sign_idx = (gt_labels < (num_classes / 2)).to(dtype=torch.int32)
    label_perturbation_signs = signs[label_sign_idx]

    # Get random label perturbations
    label_perturbation = torch.randint(1, 3, (batch_size, num_predictions)) * label_perturbation_signs

    # Shift labels randomly and clamp to stay within the number of classes
    gt_labels = torch.clamp(gt_labels + label_perturbation, 0, num_classes - 1)

    # Get direction to shift bounding boxes
    bbox_sign_idx = (prediction["pred_bbox"] < 0.5).to(dtype=torch.int32)
    bbox_perturbation_signs = signs[bbox_sign_idx]

    # Get random bbox perturbations
    bbox_perturbation = torch.rand((batch_size, num_predictions, 4)) * bbox_perturbation_signs

    # Shift bounding-boxes randomly
    gt_bbox = torch.clamp(prediction["pred_bbox"] + bbox_perturbation, 0, 1)

    keep_idc = torch.randperm(num_predictions)[0:num_targets]

    gt_query_idx = gt_labels[:, keep_idc].detach().requires_grad_(False)
    gt_bbox = gt_bbox[:, keep_idc, :].detach().requires_grad_(False)

    target = {"gt_query_idx": gt_query_idx, "gt_bbox": gt_bbox}
    return prediction, target


def generate_zeros(params):
    """
    Generates samples containing only zeros.
    :param params: Parameter tuple of (batch_size, num_predictions, num_classes, num_targets)
    :return: (prediction, target)
    """
    batch_size, num_predictions, num_classes, num_targets = params

    prediction = {
        "pred_logits": torch.zeros((batch_size, num_predictions, num_classes), requires_grad=True),
        "pred_bbox": torch.zeros((batch_size, num_predictions, 4), requires_grad=True)
    }
    target = {
        "gt_query_idx": torch.zeros((batch_size, num_targets,)),
        "gt_bbox": torch.zeros((batch_size, num_targets, 4))
    }

    return prediction, target


def generate_high_values(params):
    """
    Generates samples containing high values. Ground-truth query indices are kept at num_classes.
    :param params: Parameter tuple of (batch_size, num_predictions, num_classes, num_targets)
    :return: (prediction, target)
    """
    batch_size, num_predictions, num_classes, num_targets = params

    high_value = 1e6

    prediction = {
        "pred_logits": torch.ones((batch_size, num_predictions, num_classes), requires_grad=True) * high_value,
        "pred_bbox": torch.ones((batch_size, num_predictions, 4), requires_grad=True) * high_value
    }
    target = {
        "gt_query_idx": torch.ones((batch_size, num_targets,)) * num_classes,
        "gt_bbox": torch.ones((batch_size, num_targets, 4)) * high_value
    }

    return prediction, target


# Dictionary for accessing sample generators with fixtures
sample_generators = {
    "random_valid_samples": generate_random_valid,
    "zero_loss_samples": generate_zero_loss,
    "non_zero_loss_samples": generate_non_zero_loss,
    "zeros_samples": generate_zeros,
    "high_values_samples": generate_high_values,
}

# --- ACT & ASSERT ---

# Test scenarios randomly chosen with realistic edge cases
generic_scenarios = [
    ("random_valid_samples", (1, 2, 2, 1)),
    ("random_valid_samples", (5, 20, 10, 15)),
    ("zero_loss_samples", (3, 10, 5, 10)),
    ("zeros_samples", (3, 10, 5, 10)),
    ("high_values_samples", (3, 10, 5, 10))
]


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("generator_name, params", generic_scenarios)
def test_loss_shape_and_type(loss_fcn, device, generator_name, params):
    """
    Tests the loss function, if it returns the correct datatype on all samples.
    :param loss_fcn: Loss function to test.
    :param device: The device to run the test on.
    :param generator_name: Which type of samples to generate.
    :param params: Parameters for generating samples. Format: (batch_size, num_predictions, num_classes, num_targets)
    """
    prediction, target = to_device(sample_generators[generator_name](params), device)

    # Act
    loss = loss_fcn(prediction, target)

    # Assert
    assert loss.shape == torch.Size([])
    assert loss.dtype in [torch.float16, torch.float32, torch.float64]


# Test scenarios randomly chosen with realistic edge cases
test_zero_loss_scenarios = [
    ("zero_loss_samples", (3, 10, 5, 10)),
    ("zero_loss_samples", (1, 20, 1, 10)),
    ("zero_loss_samples", (5, 1, 20, 1)),
]


@pytest.mark.parametrize("device", tuple(devices))
@pytest.mark.parametrize("generator_name, params", test_zero_loss_scenarios)
def test_zero_loss(loss_fcn, device, generator_name, params):
    """
    Tests the loss function, if it returns 0 when it should.
    :param loss_fcn: Loss function to test.
    :param device: Device to run the test on.
    :param generator_name: Type of samples to generate.
    :param params: Parameters for generating samples. Format: (batch_size, num_predictions, num_classes, num_targets)
    """
    prediction, target = to_device(sample_generators[generator_name](params), device)

    # Act
    loss = loss_fcn(prediction, target)

    # Assert
    assert loss.shape == torch.Size([])
    assert torch.isclose(loss, torch.tensor(0.0))


# Test scenarios randomly chosen with realistic edge cases
test_non_zero_loss_scenarios = [
    ("non_zero_loss_samples", (3, 10, 5, 10)),
    ("non_zero_loss_samples", (1, 20, 1, 10)),
    ("non_zero_loss_samples", (10, 1, 20, 1)),
]


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("generator_name, params", test_non_zero_loss_scenarios)
def test_non_zero_loss(loss_fcn, device, generator_name, params):
    """
    Tests the loss function, if it returns non-0 when it should.
    :param loss_fcn: Loss function to test.
    :param device: Device to run the test on.
    :param generator_name: Type of samples to generate.
    :param params: Parameters for generating samples. Format: (batch_size, num_predictions, num_classes, num_targets)
    """
    prediction, target = sample_generators[generator_name](params)

    # Act
    loss = loss_fcn(prediction, target)

    # Assert
    assert loss.shape == torch.Size([])
    assert loss > 0


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("generator_name, params", generic_scenarios)
def test_positive_loss(loss_fcn, device, generator_name, params):
    """
    Tests the loss function, if it returns >=0 on all samples.
    :param loss_fcn: Loss function to test.
    :param device: Device to run the test on.
    :param generator_name: Type of samples to generate.
    :param params: Parameters for generating samples. Format: (batch_size, num_predictions, num_classes, num_targets)
    """
    prediction, target = to_device(sample_generators[generator_name](params), device)

    # Act
    loss = loss_fcn(prediction, target)

    # Assert
    assert loss.shape == torch.Size([])
    assert loss >= 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("generator_name, params", generic_scenarios)
def test_loss_gradient_flow(loss_fcn, device, generator_name, params):
    """
    Tests the loss function, if it generates a gradient on its input when loss.backward() is called.
    :param loss_fcn: Loss function to test.
    :param device: Device to run the test on.
    :param generator_name: Type of samples to generate.
    :param params: Parameters for generating samples. Format: (batch_size, num_predictions, num_classes, num_targets)
    """
    prediction, target = to_device(sample_generators[generator_name](params), device)

    # Act
    loss = loss_fcn(prediction, target)
    loss.backward()

    # Assert
    assert loss.requires_grad
    for v in prediction.values():
        assert v.grad is not None
