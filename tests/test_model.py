from copy import deepcopy

import torch

from fs2.config import ROOT, load_config, seed_everything
from fs2.data import collate, forward_batch
from fs2.loss import loss_terms
from fs2.model import create_model
from vendor.ming024.model.modules import LengthRegulator


def tiny_config():
    config = load_config(ROOT / "English/config.yaml")
    model = config["model"]
    for key in ["encoder_layer", "decoder_layer"]:
        model["transformer"][key] = 1
    for key in ["encoder_hidden", "decoder_hidden"]:
        model["transformer"][key] = 32
    model["transformer"]["conv_filter_size"] = 64
    model["variance_predictor"]["filter_size"] = 32
    model["postnet_hidden"] = 32
    model["max_seq_len"] = 8  # Force the long-sequence path during training.
    return config


def test_reference_model_backward_and_masked_losses():
    seed_everything(5)
    torch.set_num_threads(2)
    model = create_model(tiny_config(), {"pitch": [-2, 2, 5, 0.4], "energy": [-2, 2, 1, 0.2]}, {"voice": 0})
    examples = []
    for count in [5, 4]:
        examples.append({"tokens": torch.tensor([1, 3, 4, 5, 1][:count]), "durations": torch.tensor([2] * count), "mel": torch.randn(count * 2, 80), "pitch": torch.randn(count), "energy": torch.randn(count), "speaker": 0, "id": str(count)})
    batch = collate(examples)
    prediction = forward_batch(model, batch)
    assert prediction[0].shape == (2, 10, 80)
    terms = loss_terms(batch, prediction)
    terms["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    changed = deepcopy(batch)
    changed["mel"][1, 8:] = 10000
    changed["pitch"][1, 4:] = 10000
    changed["energy"][1, 4:] = 10000
    assert torch.equal(terms["total"], loss_terms(changed, prediction)["total"])


def test_length_regulator_uses_input_device_and_exact_counts():
    regulator = LengthRegulator()
    x = torch.arange(12).reshape(1, 3, 4).float()
    expanded, lengths = regulator(x, torch.tensor([[1, 0, 2]]), None)
    assert expanded.shape == (1, 3, 4)
    assert lengths.device == x.device
    assert lengths.tolist() == [3]
    assert torch.equal(expanded[0], x[0, [0, 2, 2]])
