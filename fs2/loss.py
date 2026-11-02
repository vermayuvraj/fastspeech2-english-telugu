"""Reference FS2 objectives, excluding padding and zero-duration variance units."""
import torch
import torch.nn.functional as F


def loss_terms(batch, prediction):
    mel, postnet, pitch, energy, log_duration, _, src_mask, mel_mask, _, _ = prediction
    valid_mel = ~mel_mask
    valid_src = ~src_mask
    valid_variance = valid_src & (batch["durations"] > 0)
    target = batch["mel"][:, :mel.shape[1]]
    terms = {
        "mel": F.l1_loss(mel[valid_mel], target[valid_mel]),
        "postnet": F.l1_loss(postnet[valid_mel], target[valid_mel]),
        "duration": F.mse_loss(log_duration[valid_src], torch.log1p(batch["durations"].float())[valid_src]),
        "pitch": F.mse_loss(pitch[valid_variance], batch["pitch"][valid_variance]),
        "energy": F.mse_loss(energy[valid_variance], batch["energy"][valid_variance]),
    }
    terms["total"] = sum(terms.values())
    return terms


def denominators(batch):
    return {"mel": int(batch["mel_lens"].sum()) * batch["mel"].shape[-1], "postnet": int(batch["mel_lens"].sum()) * batch["mel"].shape[-1], "duration": int(batch["src_lens"].sum()), "pitch": int((batch["durations"] > 0).sum()), "energy": int((batch["durations"] > 0).sum())}
