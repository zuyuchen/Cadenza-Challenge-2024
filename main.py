import logging
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from omegaconf import OmegaConf
import pyloudnorm as pyln
from clarity.enhancer.nalr import NALR
from recipes.cad_icassp_2024.baseline.evaluate import (remix_stems)
from recipes.cad_icassp_2024.baseline.enhance import process_remix_for_listener
import sys
sys.path.append("/Users/chenzuyu/haaqinet/src")
import HAAQI_Net
from HAAQI_Net import HAAQINet

from BEATs import BEATs, BEATsConfig
from clarity.utils.audiogram import Listener
from clarity.utils.file_io import read_signal
from clarity.utils.signal_processing import resample, clip_signal, to_16bit, compute_rms, normalize_signal
from clarity.evaluator.haaqi import compute_haaqi
from clarity.utils.source_separation_support import get_device
from clarity.utils.flac_encoder import FlacEncoder
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
import random

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ----------------- Write to Save --------------------
# ----------------- Save Audio --------------------
def save_flac_signal(signal: np.ndarray, filename: Path, signal_sample_rate, output_sample_rate,
                     do_clip_signal=False, do_soft_clip=False, do_scale_signal=False):
    if signal_sample_rate != output_sample_rate:
        signal = resample(signal, signal_sample_rate, output_sample_rate)
    if do_scale_signal:
        signal = signal / np.max(np.abs(signal))
    elif do_clip_signal:
        signal, n_clipped = clip_signal(signal, do_soft_clip)
        if n_clipped > 0:
            logger.warning(f"Clipped {n_clipped} samples in {filename}")
    signal = to_16bit(signal)
    FlacEncoder().encode(signal, output_sample_rate, filename)

# ----------------- Models --------------------
# ----------------- Modified Source Separator with Frozen Layers ----------------------
class SourceSeparator(nn.Module):
    def __init__(self):
        super().__init__()
        bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB
        self.model = bundle.get_model()
        self.sample_rate = bundle.sample_rate
        self.sources = self.model.sources

        # Freeze encoder and masknet
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in ["encoder", "masknet"]):
                param.requires_grad = False

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.model(signal)

    def decompose_signal(
        self,
        signal: torch.Tensor,  # Now accepts a tensor directly
        signal_sample_rate: int,
        device: torch.device,

        normalise: bool = True
    ) -> dict[str, torch.Tensor]:
        """Decompose the signal into sources using PyTorch tensors.

        Args:
            signal (torch.Tensor): Input signal tensor [channels, samples].
            signal_sample_rate (int): Sample rate of the signal.
            device (torch.device): Device to perform computation on.
            listener (Listener): Listener object for audiogram.
            normalise (bool): Whether to normalize the signal.

        Returns:
            dict: Dictionary of source tensors [channels, samples].
        """
        # Ensure signal is [channels, samples]
        if signal.shape[0] > signal.shape[1]:
            signal = signal.transpose(0, 1)

        # Resample if necessary
        if signal_sample_rate != self.sample_rate:
            signal = torchaudio.transforms.Resample(signal_sample_rate, self.sample_rate).to(device)(signal)

        # Normalize the signal
        if normalise:
            # # Compute RMS and normalize using PyTorch operations
            # signal_rms = torch.sqrt(torch.mean(signal ** 2))
            # if signal_rms > 0:
            #     signal = signal / signal_rms
            ref = signal.mean(0)
            signal = (signal - ref.mean()) / ref.std()

        # Add batch dimension: [1, channels, samples]
        signal_tensor = signal.unsqueeze(0)
        # Forward pass through the model
        separated = self(signal_tensor)[0]  # [num_sources, channels, samples]

        # Create dictionary of sources
        sources = [s for s in separated]
        return dict(zip(self.sources, sources))

class GainPredictor(nn.Module):
    def __init__(self, audiogram_dim=8, num_stems=4, db_range=(-20.0, 20.0)):
        super().__init__()
        self.audiogram_dim = audiogram_dim
        self.num_stems = num_stems
        self.db_range = db_range

        # Define convolutional layers with residual connections
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # Compute the flattened feature dimension: 64 channels * audiogram_dim
        self.flattened_dim = 64 * audiogram_dim  # 64 * 8 = 512
        self.db_output = nn.Linear(self.flattened_dim, num_stems)  # [512, 4]

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.conv:
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu', a=0.01)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.db_output.weight)
        nn.init.zeros_(self.db_output.bias)

    def forward(self, audiogram):
        x = self.conv[0](audiogram)  # [batch_size, 16, 8]
        x_res = x  # Residual for first block
        x = self.conv[1](x)  # LeakyReLU
        x = self.conv[2](x)  # [batch_size, 32, 8]
        x_res = nn.Conv1d(16, 32, kernel_size=1)(x_res)  # 1x1 conv to match channels
        x = x + x_res  # Add residual
        x_res = x  # Residual for second block
        x = self.conv[3](x)  # LeakyReLU
        x = self.conv[4](x)  # [batch_size, 64, 8]
        x_res = nn.Conv1d(32, 64, kernel_size=1)(x_res)  # 1x1 conv to match channels
        x = x + x_res  # Add residual
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 64 * 8]
        gains = self.db_output(x)  # [batch_size, num_stems]
        min_db, max_db = self.db_range
        gains = torch.tanh(gains)  # Constrain to (-1, 1)
        gains = min_db + (max_db - min_db) * (gains + 1) / 2  # Map to (min_db, max_db)

        return gains  # [batch_size, num_stems], gains in dB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)

def loudness_mismatch_loss(enh, ref):
    # enh, ref: [batch, samples, channels]
    if torch.isnan(enh).any() or torch.isinf(enh).any():
        logger.error(f"NaN or Inf in enhanced tensor: {enh}")
        raise ValueError("Enhanced tensor contains NaN or Inf")
    if torch.isnan(ref).any() or torch.isinf(ref).any():
        logger.error(f"NaN or Inf in reference tensor: {ref}")
        raise ValueError("Reference tensor contains NaN or Inf")

    eps = 1e-8
    # Compute mean squared over samples (dim=1)
    mean_squared_enh = (enh.pow(2).mean(dim=1) + eps)  # [batch, channels]
    mean_squared_ref = (ref.pow(2).mean(dim=1) + eps)
    # logger.info(f"Mean squared enh: {mean_squared_enh}, Mean squared ref: {mean_squared_ref}")

    # Compute RMS
    rms_enh = mean_squared_enh.sqrt()  # [batch, channels]
    rms_ref = mean_squared_ref.sqrt()
    # logger.info(f"RMS enh: {rms_enh}, RMS ref: {rms_ref}")

    # Compute loss as mean squared difference over batch and channels
    diff = rms_enh - rms_ref
    loss = (diff ** 2).mean()
    # logger.info(f"Loudness loss: {loss.item()}")

    if torch.isnan(loss) or torch.isinf(loss):
        logger.error(f"NaN or Inf in loudness loss: diff={diff}")
        raise ValueError("Loudness loss computation resulted in NaN or Inf")

    return loss


# perceptual loss function
import sys
sys.path.append("/Users/chenzuyu/haaqinet/src")
import argparse
from pathlib import Path
from typing import List, Tuple


def compute_loss_and_backprop(
    enhanced_stems_batch: dict[str, torch.Tensor],
    reference_stems_batch: dict[str, torch.Tensor],
    hearing_levels_batch: torch.Tensor,
    haaqi_net_model,
    optimizer,
    gain_model,
    separator,
    alpha: float = 1.0
):
    """
    Compute loss and backpropagate, including perceptual loss from HAAQI-Net, at the stem level.

    Args:
        enhanced_stems_batch (dict): Dictionary of enhanced stem tensors [batch, channels, samples].
        reference_stems_batch (dict): Dictionary of reference stem tensors [batch, channels, samples].
        hearing_levels_batch (Tensor): [batch, 8] hearing level input.
        haaqi_net_model (nn.Module): Pretrained HAAQI-Net model.
        optimizer (Optimizer): Optimizer for training.
        gain_model (nn.Module): Gain prediction model.
        separator (nn.Module): Source separation model.
        alpha (float): Weight for perceptual loss.

    Returns:
        dict: Dictionary with individual loss components and total loss.
    """
    mse_loss = 0.0
    sdr_loss = 0.0
    loudness_loss = 0.0
    perceptual_loss = 0.0
    haaqi_score = 0.0
    num_stems = len(enhanced_stems_batch)

    for src in enhanced_stems_batch.keys():
        enhanced_tensor = torch.stack(enhanced_stems_batch[src])  # [batch, channels, samples]
        reference_tensor = torch.stack(reference_stems_batch[src])  # [batch, channels, samples]

        # MSE loss
        mse_loss += torch.mean((enhanced_tensor - reference_tensor) ** 2)

        # SDR loss
        sdr_loss += -sdr_metric(
            enhanced_tensor.permute(0, 2, 1),
            reference_tensor.permute(0, 2, 1)
        ).mean()

        # Loudness loss
        loudness_loss = loudness_mismatch_loss(enhanced_tensor, reference_tensor)

        # Perceptual loss with HAAQI-Net
        enhanced_mono = enhanced_tensor.mean(dim=2, keepdim=True).permute(0, 1, 2)  # [batch, samples, 1]
        resampler = torchaudio.transforms.Resample(44100, 16000).to(device)
        enhanced_mono_batch = torch.stack([
            resampler(waveform.squeeze(-1)).unsqueeze(-1)
            for waveform in enhanced_mono
        ])  # [batch, samples, 1]
        frame_scores, haaqi_score_src, *_ = haaqi_net_model(enhanced_mono_batch.squeeze(-1), hearing_levels_batch)
        perceptual_loss += -haaqi_score_src.mean()
        haaqi_score += haaqi_score_src.mean()

    # Average losses across stems
    mse_loss /= num_stems
    sdr_loss /= num_stems
    loudness_loss /= num_stems
    perceptual_loss /= num_stems
    haaqi_score /= num_stems

    total_loss = mse_loss + 0.3 * sdr_loss + 2.0 * loudness_loss + alpha * perceptual_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    logger.info(
        f"Total Loss={total_loss:.4f}, MSE={mse_loss:.4f}, SDR={sdr_loss:.4f}, "
        f"Loudness={loudness_loss:.4f}, Perceptual={perceptual_loss:.4f}, HAAQI={haaqi_score}"
    )
    return {
        "total_loss": total_loss.item(),
        "mse_loss": mse_loss.item(),
        "sdr_loss": -sdr_loss.item(),
        "loudness_loss": loudness_loss.item(),
        "perceptual_loss": perceptual_loss.item(),
        "haaqi_score": haaqi_score.item()
    }


# ----------------- Results File --------------------
class ResultsFile:
    def __init__(self, filename, header_columns):
        self.filename = Path(filename)
        self.header_columns = header_columns
        if not self.filename.exists():
            with open(self.filename, 'w', encoding='utf-8') as f:
                f.write(','.join(self.header_columns) + '\n')

    def add_result(self, result):
        with open(self.filename, 'a', encoding='utf-8') as f:
            values = [str(result[col]) for col in self.header_columns]
            f.write(','.join(values) + '\n')

# ----------------- Utility Functions --------------------
def compute_lufs(signal: np.ndarray, sample_rate: float) -> float:
    """Compute the integrated LUFS of a signal using pyloudnorm."""
    meter = pyln.Meter(int(sample_rate))
    lufs = meter.integrated_loudness(signal)
    if lufs == -np.inf:
        lufs = -80  # Fallback for silent signals
    return lufs
def compute_batch_haaqi_scores(
    enhanced: np.ndarray,
    reference: np.ndarray,
    sample_rate: float,
    haaqi_sample_rate: float,
    audiograms_left: list,
    audiograms_right: list,
) -> tuple[float, float, float]:

    """Compute average HAAQI scores for a batch."""
    left_scores = []
    right_scores = []
    for i in range(enhanced.shape[0]):
        left_score = compute_haaqi(
            processed_signal=resample(enhanced[i,:,0], sample_rate, haaqi_sample_rate),
            reference_signal=resample(reference[i,:,0], sample_rate, haaqi_sample_rate),
            processed_sample_rate=haaqi_sample_rate,
            reference_sample_rate=haaqi_sample_rate,
            audiogram=audiograms_left[i],
            equalisation=2,
            # level1=65 - 20 * np.log10(compute_rms(reference_mixture[i, 0])),
        )
        right_score = compute_haaqi(
            processed_signal=resample(enhanced[i, :, 1], sample_rate, haaqi_sample_rate),
            reference_signal=resample(reference[i, :, 1], sample_rate, haaqi_sample_rate),
            processed_sample_rate=haaqi_sample_rate,
            reference_sample_rate=haaqi_sample_rate,
            audiogram=audiograms_right[i],
            equalisation=2,
            # level1=65 - 20 * np.log10(compute_rms(reference_mixture[i, 1])),
        )
        left_scores.append(left_score)
        right_scores.append(right_score)
    avg_left_score = np.mean(left_scores)
    avg_right_score = np.mean(right_scores)
    avg_score = np.mean([avg_left_score, avg_right_score])
    return avg_left_score, avg_right_score, avg_score
def make_scene_listener_list(scenes_listeners: dict, max_scenes: int) -> list:
    scene_listener_pairs = [
        (scene, listener)
        for scene in scenes_listeners
        for listener in scenes_listeners[scene]
    ]
    return scene_listener_pairs[:max_scenes]  # 15 scenes

# load HAAQI_Net model
def load_haaqi_net_model(config_path: str, model_path: str, device: torch.device):
    """
    Load pretrained HAAQI-Net model with BEATs frontend.

    Args:
        config_path (str): Path to config YAML file.
        model_path (str): Path to .pth model weights.
        device (torch.device): CUDA or CPU device.

    Returns:
        haaqi_net_model (torch.nn.Module): HAAQI-Net ready for inference.
    """
    # Load config
    config = OmegaConf.load(config_path)

    # Load BEATs frontend
    beats_ckpt = torch.load(config["beats_model_path"], map_location=device)
    beats_cfg = BEATsConfig(beats_ckpt["cfg"])
    beats_model = BEATs(beats_cfg)
    beats_model.load_state_dict(beats_ckpt["model"])
    beats_model.to(device)
    beats_model.eval()
    for p in beats_model.parameters():
        p.requires_grad = False

    # Load HAAQI-Net an Initialize model
    haaqi_net_model = HAAQINet(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        linear_output=config["model"]["linear_output"],
        act_fn=config["model"]["act_fn"],
        beats_model=beats_model
    )
    # Load model weights
    ckpt = torch.load(model_path, map_location=device)
    haaqi_net_model.load_state_dict(ckpt["model"])
    haaqi_net_model.to(device)
    haaqi_net_model.eval()

    return haaqi_net_model

# def apply_gains(stems: dict[str, torch.Tensor], sample_rate: float, gain_matrix: torch.Tensor) -> dict[str, torch.Tensor]:
#     """Apply gain to the signal by using LUFS, operating on PyTorch tensors with tensor gains.
#
#     Args:
#         stems (dict): Dictionary of stem tensors [channels, samples].
#         sample_rate (float): Sample rate of the signal.
#         gain_matrix (torch.Tensor): Tensor of gains in dB [num_stems].
#
#     Returns:
#         dict: Dictionary of stems with applied gains, as PyTorch tensors.
#     """
#     meter = pyln.Meter(int(sample_rate))
#     stems_gain = {}
#
#     for idx, (stem_str, stem_tensor) in enumerate(stems.items()):
#         # Ensure shape is [channels, samples] (transpose if needed)
#         if stem_tensor.shape[0] > stem_tensor.shape[1]:
#             stem_tensor = stem_tensor.transpose(0, 1)
#
#         # Compute integrated loudness (temporary detachment for pyln)
#         stem_np = stem_tensor.detach().cpu().numpy()
#         stem_lufs = meter.integrated_loudness(stem_np.T)
#         if stem_lufs == -np.inf:
#             stem_lufs = -80
#
#         # Use tensor gain from gain_matrix
#         gain = gain_matrix[idx]  # [1], keep as tensor
#         target_lufs = stem_lufs + gain  # Keep as tensor
#
#         # Compute gain factor in PyTorch (linear scale)
#         # LUFS to linear gain: 10^((target_lufs - stem_lufs) / 20)
#         gain_db = target_lufs - stem_lufs  # This is just the gain, since target_lufs includes gain
#         gain_linear = 10 ** (gain_db / 20.0)  # Linear scaling factor
#
#         # Apply gain in PyTorch to preserve computation graph
#         normalized_tensor = stem_tensor * gain_linear  # Gradient flows through gain_matrix and stem_tensor
#         stems_gain[stem_str] = normalized_tensor
#
#     return stems_gain

# ----------------- Validation Functions --------------------

def apply_gains(stems: dict[str, torch.Tensor], gain_matrix: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Fully differentiable gain application. Avoids LUFS-based scaling.
    Assumes gain_matrix: [num_stems], in dB.
    """
    stems_gain = {}
    for idx, (stem_str, stem_tensor) in enumerate(stems.items()):
        gain_dB = gain_matrix[idx]  # Tensor scalar
        gain_dB = torch.clamp(gain_dB, -40.0, 40.0)  # Prevent explosion
        gain_linear = 10 ** ((gain_dB + 1e-7) / 20.0)  # Add epsilon to prevent log(0)
        stems_gain[stem_str] = stem_tensor * gain_linear  # [channels, samples]
    return stems_gain

def validate(gain_model, separator, config, epoch):
    gain_model.eval()

    scenes = json.load(open(config.path.validation_scenes_file))
    songs = json.load(open(config.path.validation_music_file))
    scene_listeners = json.load(open(config.path.validation_scene_listeners_file))
    listeners = Listener.load_listener_dict(config.path.validation_listeners_file)

    validation_dir = Path(config.path.validation_music_dir)
    segment_duration = config.validate.segment_duration
    segment_samples = int(config.sample_rate * segment_duration)
    exp_folder = Path(config.path.exp_folder) / f"epoch_{epoch}" / "validation"
    exp_folder.mkdir(parents=True, exist_ok=True)

    scores_headers = ["left_score", "right_score", "score", "lufs_diff"]
    results_file = ResultsFile(f"validation_scores_epoch_{epoch}.csv", scores_headers)

    scene_listener_pairs = make_scene_listener_list(scene_listeners, config.validate.max_scenes)
    num_scenes = len(scene_listener_pairs)

    count = 0
    lufs_diff = 0
    enhanced_batch = []
    reference_batch = []
    listener_batch = []

    with torch.no_grad():
        for idx, (scene_id, listener_id) in enumerate(scene_listener_pairs, 1):
            scene = scenes[scene_id]
            song_key = f"{scene['music']}-{scene['head_loudspeaker_positions']}"
            song_path = validation_dir / songs[song_key]["Path"]

            logger.info(f"[{idx:03d}/{num_scenes:03d}] Validating {scene_id} for listener {listener_id}")

            try:
                mix = read_signal(song_path / "mixture.wav", config.sample_rate)
            except Exception as e:
                logger.warning(f"Failed to read {song_path / 'mixture.wav'}: {e}")
                continue

            expected_samples = segment_samples
            if mix.shape[0] != expected_samples:
                logger.warning(f"Clip {song_path / 'mixture.wav'} has {mix.shape[0]} samples, expected {expected_samples}")
                continue

            signal = torch.tensor(mix, dtype=torch.float32, device=device)
            listener = listeners[listener_id]
            audiogram_np = np.stack([listener.audiogram_left.levels, listener.audiogram_right.levels])
            audiogram_np = audiogram_np / 80
            audiogram = torch.tensor(audiogram_np, dtype=torch.float32).unsqueeze(0).to(device)
            # Pass the tensor directly to decompose_signal
            stems_dict = separator.decompose_signal(signal, config.sample_rate, device, listener)
            gain_matrix = gain_model(audiogram)

            enhanced_stems = apply_gains(stems_dict, gain_matrix.squeeze())
            enhanced = remix_stems({src: enhanced_stems[src].detach().cpu().numpy().T for src in separator.sources},
                                  signal.cpu().numpy(), config.sample_rate)

            enhancer = NALR(**config.nalr)
            enhanced = process_remix_for_listener(
                signal=enhanced,
                enhancer=enhancer,
                compressor=None,
                listener=listener,
                apply_compressor=config.apply_compressor,
            )

            reference_stems = {}
            for i, src in enumerate(separator.sources):
                try:
                    stem = read_signal(song_path / f"{src}.wav", config.sample_rate)
                except Exception as e:
                    logger.warning(f"Failed to read {song_path / f'{src}.wav'}: {e}")
                    continue
                if stem.shape[0] != expected_samples:
                    logger.warning(f"Clip {song_path / f'{src}.wav'} has {stem.shape[0]} samples, expected {expected_samples}")
                    continue
                if stem.shape[0] < stem.shape[1]:
                    stem = stem.T
                reference_stems[src] = torch.tensor(stem, dtype=torch.float32, device=device)

            reference_stems = apply_gains(reference_stems, gain_matrix.squeeze())
            reference = remix_stems({src: reference_stems[src].detach().cpu().numpy() for src in separator.sources},
                                   signal.cpu().numpy(), config.sample_rate)
            reference = process_remix_for_listener(
                signal=reference,
                enhancer=enhancer,
                compressor=None,
                listener=listener,
                apply_compressor=config.apply_compressor,
            )

            flac_filename = exp_folder / f"{scene_id}_{listener_id}_enhanced.flac"
            save_flac_signal(
                enhanced,
                flac_filename,
                config.sample_rate,
                config.sample_rate,
                do_clip_signal=True,
                do_soft_clip=config.soft_clip
            )

            enhanced_lufs = compute_lufs(enhanced, config.sample_rate)
            reference_lufs = compute_lufs(reference, config.sample_rate)
            lufs_diff += enhanced_lufs - reference_lufs

            count += 1
            enhanced_batch.append(enhanced)
            reference_batch.append(reference)
            listener_batch.append(listener)

        enhanced_batch = np.array(enhanced_batch)
        reference_batch = np.array(reference_batch)

        left_score, right_score, score = compute_batch_haaqi_scores(
            enhanced_batch,
            reference_batch,
            config.sample_rate,
            config.HAAQI_sample_rate,
            [listener.audiogram_left for listener in listener_batch],
            [listener.audiogram_right for listener in listener_batch],
        )

        results_file.add_result({
            "left_score": left_score,
            "right_score": right_score,
            "score": score,
            "lufs_diff": lufs_diff/count,
        })

    if count > 0:
        logger.info(
            f"Epoch {epoch} Validation: Avg Left HAAQI={left_score:.4f}, "
            f"Avg Right HAAQI={right_score:.4f}, Avg HAAQI={score:.4f}, "
            f"Avg LUFS Diff={lufs_diff/count:.2f}"
        )

# ----------------- Training Loop ----------------------
def train():
    config = OmegaConf.load("config.yaml")
    scenes = json.load(open(config.path.scenes_file))
    songs = json.load(open(config.path.music_file))
    scene_listeners = json.load(open(config.path.scene_listeners_file))
    listeners = Listener.load_listener_dict(config.path.listeners_file)

    gain_model = GainPredictor().to(device)
    separator = SourceSeparator().to(device)

    optimizer = torch.optim.Adam(
        list(gain_model.parameters()) + list(filter(lambda p: p.requires_grad, separator.parameters())),
        lr=1e-4
    )
    segment_duration = config.train.segment_duration
    segment_samples = int(config.sample_rate * segment_duration)
    batch_size = config.train.batch_size

    total_losses, mse_losses, sdr_losses, loudness_losses, perceptual_losses = [], [], [], [], []
    scores_headers = ["epoch", "batch", "total_loss", "mse_loss", "sdr_loss", "loudness_loss", "perceptual_loss", "haaqi_score"]
    training_logger = ResultsFile("training_scores.csv", scores_headers)

    for epoch in range(config.train.epochs):
        gain_model.train()
        separator.train()
        epoch_loss, epoch_mse, epoch_sdr, epoch_loudness, epoch_perceptual = 0.0, 0.0, 0.0, 0.0, 0.0
        segments_used = 0
        batch_count = 0

        exp_folder = Path(config.path.exp_folder) / f"epoch_{epoch+1}" / "training"
        exp_folder.mkdir(parents=True, exist_ok=True)

        enhanced_stems_batch = {src: [] for src in separator.sources}
        reference_stems_batch = {src: [] for src in separator.sources}
        hearing_levels_batch = []
        batch_segment_count = 0
        batch_seg_indices = []

        all_scene_listener_pairs = [(scene_id, listener_id) for scene_id in scene_listeners for listener_id in scene_listeners[scene_id]]
        random.shuffle(all_scene_listener_pairs)

        for scene_id, listener_id in all_scene_listener_pairs:
            if segments_used >= config.train.max_segments_per_epoch:
                break

            scene = scenes[scene_id]
            song_key = f"{scene['music']}-{scene['head_loudspeaker_positions']}"
            song_path = Path(config.path.music_dir) / songs[song_key]["Path"]
            try:
                mix = read_signal(song_path / "mixture.wav", config.sample_rate)
            except Exception as e:
                logger.warning(f"Failed to read {song_path / 'mixture.wav'}: {e}")
                continue

            mix = mix[:len(mix) - len(mix) % segment_samples]
            segment_positions = list(range(0, len(mix), segment_samples))
            random.shuffle(segment_positions)

            for seg_idx in segment_positions:
                if segments_used >= config.train.max_segments_per_epoch:
                    break

                segment = mix[seg_idx: seg_idx + segment_samples]
                if segment.shape[0] < segment_samples:
                    continue

                segment_rms = compute_rms(segment)
                percentile_threshold = np.percentile(np.abs(segment), 95)
                if segment_rms < 0.01 and percentile_threshold < 0.01:
                    logger.info(
                        f"Skipping mostly silent segment at {seg_idx}, RMS: {segment_rms}, 95th percentile: {percentile_threshold}")
                    continue

                segments_used += 1
                signal = torch.tensor(segment.T, dtype=torch.float32, requires_grad=True).to(device)
                listener = listeners[listener_id]
                audiogram_np = np.stack([listener.audiogram_left.levels, listener.audiogram_right.levels])
                audiogram_np = audiogram_np / 80
                # logger.info(f"Audiogram for listener {listener_id} at {seg_idx}: {audiogram_np.flatten().tolist()}")
                audiogram = torch.tensor(audiogram_np, dtype=torch.float32).unsqueeze(0).to(device)

                if torch.isnan(audiogram).any() or torch.isinf(audiogram).any():
                    logger.error(f"Audiogram contains NaN or Inf: {audiogram}")
                    raise ValueError("Audiogram contains invalid values")

                # Pass the tensor directly to decompose_signal
                stems_dict = separator.decompose_signal(signal, config.sample_rate, device, normalise=False)
                gain_matrix = gain_model(audiogram)

                if torch.isnan(gain_matrix).any() or torch.isinf(gain_matrix).any():
                    logger.error(f"Gain matrix contains NaN or Inf at seg_idx {seg_idx}: {gain_matrix}")
                    # continue

                enhanced_stems = apply_gains(stems_dict, gain_matrix.squeeze())

                reference_stems = {}
                for i, src in enumerate(separator.sources):
                    try:
                        stem = read_signal(song_path / f"{src}.wav", config.sample_rate)
                    except Exception as e:
                        logger.warning(f"Failed to read {song_path / f'{src}.wav'}: {e}")
                        continue
                    stem = stem[seg_idx: seg_idx + segment_samples]
                    if stem.shape[0] < stem.shape[1]:
                        stem = stem.T
                    reference_stems[src] = torch.tensor(stem, dtype=torch.float32, device=device, requires_grad=True)
                reference_stems = apply_gains(reference_stems, gain_matrix.squeeze())

                for src in separator.sources:
                    enhanced_stems_batch[src].append(enhanced_stems[src].T)
                    reference_stems_batch[src].append(reference_stems[src])
                hearing_levels_batch.append(audiogram.squeeze(0).mean(0))
                batch_segment_count += 1
                batch_seg_indices.append(seg_idx)

                if batch_segment_count == batch_size:
                    # logger.info(f"Predicted gains at {seg_idx}: {gain_matrix.squeeze().tolist()}")

                    hearing_levels_batch_tensor = torch.stack(hearing_levels_batch).to(device)

                    haaqi_net_model = load_haaqi_net_model(
                        config_path=str(Path.home() / "haaqinet/HAAQI-Net/src/config.yaml"),
                        model_path=str(Path.home() / "haaqinet/HAAQI-Net/model/haaqi_net_best.pth"),
                        device=device
                    )
                    for param in haaqi_net_model.parameters():
                        param.requires_grad = False

                    batch_loss_dict = compute_loss_and_backprop(
                        enhanced_stems_batch,
                        reference_stems_batch,
                        hearing_levels_batch_tensor,
                        haaqi_net_model,
                        optimizer,
                        gain_model,
                        separator,
                        alpha=10.0
                    )

                    # Logg the training result of the batch
                    training_logger.add_result({
                        "epoch": epoch + 1,
                        "batch": batch_count + 1,
                        "total_loss": batch_loss_dict["total_loss"],
                        "mse_loss": batch_loss_dict["mse_loss"],
                        "sdr_loss": batch_loss_dict["sdr_loss"],
                        "loudness_loss": batch_loss_dict["loudness_loss"],
                        "perceptual_loss": batch_loss_dict["perceptual_loss"],
                        "haaqi_score": batch_loss_dict["haaqi_score"],
                    })

                    epoch_loss += batch_loss_dict['total_loss']
                    epoch_mse += batch_loss_dict['mse_loss']
                    epoch_sdr += batch_loss_dict['sdr_loss']
                    epoch_loudness += batch_loss_dict['loudness_loss']
                    epoch_perceptual += batch_loss_dict['perceptual_loss']
                    batch_count += 1

                    for i in range(batch_size):
                        enhanced_stems_dict = {src: enhanced_stems_batch[src][i].detach().cpu().numpy()
                                              for src in separator.sources}
                        enhanced_np = remix_stems(enhanced_stems_dict, segment, config.sample_rate)
                        enhancer = NALR(**config.nalr)
                        enhanced_np = process_remix_for_listener(
                            signal=enhanced_np,
                            enhancer=enhancer,
                            compressor=None,
                            listener=listener,
                            apply_compressor=config.apply_compressor,
                        )

                        current_seg_idx = batch_seg_indices[i]
                        flac_filename = exp_folder / f"{scene_id}_{listener_id}_seg{current_seg_idx}_enhanced.flac"
                        save_flac_signal(
                            enhanced_np,
                            flac_filename,
                            config.sample_rate,
                            config.sample_rate,
                            do_clip_signal=True,
                            do_soft_clip=config.soft_clip
                        )

                    # Reset the batch
                    enhanced_stems_batch = {src: [] for src in separator.sources}
                    reference_stems_batch = {src: [] for src in separator.sources}
                    hearing_levels_batch.clear()
                    batch_segment_count = 0
                    batch_seg_indices.clear()

        validate(gain_model, separator, config, epoch + 1)
        total_losses.append(epoch_loss)
        mse_losses.append(epoch_mse)
        sdr_losses.append(epoch_sdr)
        loudness_losses.append(epoch_loudness)
        perceptual_losses.append(epoch_perceptual)

        logger.info(
            f"Epoch {epoch+1}: Total Loss={epoch_loss:.4f}, MSE={epoch_mse:.4f}, SDR={epoch_sdr:.4f}, "
            f"Loudness={epoch_loudness:.4f}, Perceptual={epoch_perceptual:.4f}"
        )

    plt.plot(total_losses, label="Total")
    plt.plot(mse_losses, label="MSE")
    plt.plot(sdr_losses, label="SDR")
    plt.plot(loudness_losses, label="Loudness")
    plt.plot(perceptual_losses, label="Perceptual")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    train()
    # gain_model = GainPredictor().to(device)
    # separator = SourceSeparator().to(device)
    # # Load config
    # config = OmegaConf.load("config.yaml")
    # validate(gain_model, separator, config, 1)

