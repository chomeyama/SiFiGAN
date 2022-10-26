# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/bigpon/QPPWG

"""

from logging import getLogger
from multiprocessing import Manager

import librosa
import numpy as np
import soundfile as sf
from hydra.utils import to_absolute_path
from joblib import load
from sifigan.utils import check_filename, read_hdf5, read_txt, validate_length
from torch.utils.data import Dataset

# A logger for this file
logger = getLogger(__name__)


class AudioFeatDataset(Dataset):
    """PyTorch compatible audio and acoustic feat. dataset."""

    def __init__(
        self,
        stats,
        audio_list,
        feat_list,
        audio_length_threshold=None,
        feat_length_threshold=None,
        return_filename=False,
        allow_cache=False,
        sample_rate=24000,
        hop_size=120,
        aux_feats=["mcep", "bap"],
    ):
        """Initialize dataset.

        Args:
            stats (str): Filename of the statistic hdf5 file.
            audio_list (str): Filename of the list of audio files.
            feat_list (str): Filename of the list of feature files.
            audio_length_threshold (int): Threshold to remove short audio files.
            feat_length_threshold (int): Threshold to remove short feature files.
            return_filename (bool): Whether to return the filename with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
            sample_rate (int): Sampling frequency.
            hop_size (int): Hope size of acoustic feature
            aux_feats (str): Type of auxiliary features.

        """
        # load audio and feature files & check filename
        audio_files = read_txt(to_absolute_path(audio_list))
        feat_files = read_txt(to_absolute_path(feat_list))
        assert check_filename(audio_files, feat_files)

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [sf.read(to_absolute_path(f)).shape[0] for f in audio_files]
            idxs = [
                idx
                for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logger.warning(
                    f"Some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            feat_files = [feat_files[idx] for idx in idxs]
        if feat_length_threshold is not None:
            f0_lengths = [
                read_hdf5(to_absolute_path(f), "/f0").shape[0] for f in feat_files
            ]
            idxs = [
                idx
                for idx in range(len(feat_files))
                if f0_lengths[idx] > feat_length_threshold
            ]
            if len(feat_files) != len(idxs):
                logger.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(feat_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            feat_files = [feat_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"${audio_list} is empty."
        assert len(audio_files) == len(
            feat_files
        ), f"Number of audio and features files are different ({len(audio_files)} vs {len(feat_files)})."

        self.audio_files = audio_files
        self.feat_files = feat_files
        self.return_filename = return_filename
        self.allow_cache = allow_cache
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.aux_feats = aux_feats
        logger.info(f"Feature type : {self.aux_feats}")

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

        # define feature pre-processing function
        self.scaler = load(stats)

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_filename = True).
            ndarray: Audio signal (T,).
            ndarray: Auxiliary features (T', C).
            ndarray: F0 sequence (T', 1).
            ndarray: Continuous F0 sequence (T', 1).¥

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]
        # load audio and features
        audio, sr = sf.read(to_absolute_path(self.audio_files[idx]))
        if sr != self.sample_rate:
            logger.warning(
                f"Resampling {self.audio_files[idx]} incurs extra computational cost."
                + "It is recommended to prepare audio files with the desired sampling rate in advance."
            )
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        # audio & feature pre-processing
        audio = audio.astype(np.float32)

        # get auxiliary features
        aux_feats = []
        for feat_type in self.aux_feats:
            aux_feat = read_hdf5(
                to_absolute_path(self.feat_files[idx]), f"/{feat_type}"
            )
            aux_feat = self.scaler[f"{feat_type}"].transform(aux_feat)
            aux_feats += [aux_feat]
        aux_feats = np.concatenate(aux_feats, axis=1)

        # get dilated factor sequences
        f0 = read_hdf5(to_absolute_path(self.feat_files[idx]), "/f0")  # descrete F0
        cf0 = read_hdf5(to_absolute_path(self.feat_files[idx]), "/cf0")  # continuous F0

        # adjust length
        aux_feats, f0, cf0, audio = validate_length(
            (aux_feats, f0, cf0), (audio,), self.hop_size
        )

        if self.return_filename:
            items = self.feat_files[idx], audio, aux_feats, f0, cf0
        else:
            items = audio, aux_feats, f0, cf0

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class FeatDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(
        self,
        stats,
        feat_list,
        feat_length_threshold=None,
        return_filename=False,
        allow_cache=False,
        sample_rate=24000,
        hop_size=120,
        aux_feats=["mcep", "bap"],
        f0_factor=1.0,
    ):
        """Initialize dataset.

        Args:
            stats (str): Filename of the statistic hdf5 file.
            feat_list (str): Filename of the list of feature files.
            feat_length_threshold (int): Threshold to remove short feature files.
            return_filename (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
            sample_rate (int): Sampling frequency.
            hop_size (int): Hope size of acoustic feature
            aux_feats (str): Type of auxiliary features.
            f0_factor (float): Ratio of scaled f0.

        """
        # load feat. files
        feat_files = read_txt(to_absolute_path(feat_list))

        # filter by threshold
        if feat_length_threshold is not None:
            f0_lengths = [
                read_hdf5(to_absolute_path(f), "/f0").shape[0] for f in feat_files
            ]
            idxs = [
                idx
                for idx in range(len(feat_files))
                if f0_lengths[idx] > feat_length_threshold
            ]
            if len(feat_files) != len(idxs):
                logger.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(feat_files)} -> {len(idxs)})."
                )
            feat_files = [feat_files[idx] for idx in idxs]

        # assert the number of files
        assert len(feat_files) != 0, f"${feat_list} is empty."

        self.feat_files = feat_files
        self.return_filename = return_filename
        self.allow_cache = allow_cache
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.aux_feats = aux_feats
        self.f0_factor = f0_factor
        logger.info(f"Feature type : {self.aux_feats}")

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(feat_files))]

        # define feature pre-processing function
        self.scaler = load(stats)

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_filename = True).
            ndarray: Auxiliary feature (T', C).
            ndarray: F0 sequence (T', 1).
            ndarray: Continuous F0 sequence (T', 1).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        # get auxiliary features
        aux_feats = []
        for feat_type in self.aux_feats:
            aux_feat = read_hdf5(
                to_absolute_path(self.feat_files[idx]), f"/{feat_type}"
            )
            if feat_type in ["f0", "cf0"]:  # f0 scaling
                aux_feat *= self.f0_factor
            aux_feat = self.scaler[f"{feat_type}"].transform(aux_feat)
            aux_feats += [aux_feat]
        aux_feats = np.concatenate(aux_feats, axis=1)

        # get f0 sequences
        f0 = read_hdf5(to_absolute_path(self.feat_files[idx]), "/f0")  # descrete F0
        cf0 = read_hdf5(to_absolute_path(self.feat_files[idx]), "/cf0")  # continuous F0

        # adjust length
        aux_feats, f0, cf0 = validate_length((aux_feats, f0, cf0))

        # f0 scaling
        f0 *= self.f0_factor
        cf0 *= self.f0_factor

        if self.return_filename:
            items = self.feat_files[idx], aux_feats, f0, cf0
        else:
            items = aux_feats, f0, cf0

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.feat_files)
