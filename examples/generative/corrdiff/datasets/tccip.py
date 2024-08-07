# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import logging
import random

import cftime
import cv2
from hydra.utils import to_absolute_path
import numpy as np
import zarr

from .base import ChannelMetadata, DownscalingDataset
from .img_utils import reshape_fields
from .norm import denormalize, normalize

logger = logging.getLogger(__file__)


def get_target_normalizations_v1(group):
    """Get target normalizations using center and scale values from the 'group'."""
    return group["TCCIP_center"][:], group["TCCIP_scale"][:]


def get_target_normalizations_v2(group):
    """Change the normalizations of the non-gaussian output variables"""
    center = group["TCCIP_center"]
    scale = group["TCCIP_scale"]
    variable = group["TCCIP_variable"]

    center = np.where(variable == "maximum_radar_reflectivity", 25.0, center)
    center = np.where(variable == "eastward_wind_10m", 0.0, center)
    center = np.where(variable == "northward_wind_10m", 0, center)

    scale = np.where(variable == "maximum_radar_reflectivity", 25.0, scale)
    scale = np.where(variable == "eastward_wind_10m", 20.0, scale)
    scale = np.where(variable == "northward_wind_10m", 20.0, scale)
    return center, scale


class _ZarrDataset(DownscalingDataset):
    """A Dataset for loading paired training data from a Zarr-file

    This dataset should not be modified to add image processing contributions.
    """

    path: str

    def __init__(
        self, path: str, get_target_normalization=get_target_normalizations_v1
    ):
        #TODO add mask path in config
        # self.valid_mask = np.load("/ws_src/TCCIPERA5_2013_2022/valid_mask.npy")
        # self.valid_mask = torch.from_numpy(valid_mask).cuda()
        
        self.path = path
        self.group = zarr.open_consolidated(path)
        self.get_target_normalization = get_target_normalization

        # valid indices
        TCCIP_valid = self.group["TCCIP_valid"]
        ERA5_valid = self.group["ERA5_valid"]
        if not (
            ERA5_valid.ndim == 2
            and TCCIP_valid.ndim == 1
            and TCCIP_valid.shape[0] == ERA5_valid.shape[0]
        ):
            raise ValueError("Invalid dataset shape")
        ERA5_all_channels_valid = np.all(ERA5_valid, axis=-1)
        valid_times = TCCIP_valid & ERA5_all_channels_valid
        # need to cast to bool since cwb_valis is stored as an int8 type in zarr.
        self.valid_times = valid_times != 0

        logger.info("Number of valid times: %d", len(self))
        logger.info("input_channels:%s", self.input_channels())
        logger.info("output_channels:%s", self.output_channels())

    def _get_valid_time_index(self, idx):
        time_indexes = np.arange(self.group["time"].size)
        if not self.valid_times.dtype == np.bool_:
            raise ValueError("valid_times must be a boolean array")
        valid_time_indexes = time_indexes[self.valid_times]
        return valid_time_indexes[idx]

    def __getitem__(self, idx):
        idx_to_load = self._get_valid_time_index(idx)
        target = self.group["TCCIP"][idx_to_load]
        input = self.group["ERA5"][idx_to_load]
        label = 0

        target = self.normalize_output(target[None, ...])[0]
        input = self.normalize_input(input[None, ...])[0]

        # target = target * self.valid_mask

        return target, input, label

    def longitude(self):
        """The longitude. useful for plotting"""
        return np.tile(np.expand_dims(np.array(self.group["lon"])[::-1], axis=0), (len(self.group["lon"]),1))
        # return self.group["lon"]

    def latitude(self):
        """The latitude. useful for plotting"""
        return np.tile(np.expand_dims(self.group["lat"], axis=1),(1,len(self.group["lat"])))
        # return self.group["lat"]

    def _get_channel_meta(self, variable, level):
        if np.isnan(level):
            level = ""
        return ChannelMetadata(name=variable, level=str(level))

    def input_channels(self):
        """Metadata for the input channels. A list of dictionaries, one for each channel"""
        variable = self.group["ERA5_variable"]
        level = self.group["ERA5_pressure"]
        return [self._get_channel_meta(*v) for v in zip(variable, level)]

    def output_channels(self):
        """Metadata for the output channels. A list of dictionaries, one for each channel"""
        variable = self.group["TCCIP_variable"]
        level = self.group["TCCIP_pressure"]
        return [self._get_channel_meta(*v) for v in zip(variable, level)]

    def _read_time(self):
        """The vector of time coordinate has length (self)"""

        return cftime.num2date(
            self.group["time"], units=self.group["time"].attrs["units"]
        )

    def time(self):
        """The vector of time coordinate has length (self)"""
        time = self._read_time()
        return time[self.valid_times].tolist()

    def image_shape(self):
        """Get the shape of the image (same for input and output)."""
        return self.group["TCCIP"].shape[-2:]

    def _select_norm_channels(self, means, stds, channels):
        if channels is not None:
            means = means[channels]
            stds = stds[channels]
        return (means, stds)

    def normalize_input(self, x, channels=None):
        """Convert input from physical units to normalized data."""
        norm = self._select_norm_channels(
            self.group["ERA5_center"], self.group["ERA5_scale"], channels
        )
        return normalize(x, *norm)

    def denormalize_input(self, x, channels=None):
        """Convert input from normalized data to physical units."""
        norm = self._select_norm_channels(
            self.group["ERA5_center"], self.group["ERA5_scale"], channels
        )
        return denormalize(x, *norm)

    def normalize_output(self, x, channels=None):
        """Convert output from physical units to normalized data."""
        norm = self.get_target_normalization(self.group)
        norm = self._select_norm_channels(*norm, channels)
        return normalize(x, *norm)

    def denormalize_output(self, x, channels=None):
        """Convert output from normalized data to physical units."""
        norm = self.get_target_normalization(self.group)
        norm = self._select_norm_channels(*norm, channels)
        return denormalize(x, *norm)

    def info(self):
        return {
            "target_normalization": self.get_target_normalization(self.group),
            "input_normalization": (
                self.group["ERA5_center"][:],
                self.group["ERA5_scale"][:],
            ),
        }

    def __len__(self):
        return self.valid_times.sum()


class FilterTime(DownscalingDataset):
    """Filter a time dependent dataset"""

    def __init__(self, dataset, filter_fn):
        """
        Args:
            filter_fn: if filter_fn(time) is True then return point
        """
        self._dataset = dataset
        self._filter_fn = filter_fn
        self._indices = [i for i, t in enumerate(self._dataset.time()) if filter_fn(t)]

    def longitude(self):
        """Get longitude values from the dataset."""
        return self._dataset.longitude()

    def latitude(self):
        """Get latitude values from the dataset."""
        return self._dataset.latitude()

    def input_channels(self):
        """Metadata for the input channels. A list of dictionaries, one for each channel"""
        return self._dataset.input_channels()

    def output_channels(self):
        """Metadata for the output channels. A list of dictionaries, one for each channel"""
        return self._dataset.output_channels()

    def time(self):
        """Get time values from the dataset."""
        time = self._dataset.time()
        return [time[i] for i in self._indices]

    def info(self):
        """Get information about the dataset."""
        return self._dataset.info()

    def image_shape(self):
        """Get the shape of the image (same for input and output)."""
        return self._dataset.image_shape()

    def normalize_input(self, x, channels=None):
        """Convert input from physical units to normalized data."""
        return self._dataset.normalize_input(x, channels=channels)

    def denormalize_input(self, x, channels=None):
        """Convert input from normalized data to physical units."""
        return self._dataset.denormalize_input(x, channels=channels)

    def normalize_output(self, x, channels=None):
        """Convert output from physical units to normalized data."""
        return self._dataset.normalize_output(x, channels=channels)

    def denormalize_output(self, x, channels=None):
        """Convert output from normalized data to physical units."""
        return self._dataset.denormalize_output(x, channels=channels)

    def __getitem__(self, idx):
        return self._dataset[self._indices[idx]]

    def __len__(self):
        return len(self._indices)


def is_2022(time):
    """Check if the given time is in the year 2022."""
    return time.year == 2022


def is_not_2022(time):
    """Check if the given time is not in the year 2022."""
    return not is_2022(time)


class ZarrDataset(DownscalingDataset):
    """A Dataset for loading paired training data from a Zarr-file with the
    following schema::

        xarray.Dataset {
        dimensions:
                south_north = 384 ;
                west_east = 384 ;
                time = 3650 ;
                TCCIP_channel = 4 ;
                ERA5_channel = 20 ;


    // global attributes:
    }
    """

    path: str

    def __init__(
        self,
        dataset,
        in_channels=(0, 1, 2, 3, 4, 8, 12, 16),
        out_channels=(0, 1, 2, 3),
        img_shape_x=384,
        img_shape_y=384,
        roll=False,
        add_grid=True,
        ds_factor=1,
        train=True,
        all_times=False,
        n_history=0,
        min_path=None,
        max_path=None,
        global_means_path=None,
        global_stds_path=None,
        normalization="v1",
    ):
        if not all_times:
            self._dataset = (
                FilterTime(dataset, is_not_2022)
                if train
                else FilterTime(dataset, is_2022)
            )
        else:
            self._dataset = dataset

        self.train = train
        self.img_shape_x = img_shape_x
        self.img_shape_y = img_shape_y
        self.roll = roll
        self.grid = add_grid
        self.ds_factor = ds_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_history = n_history
        self.min_path = min_path
        self.max_path = max_path
        self.global_means_path = (
            to_absolute_path(global_means_path)
            if (global_means_path is not None)
            else None
        )
        self.global_stds_path = (
            to_absolute_path(global_stds_path)
            if (global_stds_path is not None)
            else None
        )
        self.normalization = normalization

    def info(self):
        """Check if the given time is not in the year 2021."""
        return self._dataset.info()

    def __getitem__(self, idx):
        (target, input, _) = self._dataset[idx]
        # crop and downsamples
        # rolling
        if self.train and self.roll:
            y_roll = random.randint(0, self.img_shape_y)
        else:
            y_roll = 0

        # channels
        input = input[self.in_channels, :, :]
        target = target[self.out_channels, :, :]
        
        if self.ds_factor > 1:
            target = self._create_lowres_(target, factor=self.ds_factor)

        reshape_args = (
            y_roll,
            self.train,
            self.n_history,
            self.in_channels,
            self.out_channels,
            self.img_shape_x,
            self.img_shape_y,
            self.min_path,
            self.max_path,
            self.global_means_path,
            self.global_stds_path,
            self.normalization,
            self.roll,
        )
        # SR
        input = reshape_fields(
            input,
            "inp",
            *reshape_args,
            normalize=False,
        )  # 3x720x1440
        target = reshape_fields(
            target, "tar", *reshape_args, normalize=False
        )  # 3x720x1440

        return target, input, idx

    def input_channels(self):
        """Metadata for the input channels. A list of dictionaries, one for each channel"""
        in_channels = self._dataset.input_channels()
        in_channels = [in_channels[i] for i in self.in_channels]
        return in_channels

    def output_channels(self):
        """Metadata for the output channels. A list of dictionaries, one for each channel"""
        out_channels = self._dataset.output_channels()
        return [out_channels[i] for i in self.out_channels]

    def __len__(self):
        return len(self._dataset)

    def longitude(self):
        """Get longitude values from the dataset."""
        lon = self._dataset.longitude()
        return lon if self.train else lon[..., : self.img_shape_x]

    def latitude(self):
        """Get latitude values from the dataset."""
        lat = self._dataset.latitude()
        return lat if self.train else lat[..., : self.img_shape_y]

    def time(self):
        """Get time values from the dataset."""
        return self._dataset.time()

    def image_shape(self):
        """Get the shape of the image (same for input and output)."""
        return (self.img_shape_x, self.img_shape_y)

    def normalize_input(self, x):
        """Convert input from physical units to normalized data."""
        x_norm = self._dataset.normalize_input(
            x[:, : len(self.in_channels)], channels=self.in_channels
        )
        return np.concatenate((x_norm, x[:, self.in_channels :]), axis=1)

    def denormalize_input(self, x):
        """Convert input from normalized data to physical units."""
        x_denorm = self._dataset.denormalize_input(
            x[:, : len(self.in_channels)], channels=self.in_channels
        )
        return np.concatenate((x_denorm, x[:, len(self.in_channels) :]), axis=1)

    def normalize_output(self, x):
        """Convert output from physical units to normalized data."""
        return self._dataset.normalize_output(x, channels=self.out_channels)

    def denormalize_output(self, x):
        """Convert output from normalized data to physical units."""
        return self._dataset.denormalize_output(x, channels=self.out_channels)

    def _create_highres_(self, x, shape):
        # downsample the high res imag
        x = x.transpose(1, 2, 0)
        # upsample with bicubic interpolation to bring the image to the nominal size
        x = cv2.resize(
            x, (shape[0], shape[1]), interpolation=cv2.INTER_CUBIC
        )  # 32x32x3
        x = x.transpose(2, 0, 1)  # 3x32x32
        return x

    def _create_lowres_(self, x, factor=4):
        # downsample the high res imag
        x = x.transpose(1, 2, 0)
        x = x[::factor, ::factor, :]  # 8x8x3  #subsample
        # upsample with bicubic interpolation to bring the image to the nominal size
        x = cv2.resize(
            x, (x.shape[1] * factor, x.shape[0] * factor), interpolation=cv2.INTER_CUBIC
        )  # 32x32x3
        x = x.transpose(2, 0, 1)  # 3x32x32
        return x


def get_zarr_dataset(*, data_path, normalization="v1", all_times=False, **kwargs):
    """Get a Zarr dataset for training or evaluation."""
    data_path = to_absolute_path(data_path)
    get_target_normalization = {
        "v1": get_target_normalizations_v1,
        "v2": get_target_normalizations_v2,
    }[normalization]
    logger.info(f"Normalization: {normalization}")
    zdataset = _ZarrDataset(
        data_path, get_target_normalization=get_target_normalization
    )    
    return ZarrDataset(
        dataset=zdataset, normalization=normalization, all_times=all_times, **kwargs
    )
