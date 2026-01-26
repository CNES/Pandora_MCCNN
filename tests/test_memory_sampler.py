#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA_MCCNN
#
#     https://github.com/CNES/Pandora_MCCNN
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
#
"""
This module contains functions to test the memory sampler
"""
import time

from mc_cnn.profiling import MemorySampler


def test_init_default_values():
    """
    Test initial default values
    """
    sampler = MemorySampler()

    assert sampler.interval == 0.0005
    assert sampler._thread is None
    assert sampler._peak == 0
    assert sampler._stop.is_set() is False


def test_init_with_custom_interval():
    """
    Test initialization with custom sampling interval
    """
    sampler = MemorySampler(sampling_interval_sec=0.01)

    assert sampler.interval == 0.01


def test_init_with_custom_interval_lower_than_0_0005():
    """
    Test initialization with a custom sampling interval lower than 0.0005.
    """
    sampler = MemorySampler(sampling_interval_sec=0.000001)

    assert sampler.interval == 0.0005


def test_thread_stop():
    """
    Test thread not alive after stop
    """
    sampler = MemorySampler()
    sampler.start()
    sampler.stop()

    assert not sampler._thread.is_alive()


def test_start_sets_peak_and_thread():
    """
    Test thread alive after start and peak > 0
    """
    sampler = MemorySampler(sampling_interval_sec=0.01)

    sampler.start()

    assert sampler._thread is not None
    assert sampler._thread.is_alive()
    assert sampler._thread.daemon is True
    assert sampler._peak >= 0

    sampler.stop()


def test_peak_mb_property():
    """
    Test the peak_mb property
    """
    sampler = MemorySampler()
    sampler._peak = 1024 * 1024

    peak_mb = sampler.peak_mb

    assert peak_mb == 1.0


def test_start_stop_peak_mb_property():
    """
    Test if the peak_mb property > 0 after running sampler
    """
    sampler = MemorySampler()

    sampler.start()
    time.sleep(0.05)
    sampler.stop()

    assert sampler.peak_mb > 0


def test_deteck_peak():
    """
    Test the detection of a memory peak
    """
    array_size = 10000000
    sampler = MemorySampler()

    sampler.start()
    big_array = list(range(array_size))
    sampler.stop()

    del big_array

    assert sampler.peak_mb > array_size * 28 / (1024 * 1024)  # 28 bytes size for small int
