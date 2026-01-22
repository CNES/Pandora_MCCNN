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

import threading
import time

import psutil


def get_memory_usage_bytes() -> int:
    """
    Get current process RSS in bytes.

    :return: current process RSS in bytes
    """
    return psutil.Process().memory_info().rss


def bytes_to_mb(num_bytes: int) -> float:
    """
    Convert bytes to mega bytes.

    :param num_bytes: number of bytes

    :return: number of mega bytes
    """
    return num_bytes / (1024.0 * 1024.0)


class MemorySampler:
    """
    Background sampler to capture true peak RSS during a stage.

    Usage example:
    ms = MemorySampler().start()
    do_something()
    ms.stop()
    mem_import_peak = ms.peak_mb
    """

    def __init__(self, sampling_interval_sec: float = 0.0005):
        self.interval = max(0.0005, sampling_interval_sec)
        self._stop = threading.Event()
        self._thread = None
        self._peak = 0

    def _run(self):
        """
        Run the memory sampler.
        """
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss
                self._peak = max(self._peak, rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        """
        Start the memory sampler.
        """
        self._peak = get_memory_usage_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mem_sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """
        Stop the memory sampler.
        """
        self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join()
            except Exception:
                pass

    @property
    def peak_mb(self) -> float:
        """
        Return the peak RSS in MB.
        """
        return bytes_to_mb(self._peak)
