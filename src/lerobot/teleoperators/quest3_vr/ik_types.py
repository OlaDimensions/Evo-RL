#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class IKSolveResult:
    """Standard result container for Quest3 VR IK backends."""

    q: np.ndarray | None
    success: bool
    collision_free: bool
    solve_ms: float
    reason: str = ""


class IKBackend(abc.ABC):
    """Abstract interface for inverse-kinematics backends used by Quest3 VR."""

    @abc.abstractmethod
    def solve(self, target_T: np.ndarray, q_seed: np.ndarray | None = None) -> IKSolveResult:
        raise NotImplementedError

    @abc.abstractmethod
    def fk(self, q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def home_q(self) -> np.ndarray:
        raise NotImplementedError

    def set_previous_q(self, q: np.ndarray | None) -> None:
        return

    def clear_previous_q(self) -> None:
        return

    def clip(self, q: np.ndarray) -> np.ndarray:
        return q


class JointSeedProvider(Protocol):
    def __call__(self) -> np.ndarray | None: ...
