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

from typing import Any


def parse_buttons(text: str) -> dict[str, Any]:
    split_text = text.split(",")
    buttons: dict[str, Any] = {}
    if "R" in split_text:
        split_text.remove("R")
        buttons.update(
            {
                "A": False,
                "B": False,
                "RThU": False,
                "RJ": False,
                "RG": False,
                "RTr": False,
            }
        )
    if "L" in split_text:
        split_text.remove("L")
        buttons.update({"X": False, "Y": False, "LThU": False, "LJ": False, "LG": False, "LTr": False})
    for key in list(buttons.keys()):
        if key in split_text:
            buttons[key] = True
            split_text.remove(key)
    for elem in split_text:
        split_elem = elem.split(" ")
        if len(split_elem) < 2:
            continue
        key = split_elem[0]
        value = tuple(float(x) for x in split_elem[1:])
        buttons[key] = value
    return buttons
