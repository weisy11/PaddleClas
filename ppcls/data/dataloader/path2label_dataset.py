#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import os
from .common_dataset import CommonDataset


class Path2LabelDataset(CommonDataset):
    """Path2LabelDataset

    Args:
        image_root (str): image root, path to `ILSVRC2012`
        cls_label_path (str): path to annotation file `train_list.txt` or `val_list.txt`
        transform_ops (list, optional): list of transform op(s). Defaults to None.
        delimiter (str, optional): delimiter. Defaults to None.
    """

    def __init__(self,
                 image_root,
                 cls_label_path,
                 transform_ops=None,
                 delimiter=None):
        self.delimiter = delimiter if delimiter is not None else " "
        super(Path2LabelDataset, self).__init__(image_root, cls_label_path,
                                                transform_ops)

    def _load_anno(self, seed=None):
        assert os.path.exists(
            self._cls_path), f"path {self._cls_path} does not exist."
        assert os.path.exists(
            self._img_root), f"path {self._img_root} does not exist."
        self.images = []
        self.labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()

            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for line in lines:
                line = line.strip().split(self.delimiter)
                self.images.append(os.path.join(self._img_root, line[0]))
                self.labels.append(os.path.join(self._img_root, line[0]))
                assert os.path.exists(self.images[
                    -1]), f"path {self.images[-1]} does not exist."
