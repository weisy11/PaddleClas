# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division

from collections import defaultdict

import numpy as np
import paddle.distributed as dist
from paddle.io import DistributedBatchSampler

from ppcls.utils import logger


class IDSampler(DistributedBatchSampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        super().__init__(dataset, batch_size, drop_last=drop_last)
        assert hasattr(self.dataset,
                       "labels"), "Dataset must have labels attribute."
        self.batch_size = batch_size
        self.label_dict = defaultdict(list)
        self.drop_last = drop_last
        rank = dist.get_rank()
        num_replicas = dist.get_world_size()
        for idx, label in enumerate(self.dataset.labels):
            self.label_dict[label].append(idx)
        label_list = list(self.label_dict)
        self.label_list = label_list[rank * len(label_list) // num_replicas:(
            rank + 1) * len(label_list) // num_replicas]

    def __len__(self):
        if self.drop_last or not len(self.label_list) % self.batch_size:
            return len(self.label_list) // self.batch_size
        else:
            return len(self.label_list) // self.batch_size + 1

    def __iter__(self):
        for i in range(len(self)):
            batch_label_list = self.label_list[i * self.batch_size:(i + 1) *
                                               self.batch_size]
            batch_index = []
            for label_i in batch_label_list:
                label_i_indexes = self.label_dict[label_i]
                batch_index.extend(np.random.choice(label_i_indexes, 1))
            yield batch_index


class GraphSampler(DistributedBatchSampler):
    """First, randomly sample m identities.
        Then for each identity randomly sample n - 1 neighbour identities, so we have m * n identities totally
        Then for each identity randomly sample k instances.
        Therefore batch size equals to m * n * K, and the sampler called PKSampler.

    Args:
        dataset (Dataset): Dataset which contains list of (img_path, pid, camid))
        batch_size (int): batch size
        sample_per_id (int): number of instance(s) within an class
        shuffle (bool, optional): _description_. Defaults to True.
        id_list(list): list of (start_id, end_id, start_id, end_id) for set of ids to duplicated.
        ratio(list): list of (ratio1, ratio2..) the duplication number for ids in id_list.
        drop_last (bool, optional): whether to discard the data at the end. Defaults to True.
        sample_method (str, optional): sample method when generating prob_list. Defaults to "sample_avg_prob".
        total_epochs (int, optional): total epochs. Defaults to 0.
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 sample_per_id,
                 neighbour_num=1,
                 shuffle=True,
                 drop_last=True,
                 id_list=None,
                 ratio=None,
                 total_epochs=0):
        super().__init__(
            dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
        assert batch_size % sample_per_id == 0, \
            f"PKSampler configs error, sample_per_id({sample_per_id}) must be a divisor of batch_size({batch_size})."
        assert hasattr(self.dataset,
                       "labels"), "Dataset must have labels attribute."
        self.sample_per_id = sample_per_id
        self.label_dict = defaultdict(list)
        self.total_epochs = total_epochs
        self.neighbour_num = neighbour_num
        self.neighbour_map = {}
        for idx, label in enumerate(self.dataset.labels):
            self.label_dict[label].append(idx)
        self.label_list = list(self.label_dict)
        assert len(self.label_list) * self.sample_per_id >= self.batch_size, \
            f"batch size({self.batch_size}) should not be bigger than than #classes({len(self.label_list)})*sample_per_id({self.sample_per_id})"
        self.prob_list = np.array([1 / len(self.label_list)] *
                                  len(self.label_list))

        if id_list and ratio:
            assert len(id_list) % 2 == 0 and len(id_list) == len(ratio) * 2
            for i in range(len(self.prob_list)):
                for j in range(len(ratio)):
                    if i >= id_list[j * 2] and i <= id_list[j * 2 + 1]:
                        self.prob_list[i] = self.prob_list[i] * ratio[j]
                        break
            self.prob_list = self.prob_list / sum(self.prob_list)

        diff = np.abs(sum(self.prob_list) - 1)
        if diff > 0.00000001:
            self.prob_list[-1] = 1 - sum(self.prob_list[:-1])
            if self.prob_list[-1] > 1 or self.prob_list[-1] < 0:
                logger.error("PKSampler prob list error")
            else:
                logger.info(
                    "PKSampler: sum of prob list not equal to 1, diff is {}, change the last prob".
                    format(diff))

    def set_neighbour_map(self, neighbour_map):
        self.neighbour_map = neighbour_map

    def __iter__(self):
        # shuffle manually, same as DistributedBatchSampler.__iter__
        if self.shuffle:
            rank = dist.get_rank()
            np.random.RandomState(rank * self.total_epochs +
                                  self.epoch).shuffle(self.label_list)
            self.epoch += 1

        center_num = self.batch_size // self.sample_per_id // (
            self.neighbour_num + 1)
        for _ in range(len(self)):
            batch_index = []
            batch_label_list = []
            center_list = np.random.choice(
                self.label_list,
                size=center_num,
                replace=False,
                p=self.prob_list)
            batch_label_list.extend(center_list.copy())
            for label_i in center_list:
                if label_i not in self.neighbour_map:
                    logger.warning("Can not find neighbour of sample {}.",
                                   label_i)
                    batch_label_list.extend(
                        np.random.choice(
                            self.label_list,
                            size=self.neighbour_num,
                            replace=False,
                            p=self.prob_list))
                else:
                    neighbour_i = self.neighbour_map[label_i]
                    batch_label_list.extend(
                        np.random.choice(
                            neighbour_i,
                            size=self.neighbour_num,
                            replace=False))

            for label_i in batch_label_list:
                label_i_indexes = self.label_dict[label_i]
                if self.sample_per_id <= len(label_i_indexes):
                    batch_index.extend(
                        np.random.choice(
                            label_i_indexes,
                            size=self.sample_per_id,
                            replace=False))
                else:
                    batch_index.extend(
                        np.random.choice(
                            label_i_indexes,
                            size=self.sample_per_id,
                            replace=True))
            if not self.drop_last or len(batch_index) == self.batch_size:
                yield batch_index
