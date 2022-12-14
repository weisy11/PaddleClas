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
import paddle
import paddle.distributed as dist

from ppcls.engine.evaluation.retrieval import cal_feature, gather_dist_tensors
from ppcls.utils import logger


class SampleGraphBuilder:
    def __init__(self, engine, topk=10, sim_upper_bound=0.99):
        self.engine = engine
        self.topk = topk
        self.sim_upper_bound = sim_upper_bound

    def build_sample_graph(self):
        all_feats, all_labels, _ = cal_feature(self.engine, "graph_sampler")
        sim_block_size = self.engine.config["Global"].get("sim_block_size", 64)
        # nbr short for neighbour
        nbr_map = {}
        sections = [sim_block_size] * (len(all_feats) // sim_block_size)
        if len(all_feats) % sim_block_size:
            sections.append(len(all_feats) % sim_block_size)

        feats_blocks = paddle.split(all_feats, num_or_sections=sections)
        label_blocks = paddle.split(all_labels, num_or_sections=sections)
        if dist.get_world_size() > 1:
            r = dist.get_rank()
            n = dist.get_world_size()
            if len(feats_blocks) % n:
                expand = 1
            else:
                expand = 0
            feats_blocks = feats_blocks[r * len(feats_blocks) // n:(r + 1) *
                                        len(feats_blocks) // n]
            label_blocks = label_blocks[r * len(label_blocks) // n:(r + 1) *
                                        len(label_blocks) // n]
            if expand:
                feats_blocks.append(feats_blocks[0])
                label_blocks.append(-1)

        for i, feats_i in enumerate(feats_blocks):

            if i % self.engine.config["Global"]["print_batch_step"] == 0:
                logger.info(
                    f"build sample graph process: [{i}/{len(feats_blocks)}]")
            sim_matrix = paddle.matmul(feats_i, all_feats, transpose_y=True)
            sim_matrix *= (sim_matrix <= self.sim_upper_bound)
            nbr_idx_list = paddle.argsort(
                sim_matrix, axis=1, descending=True)[:, :self.topk]
            center_labels = paddle.concat(
                [center_i for center_i in label_blocks[i]])
            if dist.get_world_size() > 1:
                nbr_idx_list = gather_dist_tensors(nbr_idx_list)
                center_labels = gather_dist_tensors(center_labels)

            for j, nbx_idx_j in enumerate(nbr_idx_list):
                center_i = center_labels[j]
                if center_i == -1:
                    continue
                topk_labels = [int(all_labels[k]) for k in nbx_idx_j]
                nbr_map[int(center_i)] = topk_labels

        self.engine.train_dataloader.set_neighbour_map(nbr_map)
