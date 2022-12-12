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

from ppcls.engine.evaluation.retrieval import cal_feature


class SampleGraphBuilder:
    def __init__(self, engine, topk=10, sim_upper_bound=0.99):
        self.engine = engine
        self.topk = topk
        self.sim_upper_bound = sim_upper_bound

    def build_sample_graph(self):
        features, label, _ = cal_feature(self.engine, "graph_sampler")
        sim_block_size = self.engine.config["Global"].get("sim_block_size", 64)
        neighbour_map = {}
        sections = [sim_block_size] * (len(features) // sim_block_size)
        if len(features) % sim_block_size:
            sections.append(len(features) % sim_block_size)

        fea_blocks = paddle.split(features, num_or_sections=sections)
        label_blocks = paddle.split(label, num_or_sections=sections)
        for block_idx, block_fea in enumerate(fea_blocks):
            similarity_matrix = paddle.matmul(
                block_fea, features, transpose_y=True)
            similarity_matrix *= (similarity_matrix <= self.sim_upper_bound)
            neighbour_index_list = paddle.argsort(
                similarity_matrix, axis=1, descending=True)
            for i, args_i in enumerate(neighbour_index_list):
                label_i = int(label_blocks[block_idx][i])
                topk_labels = [int(label[i]) for i in args_i[0:self.topk]]
                neighbour_map[label_i] = topk_labels
        self.engine.train_dataloader.set_neighbour_map(neighbour_map)
