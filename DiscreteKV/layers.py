"""
Author: heyuwei he20010515@163.com
Date: 2023-05-09 19:11:24
LastEditors: heyuwei he20010515@163.com
LastEditTime: 2023-05-09 19:14:03
FilePath: /DiscreteKV/DiscreteKV/layers.py
Description: layers
"""
import torch
from torch import nn
import torch.random
import numpy as np
import matplotlib.pyplot as plt


class CodeBook(nn.Module):
    def __init__(
        self, input_channel: int, output_channel: int, kvpairs_num: int = 8
    ) -> None:
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.keys = (torch.rand([kvpairs_num, input_channel]) - 0.5) * 3
        self.values = torch.nn.Parameter(torch.randn([kvpairs_num, output_channel]))
        self.kvpairs_num = kvpairs_num

    def forward(self, x: torch.Tensor):
        """
        input:
            x:torch.Tensor   x.shape == [batch_size,input_channel]
        output:
            y:torch.Tensor   y.shape == [batch_size,output_channel]
        """
        batchsz = x.shape[0]
        view_x = x.view([batchsz, 1, self.input_channel])
        view_key = self.keys.view([1, self.kvpairs_num, self.input_channel])
        distances = (view_x - view_key) ** 2  # broading cast
        distances = torch.sum(distances, dim=-1)
        distances = torch.sqrt(distances)
        min_index = torch.argmin(distances, -1)
        return self.values[min_index]


class DiscreteKV(nn.Module):
    def __init__(
        self,
        input_channel: int,  # size of input feature
        codebook_input_channel: int,  #
        output_channel: int,
        codebook_num: int,
        kv_pairs_num: int,
    ) -> None:
        super().__init__()
        self.codeblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_channel, codebook_input_channel),
                    CodeBook(codebook_input_channel, output_channel, kv_pairs_num),
                )
                for _ in range(codebook_num)
            ]
        )

    def forward(self, x: torch.Tensor):
        ys = [m(x) for m in self.codeblocks]
        ys = torch.concat(
            ys, dim=-1
        )  # ys.shape [batch_size,codebook_num * output_channels]
        return ys
