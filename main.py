"""
Author: heyuwei he20010515@163.com
Date: 2023-05-09 19:31:37
LastEditors: heyuwei he20010515@163.com
LastEditTime: 2023-05-09 22:49:21
FilePath: /DiscreteKV/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from DiscreteKV import DiscreteKV
from datasets import Circle


def main():
    x, y = Circle.get_tensor()
    net = nn.Sequential(
        DiscreteKV(
            input_channel=2,
            codebook_input_channel=4,
            output_channel=20,
            codebook_num=4,
            kv_pairs_num=20,
        ),
        nn.Linear(80, 8),
    )
    Circle.show_net_status(net)
    plt.show()
    x = torch.Tensor(x)
    y = torch.Tensor(y).long()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    for _ in range(100):
        net.train()
        output = net(x)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()
        pred = torch.argmax(output, dim=-1)
        acc = torch.sum(pred == y) / len(y)
        print("loss\t", loss.item(), "\tacc\t", acc.item())

        opt.step()

    Circle.show_net_status(net)
    plt.show()


def main2():
    x, y = Circle.get_tensor()
    net = nn.Sequential(
        DiscreteKV(
            input_channel=2,
            codebook_input_channel=4,
            output_channel=64,
            codebook_num=1,
            kv_pairs_num=400,
        ),
        nn.Linear(64, 8),
    )
    opt = torch.optim.SGD(net.parameters(), lr=0.001)
    y = y.long()
    Circle.show_net_status(net)
    plt.show()
    trained_classes = []
    for classes in range(8):
        train_x = x[y == classes]
        train_label = y[y == classes]
        for _ in range(100):
            net.train()
            output = net(train_x)
            loss = nn.functional.cross_entropy(output, train_label)
            loss.backward()
            print(loss.item())
            opt.step()
        trained_classes.append(classes)
        Circle.show_net_status(net, hightlight_classes=trained_classes)
        plt.show()


if __name__ == "__main__":
    main2()
