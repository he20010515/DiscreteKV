"""
Author: heyuwei he20010515@163.com
Date: 2023-05-09 22:29:14
LastEditors: heyuwei he20010515@163.com
LastEditTime: 2023-05-09 22:30:53
FilePath: /DiscreteKV/datasets/Circle.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm

# colorlist = [f"C{i}" for i in range(8)]
# plt_cmap = ListedColormap(colorlist)
plt_cmap = matplotlib.cm.get_cmap("tab10")
circle_map = [2, 3, 4, 5, 6, 7, 0, 1]


class Circle:
    @staticmethod
    def get_tensor():
        total_classes = 8
        thetas = np.linspace(0, 2 * np.pi, total_classes + 1)
        r = 0.1 / 4
        centers = np.cos(thetas), np.sin(thetas)
        total_point = []
        total_label = []
        for xc, yc, label in zip(*centers, range(total_classes)):
            points = np.random.normal(0, r, size=[100, 2])
            points = points + np.array([xc, yc])
            total_point.append(points)
            total_label += [label] * 100
        total_point = np.concatenate(total_point)
        total_label = np.array(total_label)
        return torch.Tensor(total_point), torch.Tensor(total_label)

    @staticmethod
    def show_net_status(
        net: torch.nn.Module,
        ax: plt.Axes = None,
        xrange=(-2, 2),
        yrange=(-2, 2),
        points=200,
        hightlight_classes=[],
    ):
        net.eval()
        if ax is None:
            ax = plt.subplot()
            ax.set_aspect(1)
        x, y = Circle.get_tensor()
        X = torch.linspace(*xrange, points)
        Y = torch.linspace(*yrange, points)
        X, Y = torch.meshgrid(X, Y)
        samples = torch.stack([X.reshape([-1]), Y.reshape([-1])]).T
        label = torch.argmax(net(samples), dim=-1)
        Z = label.reshape_as(X)
        Z = Z.numpy()
        newZ = np.zeros([*Z.shape, 4])
        newZ[:, :, :] = plt_cmap(Z)
        # ax.imshow(
        #     newZ,
        #     interpolation="nearest",
        #     extent=[*xrange, *yrange],
        #     alpha=0.8,
        # )
        ax.scatter(
            X.flatten(),
            Y.flatten(),
            color=[c for c in newZ.reshape(-1, 4)],
            edgecolors=None,
        )
        for class_ in range(8):
            if class_ in hightlight_classes:
                ax.scatter(
                    x[y == class_][:, 0],
                    x[y == class_][:, 1],
                    color=plt_cmap(class_),
                    marker="D",
                    edgecolors="white",
                )
            else:
                ax.scatter(
                    x[y == class_][:, 0],
                    x[y == class_][:, 1],
                    color=plt_cmap(class_),
                    edgecolors="black",
                )
