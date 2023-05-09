"""
Author: heyuwei he20010515@163.com
Date: 2023-05-09 19:31:37
LastEditors: heyuwei he20010515@163.com
LastEditTime: 2023-05-09 19:31:45
FilePath: /DiscreteKV/main.py
Description: develop-main py script
"""
import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_dataset():
    total_classes = 8
    thetas = np.linspace(0, 2 * np.pi, total_classes)
    r = 0.1
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
    return total_point, total_label


def main():
    x, y = generate_dataset()
    for class_ in set(y):
        plt.scatter(x[y == class_][:, 0], x[y == class_][:, 1], c="r")
    plt.show()


if __name__ == "__main__":
    main()
