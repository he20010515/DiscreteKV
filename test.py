"""
Author: heyuwei he20010515@163.com
Date: 2023-05-09 23:54:01
LastEditors: heyuwei he20010515@163.com
LastEditTime: 2023-05-09 23:54:04
FilePath: /DiscreteKV/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# 定义分类函数
def classify_point(point):
    # TODO: 实现你的分类函数
    x, y = point
    if x > 0:
        return 1
    elif y < 0:
        return 2

    return 0


# 生成二维数组
x_min, x_max = -5, 5
y_min, y_max = -5, 5
step = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
zz = np.array(
    [
        [classify_point([x, y]) for x, y in zip(row_x, row_y)]
        for row_x, row_y in zip(xx, yy)
    ]
)

# 自定义颜色映射
colors = ["r", "g", "b"]
cmap = ListedColormap(colors)

# 绘制分类结果
plt.imshow(
    zz,
    interpolation="nearest",
    extent=(x_min, x_max, y_min, y_max),
    cmap=cmap,
    aspect="auto",
    origin="lower",
)
plt.show()
