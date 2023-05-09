<!--
 * @Author: heyuwei he20010515@163.com
 * @Date: 2023-05-09 19:09:07
 * @LastEditors: heyuwei he20010515@163.com
 * @LastEditTime: 2023-05-09 22:29:44
 * @FilePath: /DiscreteKV/readme.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->

# Introduction
This work is following Discrete Key-Value Bottleneck

# TODO List
1. values 是怎么组合的?
2. Encoder 部分换成Transformer如何? 多模态情况下情况如何?
3. 不同层次级别的CodeBook. 让网络中包含层次信息. 或者引入类似Transformer的层级别注意力机制
4. 所有CodeBook的输出大小未必一样,类似特征金字塔.
