# Former Models for Long-Term Series Forecasting (LTSF)

简体中文 | [English](README_en.md)

本项目在幻方萤火超算集群上用 PyTorch 实现了 [*Informer*](https://github.com/zhouhaoyi/Informer2020) 和 [*Autoformer*](https://github.com/thuml/Autoformer) 两个模型的**分布式训练版本**，它们是近年来采用 *transformer* 系列方法进行长时间序列预测的代表模型之一。
+ [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17325)
+ [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting (NeurIPS 2021)](https://arxiv.org/abs/2106.13008)

![Informer](./img/informer.png)


## Requirements

- [hfai](https://doc.hfai.high-flyer.cn/index.html)
- torch >=1.8


## Training
原始数据来自 [Autoformer开源仓库](https://github.com/thuml/Autoformer) ，整理进 `hfai.datasets` 数据集仓库中，包括：`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `exchange_rate`, `electricity`, `national_illness`, `traffic`。 使用参考[hfai开发文档](#)。

1. 训练 informer

   提交任务至萤火集群
   ```shell
    hfai python train.py --ds ETTh1 --model informer -- -n 1 -p 30
   ```
   本地运行：
   ```shell
    python train.py --ds ETTh1 --model informer
   ```

2. 训练 Autoformer

   提交任务至萤火集群
   ```shell
    hfai python train.py --ds ETTh1 --model autoformer -- -n 1 -p 30
   ```
   本地运行：
   ```shell
    python train.py --ds ETTh1 --model autoformer
   ```


## References
+ [Informer](https://github.com/zhouhaoyi/Informer2020)
+ [Autoformer](https://github.com/thuml/Autoformer)


## Citation

```bibtex
@inproceedings{haoyietal-informer-2021,
  author    = {Haoyi Zhou and
               Shanghang Zhang and
               Jieqi Peng and
               Shuai Zhang and
               Jianxin Li and
               Hui Xiong and
               Wancai Zhang},
  title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021, Virtual Conference},
  volume    = {35},
  number    = {12},
  pages     = {11106--11115},
  publisher = {{AAAI} Press},
  year      = {2021},
}
```

```bibtex
@inproceedings{wu2021autoformer,
  title={Autoformer: Decomposition Transformers with {Auto-Correlation} for Long-Term Series Forecasting},
  author={Haixu Wu and Jiehui Xu and Jianmin Wang and Mingsheng Long},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
