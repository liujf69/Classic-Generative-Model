# 下载数据
从[Kaggle](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)中下载训练数据集（约297MB），并放到下面的文件夹中
```mkdir ./data/celebA/```
# 训练命令
使用配置4和显卡0进行训练：```python main.py -c 3 -d 0```

# 训练过程
## 第一阶段
训练VQVAE模型
## 第二阶段
测试VQVAE模型的重建能力
## 第三阶段
训练PixelCNN模型
## 第四阶段
测试基于PixelCNN和VQVAE模型的生成能力

# 参考
[VQVAE PyTorch 实现教程](https://zhuanlan.zhihu.com/p/640000410)
