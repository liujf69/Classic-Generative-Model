# Cond_diffusion_Free_guide
基于Mnist和Classifier-Free Diffusion Guidance的Conditional Diffusion Demo

# Unet施加条件
模型基于timestep embedding和label embedding来施加条件，具体公式如下：
<p align = "center">
$a_{L+1} = c_e a_L + t_e.$
</p>

对应的代码如下：  
```python
        # embed context, time step 将label和time step编码成embedding
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1) # [batchsize, n_feat*2, 1, 1]
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1) # [batchsize, n_feat*2, 1, 1]
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1) # [batchsize, n_feat, 1, 1]
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1) # [batchsize, n_feat, 1, 1]

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec) # [batchsize, 2*n_feat, 7, 7]
        # 条件驱动, 基于Classifier-Free Diffusion Guidance
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings # 对图片特征添加条件 # [batchsize, n_feat, 14, 14]
        up3 = self.up2(cemb2 * up2 + temb2, down1) # 对图片特征添加条件 # [batchsize, n_feat, 28, 28]
        out = self.out(torch.cat((up3, x), 1)) # [batchsize, 1, 28, 28]
```

# guide weight引导样本生成
模型生成样本时，通过guide weight(scale)来调整无条件生成和有条件生成的结果，具体公式如下：
<p align = "center">
$\hat{\epsilon}_{t} = (1+w)\psi(z_t, c) - w \psi(z_t).$
</p>
其中 $\psi(z_t)$ 是无条件生成的结果，$\psi(z_t, c)$ 是有条件生成的结果。

对应的代码如下：
```python
            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask) # [2*num_sample, 1, 28, 28]
            eps1 = eps[:n_sample] # 有类别指导的预测噪音 # [num_sample, 1, 28, 28]
            eps2 = eps[n_sample:] # 没有类别指导的预测噪音 # [num_sample, 1, 28, 28]
            eps = (1 + guide_w) * eps1 - guide_w * eps2 # 根据指导概率 # [0.0, 0.5, 2.0]
            x_i = x_i[:n_sample]
            # 基于DDPM公式进行去噪
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
```

# 内容强调
Classifier-Free Diffusion Guidance 需要让模型在训练中学会**有条件**和**无条件**输入的生成能力。  
具体在代码中是通过随机**dropout**类别c来实现：
```python
# 将label转换为one hot vector
c = nn.functional.one_hot(c, num_classes = self.n_classes).type(torch.float) # [batchsize, 10]

# mask out context if context_mask == 1
context_mask = context_mask[:, None] # context_mask为1的将会被dropout
context_mask = context_mask.repeat(1, self.n_classes)
context_mask = (-1 * (1 - context_mask)) # need to flip 0 <-> 1 # 原来context_mask为1的变为0
c = c * context_mask # drop out 原context_mask为1的label # [batchsize, 10] # 由于上一步变为了0，因此需要dropout的vector变为全0，形成了无条件输入的情况。
```
上述demo是类别作为条件输入，对于文本条件输入的情况，只需要在加载数据集时**随机将文本条件置为空字符串**，即可形成无条件输入的情况，从而让模型在训练中接收**无条件输入**和**有条件输入**。

# Thanks
Our project is based on the [Conditional_Diffusion_MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST)
