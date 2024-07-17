# Simple_Classifer_Guide
主要参考[guided-diffusion](https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py#L54)的核心代码进行demo的实现。

# 核心代码
1. 通过的分类器classifier计算输入图片的类别概率。
2. 根据输入的类别标签y选取对应的类别概率。
3. 根据类别概率计算输入的导数，得到对应的梯度，核心函数如下：
```python
# 核心函数
def cond_fn(x, t, y = None, classifier = None): 
    # x.shape: [batch_size, 3, 64, 64]
    # t.shape: [batch_size] 
    # y.shape: [batch_size]
    assert y is not None
    with th.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t) # [batch_size, 1000] # classifier本质上是EncoderUNetModel
        log_probs = F.log_softmax(logits, dim = -1) # [batch_size, 1000]
        selected = log_probs[range(len(logits)), y.view(-1)] # [batch_size]
        classifier_scale = 1.0 # set by ljf
        return th.autograd.grad(selected.sum(), x_in)[0] * classifier_scale # th.autograd.grad(y, x)计算y对x的导数
        # return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale # [batch_size, 3, 64, 64]
```

# 梯度添加
1. 根据上面计算得到的梯度来更新输入的均值。
```python
def condition_mean(cond_fn, p_mean_var, x, t, model_kwargs = None, classifier = None):
    # gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs) # origin version
    gradient = cond_fn(x, t, y = model_kwargs, classifier = classifier) # [batch_size, 3, 64, 64] set by ljf 
    # 根据返回的梯度更新均值
    new_mean = (
        p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float() # [batch_size, 3, 64, 64]
    )
    return new_mean
```
2. 根据更新的均值得到每一个 $x_t$ 对应的 $x_{t-1}$。
```python
sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise # [batch_size, 3, 64, 64]
```

# Run
```
python demo1.py
```
