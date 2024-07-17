import torch as th
import torch.nn.functional as F

from tools.unet import EncoderUNetModel

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

def condition_mean(cond_fn, p_mean_var, x, t, model_kwargs = None, classifier = None):
    # gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs) # origin version
    gradient = cond_fn(x, t, y = model_kwargs, classifier = classifier) # [batch_size, 3, 64, 64] set by ljf
    new_mean = (
        p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float() # [batch_size, 3, 64, 64]
    )
    return new_mean

def p_sample(x, t, cond_fn = None, model_kwargs = None, classifier = None):
        # origin version
        # out = self.p_mean_variance(
        #     model,
        #     x,
        #     t,
        #     clip_denoised=clip_denoised,
        #     denoised_fn=denoised_fn,
        #     model_kwargs=model_kwargs,
        # )

        # 这里使用random的方法模拟p_mean_variance的过程
        out = {}
        out["mean"] = th.rand(x.shape)
        out["variance"] = th.rand(x.shape)
        out["log_variance"] = th.rand(x.shape)
        out["pred_xstart"] = th.rand(x.shape)

        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = condition_mean(
                cond_fn, out, x, t, model_kwargs = model_kwargs, classifier = classifier
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise # [batch_size, 3, 64, 64]
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}


if __name__ == "__main__":
    batch_size = 2
    channel = 3
    height = 64
    width = 64
    x = th.rand(batch_size, channel, height, width) # 随机生成输入
    t = th.tensor([999]).repeat(batch_size) # 随机模拟time steps
    y = th.randint(low = 0, high = 999, size = (2,)) # 模拟类别
    classifier = EncoderUNetModel(
        image_size = 64,
        in_channels = 3,
        model_channels = 128,
        out_channels = 1000, # 1000个类别
        num_res_blocks = 3,
        attention_resolutions = [32,16,8])
    
    output = p_sample(x = x, t = t, model_kwargs = y, cond_fn = cond_fn, classifier = classifier) # 传入cond_fn函数和分类器classifier
    assert list(output['sample'].shape) == [batch_size, channel, height, width]
    print("output['sample'].shape: ", output['sample'].shape)
    print("All Done!")