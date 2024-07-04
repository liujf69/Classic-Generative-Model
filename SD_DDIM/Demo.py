import torch
import pytorch_lightning as pl

import numpy as np
from tqdm import tqdm
from functools import partial

# From https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/util.py
def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose = True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps] # 由于alphacums来自DDPM，所以本质上还是调用了DDPM的alphas_cumprod，即[0.9983, 0.9804, ..., 0.0058]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist()) # 构成alphas_prev的方法是保留前49个alphas，同时在最前面添加DDPM的alphas_cumprod[0], 即[0.9991]

    # according the the formula provided in https://arxiv.org/abs/2010.02502 论文中的公式16
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev

# From https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/util.py
# 获取 ddim 的timesteps
def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose = True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps # 1000 // 50 = 20
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c))) # 间隔c取样
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1 # 每个数值加1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out # [1, 21, 41, ..., 981]

# From https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/util.py
def noise_like(shape, device, repeat = False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

# From https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/util.py
def make_beta_schedule(schedule, n_timestep, linear_start = 1e-4, linear_end = 2e-2, cosine_s = 8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype = torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype = torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype = torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

# origin from https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py, modified by ljf
class DDPM(pl.LightningModule):
    def __init__(self, given_betas = None, beta_schedule = "linear", timesteps = 1000, linear_start = 0.00085, linear_end = 0.012, cosine_s = 8e-3):
        super().__init__()
        self.v_posterior = 0.0
        self.parameterization = "eps"
        self.register_schedule(given_betas = given_betas, beta_schedule = beta_schedule, timesteps = timesteps,
                        linear_start = linear_start, linear_end = linear_end, cosine_s = cosine_s)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):

        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                    cosine_s=cosine_s) # 计算 betas [0.00085, 0.0008547, ..., 0.012] # total 1000
        alphas = 1. - betas # 根据betas计算alphas [0.99915, 0.9991453, ..., 0.988] # total 1000
        alphas_cumprod = np.cumprod(alphas, axis=0) # 计算alphas_cumprod [0.99915, 0.99915*0.9991453, ..., ..*0.988] # 与本身及前面的数进行相乘
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1]) # 计算alphas_cumprod_prev [1, 0.99915, 0.99915*0.9991453, ...] # 保留前999位

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    # 模拟 UNet 预测
    def apply_model(self, x_noisy, t, cond, return_ids=False):
        return torch.rand(x_noisy.shape) # 随机返回一个latent 预测

# Origin from https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddim.py, modified by ljf
class DDIMSampler(object):
    def __init__(self, model, schedule = "linear", **kwargs):
        super().__init__()
        self.model = model # DDPM的model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize = "uniform", ddim_eta = 0., verbose = True):
        # 获取ddim的timesteps [1, 21, 41, ..., 981]
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method = ddim_discretize, num_ddim_timesteps = ddim_num_steps,
                                                  num_ddpm_timesteps = self.ddpm_num_timesteps, verbose = verbose)
        
        alphas_cumprod = self.model.alphas_cumprod # 使用ddpm的alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device) # lambda表达式，对每一个输入实现相同的操作

        self.register_buffer('betas', to_torch(self.model.betas)) # 使用ddpm的betas
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod)) # 使用ddpm的alphas_cumprod
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev)) #使用ddpm的alphas_cumprod_prev

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums = alphas_cumprod.cpu(),
                                                                                   ddim_timesteps = self.ddim_timesteps,
                                                                                   eta = ddim_eta,verbose = verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self, S, batch_size, shape, conditioning = None, callback = None,
               img_callback = None, quantize_x0 = False, eta = 0., mask = None, x0 = None,
               temperature = 1., noise_dropout = 0., score_corrector = None, corrector_kwargs = None,
               verbose = True, x_T = None, log_every_t = 100, unconditional_guidance_scale = 1.,
               unconditional_conditioning = None
    ):
        self.make_schedule(ddim_num_steps = S, ddim_eta = eta, verbose = verbose) # 注册各个参数
        # sampling
        C, H, W = shape # [4, 64, 64]
        size = (batch_size, C, H, W) # [3, 4, 64, 64]
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback = callback,
                                                    img_callback = img_callback,
                                                    quantize_denoised = quantize_x0,
                                                    mask = mask, x0 = x0,
                                                    ddim_use_original_steps = False,
                                                    noise_dropout = noise_dropout,
                                                    temperature = temperature,
                                                    score_corrector = score_corrector,
                                                    corrector_kwargs = corrector_kwargs,
                                                    x_T = x_T,
                                                    log_every_t = log_every_t,
                                                    unconditional_guidance_scale = unconditional_guidance_scale,
                                                    unconditional_conditioning = unconditional_conditioning,
        )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T = None, ddim_use_original_steps = False,
                      callback = None, timesteps = None, quantize_denoised = False,
                      mask = None, x0 = None, img_callback = None, log_every_t = 100,
                      temperature = 1., noise_dropout = 0., score_corrector = None, corrector_kwargs = None,
                      unconditional_guidance_scale = 1., unconditional_conditioning = None):
        device = self.model.betas.device
        b = shape[0] # batchsize
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        timesteps = self.ddim_timesteps
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = np.flip(timesteps) 
        total_steps = timesteps.shape[0] # 50
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator): # 981, 961, ..., 1
            index = total_steps - i - 1 
            ts = torch.full((b,), step, device=device, dtype=torch.long) # [981, 981, 981], [961, 961, 961], ...
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs # 更新img
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2) # [3, 4, 64, 64] -> [6, 4, 64, 64]
            t_in = torch.cat([t] * 2) # [3] -> [6]
            c_in = torch.cat([unconditional_conditioning, c]) # [3, 77, 768] -> [6, 77, 768]
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2) # using Unet
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) # free guidance

        # 使用ddpm的参数或者make_ddim_sampling_parameters()函数生成的参数，这里默认使用了后者
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device = device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device = device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device = device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device = device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt() # 论文https://arxiv.org/pdf/2010.02502中公式(12)的第一项

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t # 论文https://arxiv.org/pdf/2010.02502中公式(12)的第二项 
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature # 论文https://arxiv.org/pdf/2010.02502中公式(12)的第三项 # 由于输入的eta为0，因此sigma_t为0，因此本式的结果为0

        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise # 构成论文https://arxiv.org/pdf/2010.02502中的公式(12)，即根据x_t得到x_(t-1)
        return x_prev, pred_x0 
    
if __name__ == "__main__":

    model = DDPM() # 初始化DDPM model
    sampler = DDIMSampler(model)

    # 模拟FrozenCLIPEmbedder的输出
    batchsize = 3
    c = torch.rand(batchsize, 77, 768) # 模拟有prompt时的embedding
    uc = torch.rand(batchsize, 77, 768) # 模拟无prompt时的embedding

    # 使用ddim进行去噪
    shape = [4, 64, 64]
    scale = 7.5 # unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
    ddim_eta = 0.0 # ddim eta (eta=0.0 corresponds to deterministic sampling
    samples_ddim, _ = sampler.sample(S = 50, # 采样50步
                                    conditioning = c, # 条件embedding
                                    batch_size = batchsize,
                                    shape = shape,
                                    verbose = False,
                                    unconditional_guidance_scale = scale,
                                    unconditional_conditioning = uc, # 无条件embedding
                                    eta = ddim_eta,
                                    x_T = None)
    
    assert samples_ddim.shape[0] == batchsize
    assert list(samples_ddim[0].shape) == shape
    print("samples_ddim.shape: ", samples_ddim.shape)
    assert samples_ddim.shape[0] == batchsize
    assert list(samples_ddim.shape[1:]) == shape
    print("All Done!")