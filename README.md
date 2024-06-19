# Classic-Generative-Model
# Introduction
1. 本项目旨在以 **Toy Demo😄😄**的方式实现经典的 **AIGC⚡⚡**生成模型，避免下载复杂的数据集和需使用高端 GPU 的问题。
2. 通过直达模型🔭🔭的核心代码（配合论文阅读更佳），学习模型的核心思想，达到快速入门学习的目的。
3. 以下均为个人入门 AIGC 所阅读和编写的部分代码（**持续更新✨✨！！！**），欢迎指出问题和 **Pull Request👯👯**（PR）。
4. 以下均为个人入门 AIGC 所阅读的优秀博客📕📕和经典论文📕📕，感谢所有作者的贡献✨✨✨✨！！

# Contents
1. A simple demo about **DDPM** in the [DDPM](https://github.com/liujf69/Classic-Generative-Model/tree/main/DDPM) folder.
2. A simple demo about **GAN** in the [GAN](https://github.com/liujf69/Classic-Generative-Model/tree/main/GAN) folder.
3. A simple demo about **VAE** in the [VAE](https://github.com/liujf69/Classic-Generative-Model/tree/main/VAE) folder.
4. A simple demo about **VQVAE** in the [VQVAE](https://github.com/liujf69/Classic-Generative-Model/tree/main/VQVAE) folder.
5. A simple demo about **PixelCNN** and **VQVAE** in the [PixelCNN_VQVAE](https://github.com/liujf69/Classic-Generative-Model/tree/main/PixelCNN_VQVAE) folder.
6. A simple demo about **Autoregressive Transformer** and **VQVAE** in the [AR_VQVAE](https://github.com/liujf69/Classic-Generative-Model/tree/main/AR_VQVAE) folder.
7. A simple demo about **Conditional Autoregressive Transformer** in the [Cond_AR_Transformer](https://github.com/liujf69/Classic-Generative-Model/tree/main/Cond_AR_Transformer) folder.
8. A simple demo about **Calculating Motion FID** in the [Cal_Motion_FID](https://github.com/liujf69/Classic-Generative-Model/tree/main/Cal_Motion_FID) folder.
9. A simple demo about **Conditional Diffusion Using Classifier-Free Diffusion Guidance** in the [Cond_diffusion_Free_guide](https://github.com/liujf69/Classic-Generative-Model/tree/main/Cond_diffusion_Free_guide) folder.
10. A simple demo about **Using Diffuser** in the [Diffuser_pipeline](https://github.com/liujf69/Classic-Generative-Model/tree/main/Diffuser_pipeline) folder.
11. A simple demo about **Using LoRA based on the PEFT** in the [PEFT_LoRA](https://github.com/liujf69/Classic-Generative-Model/tree/main/PEFT_LoRA) folder.
12. A simple demo about **MoE Model** in the [Simple_MoE](https://github.com/liujf69/Classic-Generative-Model/tree/main/Simple_MoE) folder.
13. A simple demo about **DDIM Model** in the [Simple_DDIM](https://github.com/liujf69/Classic-Generative-Model/tree/main/Simple_DDIM) folder.
14. A simple demo about **UNet Model of Stable Diffusion** in the [SD_UNet](https://github.com/liujf69/Classic-Generative-Model/tree/main/SD_UNet) folder.

# Blog Recommendation
- [DDPM = 拆楼 + 建楼](https://spaces.ac.cn/archives/9119)
- [一文带你看懂DDPM和DDIM](https://zhuanlan.zhihu.com/p/666552214)
- [通俗理解GAN](https://zhuanlan.zhihu.com/p/266677860)
- [DiT详解](https://zhuanlan.zhihu.com/p/683612528)
- [变分自编码器](https://kexue.fm/archives/5253)
- [变分自编码器](https://zhuanlan.zhihu.com/p/348498294)
- [轻松理解 VQ-VAE](https://zhuanlan.zhihu.com/p/633744455)
- [文生图模型之Stable Diffusion](https://zhuanlan.zhihu.com/p/617134893)
- [LoRA 在 Stable Diffusion 中的三种应用](https://zhuanlan.zhihu.com/p/678605372)
- [扩散模型中的v-prediction](https://zhuanlan.zhihu.com/p/678942992)
- [深入浅出完整解析ControlNet](https://zhuanlan.zhihu.com/p/660924126)
- [Stable Diffusion 原理介绍](https://zhuanlan.zhihu.com/p/613337342)
- [一文读懂DDIM凭什么可以加速DDPM的采样效率](https://zhuanlan.zhihu.com/p/627616358)
- [自回归图像生成代码阅读：VQ-GAN](https://zhuanlan.zhihu.com/p/703597240)
- [详解VQGAN（一）| 结合离散化编码与Transformer的百万像素图像生成](https://zhuanlan.zhihu.com/p/515214329)

# Paper Recommendation
## Diffusion
- Denoising Diffusion Probabilistic Models [[Paper]](https://arxiv.org/pdf/2006.11239) [[Code]](https://github.com/hojonathanho/diffusion)
- DENOISING DIFFUSION IMPLICIT MODELS [[Paper]](https://arxiv.org/pdf/2010.02502) [[Code]](https://github.com/ermongroup/ddim)
- High-Resolution Image Synthesis with Latent Diffusion Models [[Paper]](https://arxiv.org/pdf/2112.10752) [[Code]](https://github.com/CompVis/latent-diffusion) [[Code]](https://github.com/CompVis/stable-diffusion)
- Diffusion Models Beat GANs on Image Synthesis [[Paper]](https://arxiv.org/pdf/2105.05233) [[Code]](https://github.com/openai/guided-diffusion)
- Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets [[Paper]](https://arxiv.org/pdf/2311.15127v1) [[Code]](https://github.com/Stability-AI/generative-models)

## Autoregressive Model
- Autoregressive Image Generation without Vector Quantization [[Paper]](https://arxiv.org/pdf/2406.11838)

## GAN
- Generative Adversarial Nets [[Paper]](https://arxiv.org/pdf/1406.2661) [[Code]](https://github.com/goodfeli/adversarial)

## VQVAE & VAE & VQ-GAN
- Neural Discrete Representation Learning [[Paper]](https://arxiv.org/pdf/1711.00937)
- Auto-Encoding Variational Bayes [[Paper]](https://arxiv.org/pdf/1312.6114)
- CV-VAE: A Compatible Video VAE for Latent Generative Video Models [[Paper]](https://arxiv.org/pdf/2405.20279) [[Code]](https://github.com/AILab-CVC/CV-VAE?tab=readme-ov-file)
- Taming Transformers for High-Resolution Image Synthesis [[Paper]](https://arxiv.org/pdf/2012.09841) [[Code]](https://github.com/CompVis/taming-transformers/tree/master)

## DIT
- Scalable Diffusion Models with Transformers [[Paper]](https://arxiv.org/pdf/2212.09748) [[Code]](https://github.com/facebookresearch/DiT)
- Hunyuan-DiT : A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding [[Paper]](https://arxiv.org/pdf/2405.08748) [[Code]](https://github.com/Tencent/HunyuanDiT)

# Project Recommendation
- stable-diffusion [[Code]](https://github.com/CompVis/stable-diffusion)
- AnimateDiff [[Code]](https://github.com/guoyww/AnimateDiff)
- Open-Sora [[Code]](https://github.com/hpcaitech/Open-Sora)
- VAR [[Code]](https://github.com/FoundationVision/VAR)
- IC-Light [[Code]](https://github.com/lllyasviel/IC-Light)
  
# Thanks
Thanks to all the authors of the above blogs and papers!

