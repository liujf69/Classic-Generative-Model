# Classic-Generative-Model
# Introduction
1. 本项目旨在以 **Toy Demo😄😄**的方式实现经典的 **AIGC⚡⚡**生成模型，避免下载复杂的数据集和需使用高端 GPU 的问题。
2. 通过直达模型🔭🔭的核心代码（配合论文阅读更佳），学习模型的核心思想，达到快速学习的目的。
3. 以下均为个人学习 AIGC 所阅读和编写的部分代码（**持续更新✨✨！！！**），欢迎指出问题和 **Pull Request👯👯**（PR）。
4. 以下均为个人学习 AIGC 所阅读的优秀博客📕📕和经典论文📕📕，感谢所有作者的贡献✨✨✨✨！！

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
15. A simple demo about **U-ViT Demo** in the [Simple_U-DiT](https://github.com/liujf69/Classic-Generative-Model/tree/main/Simple_U-ViT) folder.
16. A simple demo about **DDIM of Stable Diffusion** in the [SD_DDIM](https://github.com/liujf69/Classic-Generative-Model/tree/main/SD_DDIM) folder.
17. A simple demo about **Inpainting of Stable Diffusion** in the [SD_Inpainting](https://github.com/liujf69/Classic-Generative-Model/tree/main/SD_Inpainting) folder.
18. A simple demo about **Classifer Guidance** in the [Simple_Classifer_Guide](https://github.com/liujf69/Classic-Generative-Model/tree/main/Simple_Classifer_Guide) folder.
19. A simple demo about **Diffusion Transformers (DiTs)** in the [Simple_DIT](https://github.com/liujf69/Classic-Generative-Model/tree/main/Simple_DIT) folder.

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
- [保持ID的人像生成技术介绍：IP-Adaptor,PhotoMaker,InstantID](https://zhuanlan.zhihu.com/p/678613724)
- [Classifier Guidance 和 Classifier Free Guidance](https://zhuanlan.zhihu.com/p/660518657)
- [SD和Sora们背后的关键技术！一文搞懂所有 VAE 模型（4个AE+12个VAE原理汇总）](https://mp.weixin.qq.com/s/HzwkwjfItLEE1nmkd1-THw)
- [AIGC专栏4——Stable Diffusion原理解析-inpaint修复图片为例](https://blog.csdn.net/weixin_44791964/article/details/131997973)
- [AIGC-Stable Diffusion之Inpaint(图像修复)](https://zhuanlan.zhihu.com/p/681250295)
- [[论文理解] Classifier-free diffusion guidance](https://sunlin-ai.github.io/2022/06/01/Classifier-Free-Diffusion.html)
- [Stable Video Diffusion 结构浅析与论文速览](https://zhuanlan.zhihu.com/p/693750402)
- [Stable Video Diffusion 源码解读 (Diffusers 版)](https://zhuanlan.zhihu.com/p/701223363)
- [AIGC-Stable Diffusion之VAE](https://zhuanlan.zhihu.com/p/679772356)
- [深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识](https://zhuanlan.zhihu.com/p/643420260)
- [LoRA微调中是怎么冻结和加入AB矩阵的](https://zhuanlan.zhihu.com/p/688863868)
- [stable diffusion常用的LoRA、Dreambooth、Hypernetworks四大模型差异详解](https://zhuanlan.zhihu.com/p/694921070)
- [LoRA vs Dreambooth vs Textural Inversion vs Hypernetworks](https://zhuanlan.zhihu.com/p/612992813)
- [stable diffusion——Dreambooth原理与实践](https://zhuanlan.zhihu.com/p/620577688)
- [dreambooth原理](https://zhuanlan.zhihu.com/p/659774932)
  
# Paper Recommendation
## News and Highlights
- Rich Human Feedback for Text-to-Image Generation (CVPR 2024 best paper) [[Paper]](https://arxiv.org/pdf/2312.10240) [[Code]](https://github.com/google-research/google-research/tree/master/richhf_18k)
- Generative Image Dynamics (CVPR 2024 best paper) [[Paper]](https://generative-dynamics.github.io/static/pdfs/GenerativeImageDynamics.pdf) [[Project]](https://generative-dynamics.github.io/)
- Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (ICML 2024 best paper) [[Paper]](https://arxiv.org/pdf/2403.03206) [[Project]](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
- VideoPoet: A Large Language Model for Zero-Shot Video Generation (ICML 2024 best paper) [[Paper]](https://arxiv.org/pdf/2312.14125) [[Project]](https://sites.research.google/videopoet/)
- Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (ICML 2024 best paper) [[Paper]](https://arxiv.org/pdf/2310.16834)
## Diffusion
- Denoising Diffusion Probabilistic Models [[Paper]](https://arxiv.org/pdf/2006.11239) [[Code]](https://github.com/hojonathanho/diffusion)
- DENOISING DIFFUSION IMPLICIT MODELS [[Paper]](https://arxiv.org/pdf/2010.02502) [[Code]](https://github.com/ermongroup/ddim)
- High-Resolution Image Synthesis with Latent Diffusion Models [[Paper]](https://arxiv.org/pdf/2112.10752) [[Code]](https://github.com/CompVis/latent-diffusion) [[Code]](https://github.com/CompVis/stable-diffusion)
- Diffusion Models Beat GANs on Image Synthesis [[Paper]](https://arxiv.org/pdf/2105.05233) [[Code]](https://github.com/openai/guided-diffusion)
- Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets [[Paper]](https://arxiv.org/pdf/2311.15127v1) [[Code]](https://github.com/Stability-AI/generative-models)
- Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models [[Paper]](https://arxiv.org/pdf/2304.08818)
- Elucidating the Design Space of Diffusion-Based Generative Models [[Paper]](https://arxiv.org/pdf/2206.00364)

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
- All are Worth Words: A ViT Backbone for Diffusion Models [[Paper]](https://arxiv.org/pdf/2209.12152) [[Code]](https://github.com/baofff/U-ViT)

## AI-generated image detection
- A Single Simple Patch is All You Need for AI-generated Image Detection [[Paper]](https://arxiv.org/pdf/2402.01123) [[Code]](https://github.com/bcmi/SSP-AI-Generated-Image-Detection) [[Blog]](https://zhuanlan.zhihu.com/p/681554797)
- PatchCraft: Exploring Texture Patch for Efficient AI-generated Image Detection [[Paper]](https://arxiv.org/pdf/2311.12397) [[Code]](https://github.com/hridayK/Detection-of-AI-generated-images) [[Blog]](https://zhuanlan.zhihu.com/p/685317225)
- GenDet: Towards Good Generalizations for AI-Generated Image Detection [[Paper]](https://arxiv.org/pdf/2312.08880) [[Code]](https://github.com/GenImage-Dataset/GenImage)

# Project Recommendation
- stable-diffusion [[Code]](https://github.com/CompVis/stable-diffusion)
- ControlNet [[Code]](https://github.com/lllyasviel/ControlNet)
- AnimateDiff [[Code]](https://github.com/guoyww/AnimateDiff)
- Open-Sora [[Code]](https://github.com/hpcaitech/Open-Sora)
- Open-Sora-Plan [[Code]](https://github.com/PKU-YuanGroup/Open-Sora-Plan)
- VAR [[Code]](https://github.com/FoundationVision/VAR)
- IC-Light [[Code]](https://github.com/lllyasviel/IC-Light)
- IP-Adapter [[Code]](https://github.com/tencent-ailab/IP-Adapter)
- LivePortrait [[Code]](https://github.com/KwaiVGI/LivePortrait)
- zero123 [[Code]](https://github.com/cvlab-columbia/zero123/tree/main)
- VideoTetris [[Code]](https://github.com/YangLing0818/VideoTetris)
  
# Thanks
Thanks to all the authors of the above blogs and papers!!

