import fairseq

import torch
import torchaudio
import torch as th
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from torch.distributions import Categorical
from transformers import BertModel

from typing import Callable, List

from model.modules.rotary_embedding_torch import RotaryEmbedding
from model.modules.transformer_modules import DecoderLayerStack, FiLMTransformerDecoderLayer

def get_TextEncoder():
    Text_encoder = BertModel.from_pretrained('bert-base-chinese')
    # freeze
    for param in Text_encoder.parameters():
        param.requires_grad = False
    return Text_encoder

def get_AudioEncoder() -> ("Audio2LipRegressionTransformer", torchaudio.transforms.Resample):
    checkpoint_path = "./vq-wav2vec.pt"
    audio_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
    audio_model = audio_model[0]
    # freeze
    for param in audio_model.parameters():
        param.requires_grad = False
    audio_model.eval()
    audio_resampler = torchaudio.transforms.Resample(48000, 16000) # 采样
    return audio_model, audio_resampler

# dropout mask
def prob_mask_like(shape, prob, device):
    if prob == 1: # 推理时应该完全考虑audio
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else: # 训练时随机mask
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

class GuideTransformer(nn.Module):
    '''
    param: tokens: VQVAE的codebook中embedding的个数
    param: num_heads: 自回归transformer中多头注意力机制
    param: num_layers: 自回归transformer的层数
    param: dim: 自回归transformer中的特征长度
    param: ff_size: 自回归transformer中feedforward的特征长度
    param: dropout: dropout rate
    param: activation: 自回归transformer中使用的激活函数
    param: use_rotary: 是否使用旋转位置编码
    param: cond_feature_dim: 音频特征的特征长度
    param: emb_len: 随机生成embedding的个数
    param: num_audio_layers: 音频编码器的层数
    '''
    def __init__(self, tokens: int, num_heads: int = 4, num_layers: int = 4, dim: int = 512, ff_size: int = 1024, dropout: float = 0.1, 
            activation: Callable = F.gelu, use_rotary: bool = True, audio_feature_dim: int = 512, emb_len: int = 798, num_audio_layers: int = 2):
        super(GuideTransformer, self).__init__()
        self.tokens = tokens
        self.token_embedding = nn.Embedding(
            num_embeddings = tokens + 1,  # account for sequence start and end tokens
            embedding_dim = dim)
        
        # 文本和音频的模型
        self.Text_encoder = get_TextEncoder() # get and freeze Text Encoder
        self.audio_model, self.audio_resampler = get_AudioEncoder() # 获取音频编码器和采样器

        # 进一步处理音频特征
        self.pre_audio = self.get_process_audio_models(audio_feature_dim, num_audio_layers) # get process_audio_models # 预处理后进一步编码

        # 投影层
        self.audio_projection = nn.Linear(audio_feature_dim, dim)
        self.text_projection = nn.Linear(768, dim)

        # mlp处理音频特征获取隐变量
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim))
        
        # 随机生成embedding
        self.rand_null_cond_embed = nn.Parameter(torch.randn(1, emb_len, dim))
        self.rand_null_cond_hidden = nn.Parameter(torch.randn(1, dim))

        # layerNorm audio
        self.norm_cond = nn.LayerNorm(dim)

        # 旋转位置编码
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=dim)

        # autoregressive
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(dim, num_heads, dim_feedforward = ff_size, dropout = dropout,
                    activation = activation, batch_first = True, rotary = self.rotary)
            )
        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        self.final_layer = nn.Linear(dim, tokens)

    # 获取处理音频的模型
    def get_process_audio_models(self, cond_feature_dim: int, num_audio_layers: int):
        pre_layers = []
        for _ in range(num_audio_layers):
            pre_layers += self._build_single_audio_conv(cond_feature_dim)
        pre_layers += [torch.nn.Conv1d(cond_feature_dim, cond_feature_dim, kernel_size=1)]
        pre_layers = torch.nn.ModuleList(pre_layers)
        process_audio = nn.Sequential(*pre_layers)
        return process_audio
    
    # 进一步处理音频特征的单层网络
    def _build_single_audio_conv(self, c: int) -> List[nn.Module]:
        return [torch.nn.Conv1d(c, max(256, c), kernel_size=3, dilation=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Conv1d(max(256, c), max(256, c), kernel_size=3, dilation=2),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Conv1d(max(128, c), max(128, c), kernel_size=3, dilation=3),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Conv1d(max(128, c), c, kernel_size=3, dilation=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Conv1d(c, c, kernel_size=3, dilation=2),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Conv1d(c, c, kernel_size=3, dilation=3),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2)]

    # get autoregressive mask
    def get_tgt_mask(self, size: int, device: str) -> torch.tensor:
         # 自回归mask，下三角为True（包含主对角），上三角为False
        mask = torch.tril(torch.ones((size, size), device = device) == 1)
        mask = mask.float() # True -> 1, False -> 0
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
        return mask

    # 利用音频编码器和采样器encode audio
    def encode_audio(self, raw_audio: torch.Tensor) -> torch.Tensor:
        a0 = self.audio_resampler(raw_audio[:, :, 0])  # B x T # 采样
        with torch.no_grad():
            z0 = self.audio_model.feature_extractor(a0)
        return z0    
    
    def forward(self, tokens: th.Tensor, text_ids: th.Tensor, text_masks: th.Tensor, Text_tokenType: th.Tensor, 
            audio_condition: th.Tensor, cond_drop_prob: float = 0.0) -> torch.Tensor:
        ''' 
        param: tokens: [B T*residual_depth]
        param: text_ids: [B, max_len]
        param: text_masks: [B, max_len]
        param: Text_tokenType: [B, max_len]
        param: audio_condition: [B, 480000, 2]
        param: cond_drop_prob: the rate of ramdom select
        '''
        
        # process motion
        batch_size, device = tokens.shape[0], tokens.device
        target = self.token_embedding(tokens) # nn.Embedding x:[B T*residual_depth embedding_dim(64)]
        tgt_mask = self.get_tgt_mask(target.shape[1], target.device) # 获取自回归的掩码 [T*residual_depth, T*residual_depth]

        # process audio
        audio_embed = self.encode_audio(audio_condition) # codition: [B T*1600 C] cond_embed: [B audio_feature_dim 1998] # 编码音频获取音频特征
        audio_tokens = self.pre_audio(audio_embed).permute(0, 2, 1) # 处理音频特征 [B 1950 audio_feature_dim] # 进一步处理
        audio_tokens = self.audio_projection(audio_tokens) # linear [B 1950 audio_feature_dim->dim] # 投影层

        # 随机选取两种：随机生成的，音频提取的
        keep_mask = prob_mask_like((batch_size), 1 - cond_drop_prob, device = device) # [B]: [True, True, ..., False] True保留audio，False使用random generate

        # random sample audio_token
        rand_null_audio_embed = self.rand_null_cond_embed.to(audio_tokens.dtype)
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        audio_tokens = torch.where(keep_mask_embed, audio_tokens, rand_null_audio_embed[:, :audio_tokens.shape[1], :]) # cond_tokens: [B 1950 dim]
        mean_pooled_cond_tokens = audio_tokens.mean(dim=-2) # 取均值 [B dim]
        audio_tokens = self.norm_cond(audio_tokens) # nn.LayerNorm

        # random sample audio_hiden
        audio_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens) # 基于mlp等进一步处理获取隐变量hidden
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")
        rand_null_cond_hidden = self.rand_null_cond_hidden.to(audio_tokens.dtype)
        audio_hidden = torch.where(keep_mask_hidden, audio_hidden, rand_null_cond_hidden) # 随机选取隐变量 [B 64]

        # process text
        text_embed = self.Text_encoder(input_ids = text_ids, attention_mask = text_masks, token_type_ids = Text_tokenType).last_hidden_state
        # text_embed = self.Text_encoder(input_ids = text_ids, attention_mask = text_masks, token_type_ids = Text_tokenType).last_hidden_state[:, 0] # if ues [cls] token [B 768] should unsqueeze(1)
        text_tokens = self.text_projection(text_embed) # [B, dim]

        # autoregressive
        output = self.seqTransDecoder(target, tgt_mask = tgt_mask, audio_token = audio_tokens, audio_hidden = audio_hidden, text_token = text_tokens)
        output = self.final_layer(output) # linear
        return output
    
    def generate(self, text_ids: th.Tensor, text_masks: th.Tensor, Text_tokenType: th.Tensor, 
            audio_condition: th.Tensor, sequence_length: int, residual_depth: int = 4, batchsize: int = 1, top_p: float = 0.94, pre_tokens: th.Tensor = None) -> th.Tensor:
        with torch.no_grad():
            input_tokens = torch.zeros(batchsize, 1, dtype=th.int64).to(audio_condition.device) + self.tokens
            # 考虑当前状态前几帧的pre_tokens，做一个引导的作用
            if pre_tokens != None:
                input_tokens = torch.cat([input_tokens, pre_tokens], dim=-1)

            # 循环生成sequence_length*residual_depth个token
            for _ in range(sequence_length*residual_depth):
                curr_input_tokens = input_tokens
                logits = self.forward(tokens = curr_input_tokens, text_ids = text_ids, text_masks = text_masks, Text_tokenType = Text_tokenType, audio_condition=audio_condition)
                logits = logits[:, -1, :]  # 取最后一个时间步的输出

                # _, tokens = torch.max(logits, dim=-1) # 直接策略
                # 选择策略
                one_hot = torch.nn.functional.softmax(logits, dim=-1)
                sorted_probs, indices = torch.sort(one_hot, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                nucleus = cumulative_probs < top_p
                nucleus = torch.cat(
                    [
                        nucleus.new_ones(nucleus.shape[:-1] + (1,)),
                        nucleus[..., :-1],
                    ],
                    dim = -1,
                )
                sorted_probs[~nucleus] = 0
                sorted_probs /= sorted_probs.sum(-1, keepdim=True)
                dist = Categorical(sorted_probs)
                idx = dist.sample()
                tokens = indices.gather(-1, idx.unsqueeze(-1))
                # 将预测的token添加到输入token中，进行下一轮循环
                input_tokens = torch.cat([input_tokens, tokens], dim=-1)
            
            # 只返回生成的token
            remove_num = 1+pre_tokens.shape[0] if pre_tokens != None else 1
            tokens = input_tokens[:, remove_num:].contiguous()
            return tokens