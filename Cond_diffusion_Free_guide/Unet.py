import torch
import torch.nn as nn

# 残差连接卷积网络
class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes = 10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        """
        Args:
            x: 加噪后的图片, shape: [batchsize, 1, 28, 28]
            c: 图片的类别, shape: [batchsize]
            t: timestep, shape: [batchsize]
            context_mask: 掩码矩阵, 用于dropout图片的类别
        """

        x = self.init_conv(x) # [batchsize, 1, 28, 28] -> [batchsize, n_feat, 28, 28]
        down1 = self.down1(x) # [batchsize, n_feat, 14, 14]
        down2 = self.down2(down1) # [batchsize, 2*n_feat, 7, 7]
        hiddenvec = self.to_vec(down2) # [batchsize, 2*n_feat, 1, 1]

        # 将label转换为one hot vector
        c = nn.functional.one_hot(c, num_classes = self.n_classes).type(torch.float) # [batchsize, 10]
        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = (-1 * (1 - context_mask)) # need to flip 0 <-> 1
        c = c * context_mask # drop out 原context_mask为1的label # [batchsize, 10]
        
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
        return out