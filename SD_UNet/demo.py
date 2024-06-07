import torch
from UNet import UNetModel

def main():
    # unet config
    image_size = 32 # unused
    in_channels = 4
    out_channels = 4
    model_channels = 320
    attention_resolutions = [ 4, 2, 1 ]
    num_res_blocks = 2
    channel_mult = [ 1, 2, 4, 4 ]
    num_heads = 8
    use_spatial_transformer = True
    transformer_depth = 1
    context_dim = 768
    use_checkpoint = True
    legacy = False

    model = UNetModel(
        image_size = image_size,
        in_channels = in_channels,
        out_channels = out_channels,
        model_channels = model_channels,
        attention_resolutions = attention_resolutions,
        num_res_blocks = num_res_blocks,
        channel_mult = channel_mult,
        num_heads = num_heads,
        use_spatial_transformer = use_spatial_transformer,
        transformer_depth = transformer_depth,
        context_dim = context_dim,
        use_checkpoint = use_checkpoint,
        legacy = legacy
    ).cuda()

    shape = [3, 4, 64, 64]
    device = "cuda"
    x = torch.randn(shape, device=device)
    t = torch.tensor([981, 981, 981]).cuda()
    c = torch.ones(3, 77, 768).cuda()
    uc = torch.zeros(3, 77, 768).cuda()

    x_in = torch.cat([x] * 2) # [3, 4, 64, 64] -> [6, 4, 64, 64]
    t_in = torch.cat([t] * 2) # [3] -> [6]
    c_in = torch.cat([uc, c]) # [3, 77, 768] -> [6, 77, 768]
    e_t_uncond, e_t = model(x_in, t_in, c_in).chunk(2) # using Unet # [3, 4, 64, 64]

    print("All Done!")

if __name__ == "__main__":
    main()