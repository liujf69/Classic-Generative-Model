import torch
import numpy as np

if __name__ == "__main__":
    B = 100
    C = 3
    T = 300
    V = 25
    src_data = torch.rand(B, C, T, V)
    gen_data = torch.rand(B, C, T, V)

    np.save("./source_path/src_data.npy", src_data.numpy())
    np.save("./generate_path/gen_data.npy", gen_data.numpy())

    print("All done!")