import numpy as np
from tqdm import tqdm
from scipy import linalg

import torch
from torch.utils.data import Dataset
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d

class Feeder(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle = True)

    def __len__(self): 
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index]

def calculate_fid_given_paths(src_path, gen_path, batch_size, device = 'cpu', dims = 2048):
    # load model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    
    # cal data
    m1, s1 = calculate_activation_statistics(src_path, model, batch_size, dims, device)
    m2, s2 = calculate_activation_statistics(gen_path, model, batch_size, dims, device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

def calculate_activation_statistics(path, model, batch_size = 50, dims = 2048, device = 'cpu'):
    act = get_activations(path, model, batch_size, dims, device) # [len dims]
    mu = np.mean(act, axis = 0) # dims
    sigma = np.cov(act, rowvar = False) # [dims, dims]
    return mu, sigma

def get_activations(path, model, batch_size = 50, dims = 2048, device = 'cpu'):
    model.eval().to(device)

    # load dataset
    dataset = Feeder(path)
    print('len(dataset): ', len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = False,
        num_workers = 4
    )

    pred_arr = np.empty((len(dataset), dims)) # Len, dims
    start_idx = 0
    frame_len = 64
    for data in tqdm(dataloader): # get batch data
        data = data.float().to(device) # B C T V
        data = data[:, :, :frame_len, :] # B C T V
        with torch.no_grad():
            pred = model(data)[0] # B dims 1 1

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy() # B dims

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred # record pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1) # [dims]
    mu2 = np.atleast_1d(mu2) # [dims]
    sigma1 = np.atleast_2d(sigma1) # [dims, dims]
    sigma2 = np.atleast_2d(sigma2) # [dims, dims]

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False) # [dims, dims]
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; ''adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def main():
    src_path = './source_path/src_data.npy'
    gen_path = './generate_path/gen_data.npy'
    batch_size = 16
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    dims = 2048
    fid_value = calculate_fid_given_paths(src_path, gen_path, batch_size, device, dims)
    print('FID: ', fid_value)

if __name__ == '__main__':
    main()