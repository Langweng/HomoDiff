from ast import iter_child_nodes
import torch
import torch.nn as nn
from .update import *
from .extractor import *
from .corr import *
from .utils import *
from .flow_utils import *
from .homo_utils import *
from .decoder import GMA_update

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class hnet_layer(nn.Module):
    # def __init__(self, args):
    def __init__(self):
        super().__init__()
        # self.args = args
        self.fnet = BasicEncoder(output_dim=96, norm_fn='instance')
        # self.update_blocks = nn.ModuleList([CorrelationDecoder(input_dim=81, hidden_dim=64, output_dim=2, downsample=4),
        #                             CorrelationDecoder(input_dim=81, hidden_dim=64, output_dim=2, downsample=5),
        #                             CorrelationDecoder(input_dim=81, hidden_dim=64, output_dim=2, downsample=6)])
        self.update_blocks = nn.ModuleList([CorrelationDecoder(input_dim=81, hidden_dim=64, output_dim=2, downsample=6)])
        # self.downsample = self.args.downsample
        # self.iter = self.args.iter
        self.downsample = [4,2,1]
        self.iter = [1,1,1]
        self.memory = {"deltaD":[], "scale":[], "delta_ace":[], "iteration":[]}
        self.instance_experts = []

    def forward(self, img1, img2, idx, four_point_disp=None):
        image1, image2 = img1, img2

        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)

        batch_size = image1.shape[0]

        if idx == 0:
            four_point_disp = torch.zeros((batch_size, 2, 2, 2)).to(image1.device)
        # four_point_predictions = []

        downsample = self.downsample[idx]
        # idx = self.downsample.index(downsample)
        # print('fmap1.shape, idx', fmap1[idx].shape, idx)
        corr_fn = LocalCorr(fmap1[idx], fmap2[idx])
        # corr_fn_gyh = LocalCorr_gyh(fmap1[idx], fmap2[idx])
        coords0, _ = initialize_flow(image1, downsample=downsample)

        # for _ in range(self.iter[idx]):
        coords1 = disp_to_coords(four_point_disp, coords0, downsample=downsample) 
        # print('coords1', coords1)
        # print('coords1.shape', coords1.shape)
        # exit()
        corr = corr_fn(coords1, window_size=9)
        # print('corr.shape', corr.shape)
        # exit()
        # corr_gyh = corr_fn_gyh(coords1)
        # print('corr', corr)
        # print('corr_gyh', corr_gyh)
        # exit()
        upsample = nn.Upsample(scale_factor=256//corr.shape[-1], mode='bilinear', align_corners=False)
        corr = upsample(corr)
        four_point_delta = self.update_blocks[0](corr)
        four_point_disp =  four_point_disp + four_point_delta
        # print('four_point_disp.shape', four_point_disp.shape)
        four_point_reshape = four_point_disp.permute(0,2,3,1).reshape(-1,4,2) # [top_left, top_right, bottom_left, bottom_right], [-1, 4, 2]
        # print('four_point_reshape.shape', four_point_reshape.shape)
        # exit()
        # four_point_predictions.append(four_point_reshape)

        return four_point_reshape, four_point_disp

class Classify_loss_time(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BasicEncoder(output_dim=96, norm_fn='instance')
        self.downsample = [4,2,1]
        sz = 32
        self.update_block_4 = GMA_update(sz, classify=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(2, 1)
        self.time_embed = nn.Sequential(
            nn.Linear(81, 32),
            nn.SiLU(),
            nn.Linear(32, 81),
        )


    def forward(self, img1, img2, idx, four_point_disp=None, time=200):
        device = img1.device
        t_emb = timestep_embedding(time, 81)
        t_emb_out = self.time_embed(t_emb)
        image1, image2 = img1, img2

        fmap1 = self.encoder(image1)
        fmap2 = self.encoder(image2)

        batch_size = image1.shape[0]

        if idx == 0:
            four_point_disp = torch.zeros((batch_size, 2, 2, 2)).to(image1.device)

        downsample = self.downsample[idx]

        corr_fn = LocalCorr(fmap1[idx], fmap2[idx])

        coords0, _ = initialize_flow(image1, downsample=downsample)
        # print('four_point_disp.shape', four_point_disp.shape)
        b, _, _ = four_point_disp.shape
        four_point_disp = four_point_disp.reshape(b, 2, 2, 2).permute(0, 3, 1, 2)
        # print('four_point_disp', four_point_disp)

        coords1 = disp_to_coords(four_point_disp, coords0, downsample=downsample) 

        corr = corr_fn(coords1, window_size=9)

        upsample = nn.Upsample(scale_factor=256//corr.shape[-1], mode='bilinear', align_corners=False)
        corr = upsample(corr)

        while len(t_emb_out.shape) < len(corr.shape):
            t_emb_out = t_emb_out[..., None]
        # print(corr.shape, t_emb_out.shape)
        delta_four_point = self.update_block_4(corr + 0*t_emb_out)
        out = self.avgpool(delta_four_point)
        out = torch.flatten(out, start_dim=1)
        
        out = self.fc(out)

        return out
