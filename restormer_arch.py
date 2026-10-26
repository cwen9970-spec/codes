## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Fixed orthonormal Haar wavelet primitives
## For each 2x2 patch [[a, b], [c, d]]:
##   L = (a+b+c+d)/2, Dx = (a-b+c-d)/2, Dy = (a+b-c-d)/2, Dd = (a-b-c+d)/2
## Dx: left/right difference, Dy: top/bottom difference.
def haar_dwt(x):
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    L  = (a + b + c + d) / 2
    Dx = (a - b + c - d) / 2
    Dy = (a + b - c - d) / 2
    Dd = (a - b - c + d) / 2
    return L, Dx, Dy, Dd

def haar_idwt(L, Dx, Dy, Dd):
    a = (L + Dx + Dy + Dd) / 2
    b = (L - Dx + Dy - Dd) / 2
    c = (L + Dx - Dy - Dd) / 2
    d = (L - Dx - Dy + Dd) / 2
    B, C, h, w = L.shape
    top = torch.stack([a, b], dim=-1)               # [B, C, h, w, 2]
    bottom = torch.stack([c, d], dim=-1)
    x = torch.stack([top, bottom], dim=-2)          # [B, C, h, w, 2(row), 2(col)]
    x = x.permute(0, 1, 2, 4, 3, 5).reshape(B, C, 2 * h, 2 * w)
    return x



##########################################################################
## Wavelet Frequency Branch (parallel to GDFN)
class FrequencyBranch(nn.Module):
    def __init__(self, dim):
        super(FrequencyBranch, self).__init__()

        # Shared global frequency prompt: Linear(7C, C) -> GELU
        self.prompt_linear = nn.Linear(7 * dim, dim)
        # Modulation head: Linear(C, 3C), gain = 1 + delta
        self.gain_linear = nn.Linear(dim, 3 * dim)
        nn.init.normal_(self.gain_linear.weight, std=1e-3)
        nn.init.zeros_(self.gain_linear.bias)

        # Coefficient-domain content projections (bias-free 1x1, C -> C)
        self.content_low = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.content_mid = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.content_high = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # Frequency fusion: bias-free 1x1, 3C -> C
        self.output_proj = nn.Conv2d(3 * dim, dim, kernel_size=1, bias=False)

    def decompose(self, x):
        # Replicate-pad right/bottom to multiples of 4, then two-level DWT
        pad_h = (-x.shape[-2]) % 4
        pad_w = (-x.shape[-1]) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        L1, D1x, D1y, D1d = haar_dwt(x)
        L2, D2x, D2y, D2d = haar_dwt(L1)
        return L2, (D2x, D2y, D2d), (D1x, D1y, D1d)

    @staticmethod
    def reconstruct(L2, mid, high):
        # Separate IDWT per group; unselected slots are zeros
        D2x, D2y, D2d = mid
        D1x, D1y, D1d = high
        z2 = torch.zeros_like(L2)
        z1 = torch.zeros_like(D1x)

        F_low = haar_idwt(haar_idwt(L2, z2, z2, z2), z1, z1, z1)
        F_mid = haar_idwt(haar_idwt(z2, D2x, D2y, D2d), z1, z1, z1)
        F_high = haar_idwt(z1, D1x, D1y, D1d)
        return F_low, F_mid, F_high

    def compute_gains(self, subbands):
        # subbands order: L2, D2x, D2y, D2d, D1x, D1y, D1d
        with torch.autocast(device_type=subbands[0].device.type, enabled=False):
            e = torch.cat([s.float().abs().mean(dim=(-2, -1)) for s in subbands], dim=1)  # [B, 7C]
            p = F.gelu(self.prompt_linear(e))                                            # [B, C]
            delta = self.gain_linear(p)                                                  # [B, 3C]
            gains = 1 + delta.reshape(delta.shape[0], 3, -1, 1, 1)
        return gains.unbind(1)

    def forward(self, x):
        h, w = x.shape[-2:]
        L2, mid, high = self.decompose(x)

        g_low, g_mid, g_high = self.compute_gains([L2, *mid, *high])

        # Cast gains to the projected content dtype (e.g. FP16 under autocast)
        L2 = self.content_low(L2)
        L2 = g_low.to(L2.dtype) * L2
        mid = [self.content_mid(t) for t in mid]
        mid = [g_mid.to(t.dtype) * t for t in mid]
        high = [self.content_high(t) for t in high]
        high = [g_high.to(t.dtype) * t for t in high]

        F_low, F_mid, F_high = self.reconstruct(L2, mid, high)
        out = torch.cat([F_low[..., :h, :w], F_mid[..., :h, :w], F_high[..., :h, :w]], dim=1)
        return self.output_proj(out)



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, use_frequency_branch=False):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.use_frequency_branch = use_frequency_branch
        if use_frequency_branch:
            self.freq_branch = FrequencyBranch(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.use_frequency_branch:
            u = self.norm2(x)
            x = x + self.ffn(u) + self.freq_branch(u)
        else:
            x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,       ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        use_frequency_branch = False   ## Add wavelet frequency branch parallel to GDFN in every TransformerBlock
    ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, use_frequency_branch=use_frequency_branch) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, use_frequency_branch=use_frequency_branch) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, use_frequency_branch=use_frequency_branch) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, use_frequency_branch=use_frequency_branch) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, use_frequency_branch=use_frequency_branch) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, use_frequency_branch=use_frequency_branch) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, use_frequency_branch=use_frequency_branch) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, use_frequency_branch=use_frequency_branch) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1




##########################################################################
## Wavelet FFN (WFFN v1) -- replaces GDFN in decoder levels 3/2/1 and refinement
## Enabled only through WaveletRestormer; the original Restormer above is unchanged.

def avg3(x):
    # Fixed 3x3 mean filter; borders are normalised by the number of valid pixels
    return F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)

def box_k(x, k):
    # Separable k x k mean filter (H then W); borders normalised by valid pixels
    x = F.avg_pool2d(x, (k, 1), stride=1, padding=(k // 2, 0), count_include_pad=False)
    return F.avg_pool2d(x, (1, k), stride=1, padding=(0, k // 2), count_include_pad=False)


def spectral_stats(x, K=8):
    """Square-ring + orientation binned log-amplitude spectrum, returns B x C x 20 (fp32)."""
    B, C, H, W = x.shape
    win = torch.outer(torch.hann_window(H, device=x.device),
                      torch.hann_window(W, device=x.device))
    amp = torch.log1p(torch.fft.rfft2(x * win, norm='ortho').abs())       # B,C,H,W//2+1

    fy = (torch.fft.fftfreq(H, device=x.device).abs() / 0.5)[:, None]     # normalised to [0,1]
    fx = (torch.fft.rfftfreq(W, device=x.device) / 0.5)[None, :]
    r, rmin = torch.maximum(fy, fx), torch.minimum(fy, fx)

    ring = (r * K).long().clamp(max=K - 1)                                 # 0..7
    band_in = torch.where(r < 0.25, 0.0, torch.where(r < 0.5, 0.25, 0.5))
    orient = torch.where(rmin >= band_in, 2,                               # 2: corners
                         torch.where(fx > fy, 0, 1))                       # 0: left/right, 1: up/down
    bin_id = torch.where(ring < 2, ring, 2 + (ring - 2) * 3 + orient)      # 0..19

    idx = bin_id.flatten()
    s = torch.zeros(B, C, 20, device=x.device).index_add_(2, idx, amp.flatten(2))
    cnt = torch.bincount(idx, minlength=20).clamp(min=1)
    return s / cnt


class PromptExtractor(nn.Module):
    """Encoder skip feature -> global prompt p (B x 64)."""
    def __init__(self, in_channels, squeeze=16, width=64):
        super(PromptExtractor, self).__init__()
        self.squeeze = nn.Conv1d(in_channels, squeeze, kernel_size=1, bias=True)
        self.mlp = nn.Sequential(nn.Linear(20 * squeeze, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            s = spectral_stats(x.float())
            return self.mlp(self.squeeze(s).flatten(1))


class WFFN(nn.Module):
    """proj_in -> 2-level Haar -> per-band (remove, denoise, compensate) -> IDWT -> proj_out."""
    def __init__(self, dim, ffn_expansion_factor, k_hf, k_lfmf, prompt_dim=64, eps=1e-6):
        super(WFFN, self).__init__()
        assert k_hf % 2 == 1 and k_lfmf % 2 == 1, 'box_k sizes must be odd'
        h = int(dim * ffn_expansion_factor)
        self.h, self.k_hf, self.k_lfmf, self.eps = h, k_hf, k_lfmf, eps
        self.groups = -(-h // 8)

        self.proj_in = nn.Conv2d(dim, h, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(h, dim, kernel_size=1, bias=False)

        # LF: haze cue rho and compensation u_l (the latter is shared with MF)
        self.lf_rho = nn.Conv2d(h, h, kernel_size=5, padding=2, groups=h, bias=True)
        self.lf_ul = nn.Conv2d(h, h, kernel_size=5, padding=2, groups=h, bias=True)
        # MF / HF: per-channel 1x7 rain response and log kappa; HF compensation from ll1_raw
        self.mf_rain = nn.Conv2d(3 * h, 3 * h, kernel_size=(1, 7), padding=(0, 3), groups=3 * h, bias=True)
        self.hf_rain = nn.Conv2d(3 * h, 3 * h, kernel_size=(1, 7), padding=(0, 3), groups=3 * h, bias=True)
        self.mf_log_kappa = nn.Parameter(torch.full((1, 3 * h, 1, 1), 2.0))
        self.hf_log_kappa = nn.Parameter(torch.full((1, 3 * h, 1, 1), 2.0))
        self.hf_ul = nn.Conv2d(h, h, kernel_size=5, padding=2, groups=h, bias=True)

        # Per-block head: p -> 9 knob groups x ceil(h/8)
        self.head = nn.Sequential(nn.Linear(prompt_dim, 16), nn.GELU(), nn.Linear(16, 9 * self.groups))
        nn.init.normal_(self.head[2].weight, std=0.01)
        nn.init.constant_(self.head[2].bias, -2.0)

        self.collect_stats = False
        self.stats = {}

    def knobs(self, p):
        k = self.head(p).float().view(p.shape[0], 9, self.groups)
        k = k.repeat_interleave(8, dim=2)[:, :, :self.h, None, None]           # B x 9 x h x 1 x 1
        sp, sg = F.softplus, torch.sigmoid
        return dict(lf_A=sp(k[:, 0]), lf_tau2=sp(k[:, 1]), lf_ug=k[:, 2],
                    mf_lam=sg(k[:, 3]), mf_tau2=sp(k[:, 4]), mf_ug=k[:, 5],
                    hf_lam=sg(k[:, 6]), hf_tau2=sp(k[:, 7]), hf_ug=k[:, 8])

    def denoise(self, z, tau2, k):
        # fp32: local mean/variance, scale-normalised variance, Wiener-like gate
        mu = avg3(z)
        e = (avg3(z * z) - mu * mu).clamp(min=0)
        e_n = e / (box_k(e, k) + self.eps)
        g = e_n / (e_n + tau2 + self.eps)
        return mu + g * (z - mu), g

    def remove_rain(self, x, conv, log_kappa, lam, k):
        # fp32: energy of the 1x7 response relative to its k x k neighbourhood
        a = conv(x.float())
        s = avg3(a * a)
        r = s / (box_k(s, k) + self.eps)
        m = torch.sigmoid(torch.log(r + self.eps) - log_kappa.float())
        return (1 - lam.repeat(1, 3, 1, 1) * m) * x.float(), m

    def forward(self, x, p):
        H, W = x.shape[-2:]
        u = self.proj_in(x)
        ph, pw = (-H) % 4, (-W) % 4
        if ph or pw:
            u = F.pad(u, (0, pw, 0, ph), mode='reflect')
        LL1, *hf = haar_dwt(u)
        LF, *mf = haar_dwt(LL1)
        HF, MF = torch.cat(hf, dim=1), torch.cat(mf, dim=1)      # direction-major: [Dx; Dy; Dd]
        ll1_raw, dtype = LL1, u.dtype
        kn = self.knobs(p)

        rho = torch.sigmoid(self.lf_rho(LF).float())
        ul_lf = self.lf_ul(LF).float()
        ul_hf = self.hf_ul(ll1_raw).float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            # LF: (1) dehaze, (2) denoise, (3) compensate
            z_lf = LF.float() - kn['lf_A'] * rho
            w_lf, g_lf = self.denoise(z_lf, kn['lf_tau2'], self.k_lfmf)
            c_lf = 1 + F.softplus(kn['lf_ug'] + ul_lf)
            LF2 = c_lf * w_lf
        with torch.autocast(device_type=x.device.type, enabled=False):
            # MF: (1) remove rain, (2) denoise, (3) compensate with the LF u_l
            z_mf, m_mf = self.remove_rain(MF, self.mf_rain, self.mf_log_kappa, kn['mf_lam'], self.k_lfmf)
            w_mf, g_mf = self.denoise(z_mf, kn['mf_tau2'].repeat(1, 3, 1, 1), self.k_lfmf)
            c_mf = 1 + F.softplus((kn['mf_ug'] + ul_lf).repeat(1, 3, 1, 1))
            MF2 = c_mf * w_mf
            # HF: same steps; u_l from ll1_raw
            z_hf, m_hf = self.remove_rain(HF, self.hf_rain, self.hf_log_kappa, kn['hf_lam'], self.k_hf)
            w_hf, g_hf = self.denoise(z_hf, kn['hf_tau2'].repeat(1, 3, 1, 1), self.k_hf)
            c_hf = 1 + F.softplus((kn['hf_ug'] + ul_hf).repeat(1, 3, 1, 1))
            HF2 = c_hf * w_hf

        if self.collect_stats:
            self.stats = dict(lf_rho=rho.detach(), lf_g=g_lf.detach(), lf_comp=c_lf.detach(),
                              mf_m=m_mf.detach(), mf_g=g_mf.detach(), mf_comp=c_mf.detach(),
                              hf_m=m_hf.detach(), hf_g=g_hf.detach(), hf_comp=c_hf.detach())

        LL1n = haar_idwt(LF2.to(dtype), *MF2.to(dtype).chunk(3, dim=1))
        v = haar_idwt(LL1n, *HF2.to(dtype).chunk(3, dim=1))[..., :H, :W]
        return self.proj_out(v)


class WaveletTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, k_hf, k_lfmf):
        super(WaveletTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.wffn = WFFN(dim, ffn_expansion_factor, k_hf, k_lfmf)

    def forward(self, x, p):
        x = x + self.attn(self.norm1(x))
        x = x + self.wffn(self.norm2(x), p)
        return x


class PromptSequential(nn.ModuleList):
    """Sequential container that passes the level prompt p to every block."""
    def forward(self, x, p):
        for block in self:
            x = block(x, p)
        return x


##---------- Restormer with WFFN in decoder / refinement -----------------------
class WaveletRestormer(Restormer):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 48,
        num_blocks = [4,6,6,8],
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        dual_pixel_task = False,
        wffn_k_hf = [15, 31, 63, 63],      ## box_k for HF:    [decoder3, decoder2, decoder1, refinement]
        wffn_k_lfmf = [7, 15, 31, 31],     ## box_k for LF/MF: [decoder3, decoder2, decoder1, refinement]
    ):
        super(WaveletRestormer, self).__init__(inp_channels, out_channels, dim, num_blocks, num_refinement_blocks,
                                               heads, ffn_expansion_factor, bias, LayerNorm_type, dual_pixel_task)

        def stage(width, n, head, i):
            return PromptSequential([WaveletTransformerBlock(width, head, ffn_expansion_factor, bias, LayerNorm_type,
                                                             wffn_k_hf[i], wffn_k_lfmf[i]) for _ in range(n)])

        # Replace the GDFN decoder / refinement stages; encoder and latent keep GDFN
        self.decoder_level3 = stage(int(dim*2**2), num_blocks[2], heads[2], 0)
        self.decoder_level2 = stage(int(dim*2**1), num_blocks[1], heads[1], 1)
        self.decoder_level1 = stage(int(dim*2**1), num_blocks[0], heads[0], 2)
        self.refinement = stage(int(dim*2**1), num_refinement_blocks, heads[0], 3)

        # One prompt extractor per decoder level, fed by the encoder skip of that level
        self.prompt_level3 = PromptExtractor(int(dim*2**2))
        self.prompt_level2 = PromptExtractor(int(dim*2**1))
        self.prompt_level1 = PromptExtractor(dim)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        p3 = self.prompt_level3(out_enc_level3)
        p2 = self.prompt_level2(out_enc_level2)
        p1 = self.prompt_level1(out_enc_level1)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3, p3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2, p2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, p1)

        out_dec_level1 = self.refinement(out_dec_level1, p1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1
