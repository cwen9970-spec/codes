"""D4 test script for Restormer / WaveletRestormer (deraining and Gaussian color denoising).

The network is built from network_g of the training yml (any `type` in restormer_arch.py), so the same
script tests the GDFN baseline and the WFFN model. Run from the repository root, e.g.

  python experiment/exp4/D4/test.py --task deraining \
      --opt experiment/exp4/D4/options/Deraining_Restormer_30k_WFFN_v1.yml \
      --weights experiment/exp4/D4/experiments/Deraining_Restormer_30k_WFFN_v1/models/net_g_latest.pth

  python experiment/exp4/D4/test.py --task denoising --sigmas 15,25,50 \
      --opt experiment/exp4/D4/options/GaussianColorDenoising_Restormer_30k_WFFN_v1.yml \
      --weights experiment/exp4/D4/experiments/GaussianColorDenoising_Restormer_30k_WFFN_v1/models/net_g_latest.pth

Metrics (uint8, same conventions as the official evaluation):
  deraining  PSNR / SSIM on the Y channel (official evaluate_PSNR_SSIM.m)
  denoising  PSNR / SSIM on RGB (official evaluate_gaussian_color_denoising.py), noise seed 0 per image
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from basicsr.models.archs import restormer_arch  # noqa: E402
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim  # noqa: E402

DERAIN_SETS = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'Test2800']
DENOISE_SETS = ['CBSD68', 'Kodak', 'McMaster', 'Urban100']
EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
FACTOR = 8


def build_model(opt_path, weights, device):
    opt = dict(yaml.safe_load(Path(opt_path).read_text(encoding='utf-8'))['network_g'])
    model = getattr(restormer_arch, opt.pop('type'))(**opt)
    ckpt = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt.get('params', ckpt.get('params_ema', ckpt)), strict=True)
    return model.to(device).eval()


def images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)


def derain_pairs(root):
    """input/target (Rain13K layout, same file names) or rainy/gt (norain-xxx -> rain-xxx)."""
    if (root / 'input').is_dir() and (root / 'target').is_dir():
        return [(p, root / 'target' / p.name) for p in images(root / 'input')]
    if (root / 'rainy').is_dir() and (root / 'gt').is_dir():
        return [(root / 'rainy' / g.name.replace('norain', 'rain'), g) for g in images(root / 'gt')]
    return []


@torch.no_grad()
def restore(model, img, device):
    """img: H x W x 3 RGB float in [0, 1] (noise not clipped). Returns uint8 RGB."""
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float().to(device)
    h, w = x.shape[-2:]
    ph, pw = (-h) % FACTOR, (-w) % FACTOR
    x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    y = model(x)[..., :h, :w].clamp(0, 1)
    return (y[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)


def read_rgb(p):
    return cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def metrics(out_rgb, gt_rgb, y_channel):
    a, b = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR), cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)  # basicsr expects BGR for Y
    return (calculate_psnr(a, b, 0, test_y_channel=y_channel), calculate_ssim(a, b, 0, test_y_channel=y_channel))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True, choices=['deraining', 'denoising'])
    p.add_argument('--opt', required=True, help='training yml; only network_g is used')
    p.add_argument('--weights', required=True)
    p.add_argument('--input_dir', default=None, help='default: Deraining/Datasets/test or Denoising/Datasets/test')
    p.add_argument('--datasets', default=None, help='comma separated; default: all official sets that exist')
    p.add_argument('--sigmas', default='15,25,50', help='denoising only')
    p.add_argument('--result_dir', default=None, help='default: experiment/exp4/D4/results/<yml name>')
    p.add_argument('--save_images', action='store_true')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(args.opt, args.weights, device)
    name = Path(args.opt).stem
    result_dir = Path(args.result_dir) if args.result_dir else ROOT / 'experiment/exp4/D4/results' / name
    result_dir.mkdir(parents=True, exist_ok=True)
    default_root = 'Deraining/Datasets/test' if args.task == 'deraining' else 'Denoising/Datasets/test'
    in_root = Path(args.input_dir) if args.input_dir else ROOT / default_root
    sets = args.datasets.split(',') if args.datasets else (DERAIN_SETS if args.task == 'deraining' else DENOISE_SETS)
    summary = dict(task=args.task, opt=str(Path(args.opt).resolve()), weights=str(Path(args.weights).resolve()),
                   network=type(model).__name__, params=sum(t.numel() for t in model.parameters()), results={})

    for ds in sets:
        if args.task == 'deraining':
            pairs = derain_pairs(in_root / ds)
            runs = [(None, pairs)]
        else:
            files = images(in_root / ds) if (in_root / ds).is_dir() else []
            runs = [(int(s), [(f, f) for f in files]) for s in args.sigmas.split(',')]
        for sigma, pairs in runs:
            key = ds if sigma is None else f'{ds}_sigma{sigma}'
            if not pairs:
                print(f'skip {key}: no images under {in_root / ds}')
                continue
            psnr, ssim = [], []
            for inp, gt in pairs:
                gt_rgb = read_rgb(gt)
                img = np.float32(read_rgb(inp)) / 255.0
                if sigma is not None:
                    np.random.seed(seed=0)  # same as the official test script
                    img = img + np.random.normal(0, sigma / 255.0, img.shape).astype(np.float32)
                torch.cuda.empty_cache()
                out = restore(model, img, device)
                ps, ss = metrics(out, gt_rgb, y_channel=args.task == 'deraining')
                psnr.append(ps); ssim.append(ss)
                if args.save_images:
                    d = result_dir / key
                    d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(d / (Path(gt).stem + '.png')), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            summary['results'][key] = dict(images=len(psnr), psnr=float(np.mean(psnr)), ssim=float(np.mean(ssim)))
            print(f'{key}: {len(psnr)} images  PSNR {np.mean(psnr):.4f}  SSIM {np.mean(ssim):.4f}')

    (result_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('saved', result_dir / 'metrics.json')


if __name__ == '__main__':
    main()
