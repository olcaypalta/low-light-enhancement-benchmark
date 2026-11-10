import os, sys, csv
import numpy as np, cv2, torch, torch.nn as nn


GAMMA = 1.8
CLAHE_CLIP, CLAHE_TILE = 2.0, 8
SSR_SIGMA = 25
MSR_SIGMAS = (15, 80, 250)
PM_ITER, PM_KAPPA, PM_STEP = 5, 25, 0.2
GF_R, GF_EPS = 2, 1.0
US_SIGMA, US_AMOUNT = 3.0, 1.0
ZDCE_W = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Epoch99.pth')
ZDCE_URL = 'https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/snapshots/Epoch99.pth'

trunc = lambda a: np.clip(a, 0, 255).astype(np.uint8)


def gamma(im):
    lut = np.array([((i / 255.0) ** (1.0 / GAMMA)) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(im, lut)


def hist_eq(im):
    y = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    y[:, :, 0] = cv2.equalizeHist(y[:, :, 0])
    return cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)


def clahe_rgb(im):
    cl = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    return cv2.merge([cl.apply(c) for c in cv2.split(im)])


def clahe_lab(im):
    cl = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _ssr(im, s):
    i = im.astype(np.float64) + 1.0
    return np.log(i) - np.log(cv2.GaussianBlur(i, (0, 0), s))


def retinex_ssr(im):
    return cv2.normalize(_ssr(im, SSR_SIGMA), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def retinex_msr(im):
    r = sum(_ssr(im, s) for s in MSR_SIGMAS) / len(MSR_SIGMAS)
    return cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def perona_malik(im):
    I = im.astype(np.float32)
    c = lambda d: np.exp(-(d / PM_KAPPA) ** 2)
    for _ in range(PM_ITER):
        dN = np.roll(I, 1, 0) - I; dS = np.roll(I, -1, 0) - I
        dE = np.roll(I, -1, 1) - I; dW = np.roll(I, 1, 1) - I
        I = I + PM_STEP * (c(dN) * dN + c(dS) * dS + c(dE) * dE + c(dW) * dW)
    return trunc(I)


def _guided(I, p, r, eps):
    k = (2 * r + 1, 2 * r + 1)
    mI = cv2.boxFilter(I, -1, k); mp = cv2.boxFilter(p, -1, k)
    a = (cv2.boxFilter(I * p, -1, k) - mI * mp) / (cv2.boxFilter(I * I, -1, k) - mI * mI + eps)
    b = mp - a * mI
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


def guided(im):
    x = im.astype(np.float32)
    return trunc(np.dstack([_guided(x[:, :, c], x[:, :, c], GF_R, GF_EPS) for c in range(3)]))


def unsharp(im):
    return cv2.addWeighted(im, 1 + US_AMOUNT, cv2.GaussianBlur(im, (0, 0), US_SIGMA), -US_AMOUNT, 0)


def weighted_fusion(im):
    res = cv2.createMergeMertens().process([im, guided(im), unsharp(im)])
    return trunc(res * 255)


class _DCE(nn.Module):
    def __init__(s):
        super().__init__()
        s.relu = nn.ReLU(inplace=True)
        s.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1); s.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1); s.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        s.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1); s.e_conv5 = nn.Conv2d(64, 32, 3, 1, 1); s.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1)
        s.e_conv7 = nn.Conv2d(64, 24, 3, 1, 1)

    def forward(s, x):
        x1 = s.relu(s.e_conv1(x)); x2 = s.relu(s.e_conv2(x1)); x3 = s.relu(s.e_conv3(x2)); x4 = s.relu(s.e_conv4(x3))
        x5 = s.relu(s.e_conv5(torch.cat([x3, x4], 1))); x6 = s.relu(s.e_conv6(torch.cat([x2, x5], 1)))
        xr = torch.tanh(s.e_conv7(torch.cat([x1, x6], 1)))
        for r in torch.split(xr, 3, dim=1):
            x = x + r * (torch.pow(x, 2) - x)
        return x


_NET = None


def zero_dce(im):
    global _NET
    if _NET is None:
        if not os.path.exists(ZDCE_W):
            torch.hub.download_url_to_file(ZDCE_URL, ZDCE_W)
        _NET = _DCE(); _NET.load_state_dict(torch.load(ZDCE_W, map_location='cpu', weights_only=True)); _NET.eval()
    x = torch.from_numpy(im[:, :, ::-1].copy()).float().permute(2, 0, 1)[None] / 255
    with torch.no_grad():
        y = _NET(x)[0].permute(1, 2, 0).numpy()[:, :, ::-1] * 255
    return trunc(y)


METHODS = {'1_perona_malik': perona_malik, '2_retinex_ssr': retinex_ssr, '3_retinex_msr': retinex_msr,
           '4_clahe_rgb': clahe_rgb, '5_clahe_lab': clahe_lab, '6_gamma': gamma, '7_hist_eq': hist_eq,
           '8_weighted_fusion': weighted_fusion, '9_guided': guided, '10_unsharp': unsharp, '11_zero_dce': zero_dce}


def rd(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def wr(p, im):
    cv2.imencode('.png', im)[1].tofile(p)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


if __name__ == '__main__':
    ROOT = sys.argv[1]
    LOW = os.path.join(ROOT, 'low')
    names = sorted(os.listdir(LOW), key=lambda s: int(os.path.splitext(s)[0]))
    if '--check' in sys.argv:
        rows = []
        for m, fn in METHODS.items():
            v = [psnr(fn(rd(os.path.join(LOW, n))), rd(os.path.join(ROOT, 'reference_outputs', m, n))) for n in names[::25]]
            rows.append([m, len(v), f'{np.mean(v):.2f}', f'{np.min(v):.2f}'])
            print(m, rows[-1][2:], flush=True)
        with open(os.path.join(ROOT, 'check.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['method', 'n', 'psnr_mean', 'psnr_min']); w.writerows(rows)
    else:
        for m, fn in METHODS.items():
            od = os.path.join(ROOT, 'output', m)
            os.makedirs(od, exist_ok=True)
            for n in names:
                wr(os.path.join(od, n), fn(rd(os.path.join(LOW, n))))
            print(m, flush=True)
