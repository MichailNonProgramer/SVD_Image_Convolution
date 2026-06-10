import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim


def find_image(name):
    for ext in ('jpg', 'jpeg', 'png', 'bmp'):
        path = f"{name}.{ext}"
        if os.path.exists(path):
            return path
    return None


def trunc_svd(X, k):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def trunc_svd_pre(U, s, Vt, k):
    k = min(k, len(s))
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def build_weights(A):
    edges = np.abs(sobel(A, axis=0)) + np.abs(sobel(A, axis=1))
    return 1 + 10 * (edges / edges.max())


def weighted_lp_prox_grad(A, W, k, p, max_iter=150, alpha=0.1, tol=1e-5):
    X = trunc_svd(A, k)
    errors = []
    step = alpha
    for i in range(max_iter):
        R = A - X
        if p == 1:
            grad = W * np.sign(R)
        else:
            grad = p * W * (np.abs(R) + 1e-8) ** (p - 1) * np.sign(R)
            if p > 2:
                step = max(0.01, step * 0.99)
        Y = trunc_svd(X + step * grad, k)
        err = np.sum(W * (np.abs(A - Y) ** p))
        errors.append(err)
        if i > 0 and abs(errors[-2] - errors[-1]) < tol:
            break
        X = Y
    return X, errors


_image_configs = [("test", "JPEG"), ("base", "BMP"), ("portrait", "Портрет"), ("text", "Текст")]
image_sources = [(p, lbl) for name, lbl in _image_configs if (p := find_image(name))]
k_values = [10, 20, 30, 50, 100]
p_values = [0.5, 1.0, 1.5, 2.0]

for img_path, label in image_sources:
    A = np.array(Image.open(img_path).convert('L'), dtype=np.float64)
    W = build_weights(A)
    print(f"\n[{label}] Вычисление SVD...")
    U_A, S_A, Vt_A = np.linalg.svd(A, full_matrices=False)

    cache = {}
    svd_cache = {}
    print(f"=== {label} ===")
    print(f"{'k':>5}  {'метод':>8}  {'PSNR дБ':>10}  {'SSIM':>8}")

    for k in k_values:
        X_svd = trunc_svd_pre(U_A, S_A, Vt_A, k)
        mse_svd = np.mean((A - X_svd) ** 2)
        svd_cache[k] = {
            'X': X_svd,
            'psnr': 20 * np.log10(255) - 10 * np.log10(mse_svd),
            'ssim': ssim(A, X_svd, data_range=255),
        }
        print(f"{k:>5}  {'SVD':>8}  {svd_cache[k]['psnr']:>10.2f}  {svd_cache[k]['ssim']:>8.4f}")

        for p in p_values:
            alpha = 0.3 if p <= 1.0 else 0.1
            X_p, errors = weighted_lp_prox_grad(A, W, k, p, alpha=alpha)
            mse = np.mean((A - X_p) ** 2)
            psnr = 20 * np.log10(255) - 10 * np.log10(mse)
            ssim_val = ssim(A, X_p, data_range=255)
            cache[(k, p)] = {'X': X_p, 'errors': errors, 'psnr': psnr, 'ssim': ssim_val}
            print(f"{k:>5}  {f'Lp p={p}':>8}  {psnr:>10.2f}  {ssim_val:>8.4f}")

    n_rows = len(p_values) + 1
    n_k = len(k_values)
    fig, axes = plt.subplots(n_rows, n_k, figsize=(4 * n_k, 3.5 * n_rows))

    for col, k in enumerate(k_values):
        res = svd_cache[k]
        axes[0, col].imshow(res['X'], cmap='gray')
        axes[0, col].set_title(f"SVD k={k}\nPSNR={res['psnr']:.1f}dB\nSSIM={res['ssim']:.3f}")
        axes[0, col].axis('off')

    for row, p in enumerate(p_values):
        for col, k in enumerate(k_values):
            res = cache[(k, p)]
            axes[row + 1, col].imshow(res['X'], cmap='gray')
            axes[row + 1, col].set_title(f"Lp p={p}, k={k}\nPSNR={res['psnr']:.1f}dB\nSSIM={res['ssim']:.3f}")
            axes[row + 1, col].axis('off')

    plt.suptitle(f"{label}  (строка 0=SVD, строки 1+= Lp, столбцы=k)", fontsize=13)
    plt.tight_layout()
    plt.show()

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_values, [svd_cache[k]['psnr'] for k in k_values], 's--', linewidth=2, label='SVD')
    ax2.plot(k_values, [svd_cache[k]['ssim'] for k in k_values], 's--', linewidth=2, label='SVD')

    for p in p_values:
        psnrs = [cache[(k, p)]['psnr'] for k in k_values]
        ssims = [cache[(k, p)]['ssim'] for k in k_values]
        ax1.plot(k_values, psnrs, 'o-', linewidth=2, label=f'Lp p={p}')
        ax2.plot(k_values, ssims, 'o-', linewidth=2, label=f'Lp p={p}')

    ax1.set_xlabel('k')
    ax1.set_ylabel('PSNR (дБ)')
    ax1.set_title(f'PSNR vs k  ({label})')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('k')
    ax2.set_ylabel('SSIM')
    ax2.set_title(f'SSIM vs k  ({label})')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
