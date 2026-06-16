import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import sobel, gaussian_filter
from skimage.metrics import structural_similarity as ssim


def find_image(name):
    for ext in ('jpg', 'jpeg', 'png', 'bmp'):
        path = f"{name}.{ext}"
        if os.path.exists(path):
            return path
    return None


def trunc_svd(X, k):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r_compact = max(1, int(np.sum(s > 0)))
    r = min(k, r_compact)
    return U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]


def trunc_svd_pre(U, s, Vt, k):
    r_compact = max(1, int(np.sum(s > 0)))
    r = min(k, r_compact)
    return U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]


def build_weights(A):
    edges = np.abs(sobel(A, axis=0)) + np.abs(sobel(A, axis=1))
    W = 1 + 10 * (edges / edges.max())
    return gaussian_filter(W, sigma=1.0)


def weighted_l2_prox_grad(A, W, k, max_iter=100, alpha=0.1, tol=1e-5):
    X = trunc_svd(A, k)
    errors = []
    for it in range(max_iter):
        R = A - X
        Y = trunc_svd(X - alpha * (-2 * W * R), k)
        err = np.sum(W * (A - Y) ** 2)
        errors.append(err)
        if it > 0 and abs(errors[-2] - errors[-1]) < tol:
            break
        X = Y
    return X, errors


def weighted_l1_prox_grad(A, W, k, max_iter=100, alpha=0.2, tol=1e-5):
    X = trunc_svd(A, k)
    errors = []
    for it in range(max_iter):
        R = A - X
        Y = trunc_svd(X + alpha * W * np.sign(R), k)
        err = np.sum(W * np.abs(A - Y))
        errors.append(err)
        if it > 0 and abs(errors[-2] - errors[-1]) < tol:
            break
        X = Y
    return X, errors


_image_configs = [("test", "JPEG"), ("base", "BMP"), ("portrait", "Портрет"), ("text", "Текст")]
image_sources = [(p, lbl) for name, lbl in _image_configs if (p := find_image(name))]
k_values = [10, 20, 30, 50, 100]

for img_path, label in image_sources:
    A = np.array(Image.open(img_path).convert('L'), dtype=np.float64)
    W = build_weights(A)
    print(f"\n[{label}] Вычисление SVD...")
    U_A, S_A, Vt_A = np.linalg.svd(A, full_matrices=False)

    print(f"=== {label}: SVD vs L2 vs L1, сравнение по k ===")
    print(f"{'k':>5}  {'PSNR SVD':>10}  {'PSNR L2':>10}  {'PSNR L1':>10}  {'SSIM SVD':>10}  {'SSIM L2':>10}  {'SSIM L1':>10}")

    fig, axes = plt.subplots(4, len(k_values), figsize=(20, 16))

    for j, k in enumerate(k_values):
        X_svd = trunc_svd_pre(U_A, S_A, Vt_A, k)
        X_l2, err_l2 = weighted_l2_prox_grad(A, W, k)
        X_l1, err_l1 = weighted_l1_prox_grad(A, W, k)

        mse_svd = np.mean((A - X_svd) ** 2)
        mse_l2 = np.mean((A - X_l2) ** 2)
        mse_l1 = np.mean((A - X_l1) ** 2)
        psnr_svd = 20 * np.log10(255) - 10 * np.log10(mse_svd)
        psnr_l2 = 20 * np.log10(255) - 10 * np.log10(mse_l2)
        psnr_l1 = 20 * np.log10(255) - 10 * np.log10(mse_l1)
        ssim_svd = ssim(A, X_svd, data_range=255)
        ssim_l2 = ssim(A, X_l2, data_range=255)
        ssim_l1 = ssim(A, X_l1, data_range=255)

        axes[0, j].imshow(X_svd, cmap='gray')
        axes[0, j].set_title(f"SVD k={k}\nPSNR={psnr_svd:.1f}dB\nSSIM={ssim_svd:.3f}")
        axes[0, j].axis('off')

        axes[1, j].imshow(X_l2, cmap='gray')
        axes[1, j].set_title(f"L2 k={k}\nPSNR={psnr_l2:.1f}dB\nSSIM={ssim_l2:.3f}")
        axes[1, j].axis('off')

        axes[2, j].imshow(X_l1, cmap='gray')
        axes[2, j].set_title(f"L1 k={k}\nPSNR={psnr_l1:.1f}dB\nSSIM={ssim_l1:.3f}")
        axes[2, j].axis('off')

        axes[3, j].semilogy(err_l2, label='L2')
        axes[3, j].semilogy(err_l1, label='L1')
        axes[3, j].set_title(f"Сходимость k={k}")
        axes[3, j].legend()
        axes[3, j].grid(True)

        print(f"{k:>5}  {psnr_svd:>10.2f}  {psnr_l2:>10.2f}  {psnr_l1:>10.2f}  {ssim_svd:>10.4f}  {ssim_l2:>10.4f}  {ssim_l1:>10.4f}")

    plt.suptitle(f"SVD vs Взвешенная L2 vs Взвешенная L1: {label}")
    plt.tight_layout()
    plt.show()
