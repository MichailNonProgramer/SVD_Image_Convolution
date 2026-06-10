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
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def trunc_svd_pre(U, s, Vt, k):
    k = min(k, len(s))
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def build_weights(A):
    edges = np.abs(sobel(A, axis=0)) + np.abs(sobel(A, axis=1))
    W = 1 + 10 * (edges / edges.max())
    return gaussian_filter(W, sigma=1.0)


def weighted_l1_prox_grad(A, W, k, max_iter=150, alpha=0.2, tol=1e-5):
    X = trunc_svd(A, k)
    errors = []
    for it in range(max_iter):
        R = A - X
        X_new = trunc_svd(X + alpha * W * np.sign(R), k)
        err = np.sum(W * np.abs(A - X_new))
        errors.append(err)
        if it > 0 and abs(errors[-2] - errors[-1]) < tol:
            break
        X = X_new
    return X, errors


_image_configs = [("test", "JPEG"), ("base", "BMP"), ("portrait", "Портрет"), ("text", "Текст")]
image_sources = [(p, lbl) for name, lbl in _image_configs if (p := find_image(name))]
k_values = [10, 20, 30, 50, 100]
alpha_values = [0.05, 0.1, 0.2, 0.4]

for img_path, label in image_sources:
    A = np.array(Image.open(img_path).convert('L'), dtype=np.float64)
    W = build_weights(A)
    print(f"\n[{label}] Вычисление SVD...")
    U_A, S_A, Vt_A = np.linalg.svd(A, full_matrices=False)

    print(f"=== {label}: сравнение по k (alpha=0.2) ===")
    print(f"{'k':>5}  {'WL1 SVD':>12}  {'WL1 L1':>12}  {'PSNR SVD':>10}  {'PSNR L1':>10}  {'SSIM L1':>8}")

    fig, axes = plt.subplots(3, len(k_values) + 1, figsize=(20, 12))
    axes[0, 0].imshow(A, cmap='gray')
    axes[0, 0].set_title(f"Оригинал\n({label})")
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')

    for j, k in enumerate(k_values):
        X_svd = trunc_svd_pre(U_A, S_A, Vt_A, k)
        X_l1, errs = weighted_l1_prox_grad(A, W, k, alpha=0.2)

        wl1_svd = np.sum(W * np.abs(A - X_svd))
        wl1_l1 = np.sum(W * np.abs(A - X_l1))
        mse_svd = np.mean((A - X_svd) ** 2)
        mse_l1 = np.mean((A - X_l1) ** 2)
        psnr_svd = 20 * np.log10(255) - 10 * np.log10(mse_svd)
        psnr_l1 = 20 * np.log10(255) - 10 * np.log10(mse_l1)
        ssim_l1 = ssim(A, X_l1, data_range=255)

        axes[0, j + 1].imshow(X_l1, cmap='gray')
        axes[0, j + 1].set_title(f"L1 k={k}\nPSNR={psnr_l1:.1f}dB\nSSIM={ssim_l1:.3f}")
        axes[0, j + 1].axis('off')
        axes[1, j + 1].imshow(np.abs(A - X_l1), cmap='hot')
        axes[1, j + 1].set_title(f"Ошибка k={k}")
        axes[1, j + 1].axis('off')
        axes[2, j + 1].semilogy(errs)
        axes[2, j + 1].set_title(f"Сходимость k={k}")
        axes[2, j + 1].grid(True)

        print(f"{k:>5}  {wl1_svd:>12.2f}  {wl1_l1:>12.2f}  {psnr_svd:>10.2f}  {psnr_l1:>10.2f}  {ssim_l1:>8.4f}")

    plt.suptitle(f"Взвешенная L1: {label}, сравнение по k")
    plt.tight_layout()
    plt.show()

    print(f"\n=== {label}: сравнение по alpha (k=30) ===")
    print(f"{'alpha':>7}  {'WL1-err':>12}  {'PSNR дБ':>10}  {'SSIM':>8}  {'Итераций':>10}")

    fig2, axes2 = plt.subplots(1, len(alpha_values) + 1, figsize=(18, 5))
    axes2[0].imshow(A, cmap='gray')
    axes2[0].set_title(f"Оригинал ({label})")
    axes2[0].axis('off')

    for j, alpha in enumerate(alpha_values):
        X_l1, errs = weighted_l1_prox_grad(A, W, k=30, alpha=alpha)
        wl1 = np.sum(W * np.abs(A - X_l1))
        mse = np.mean((A - X_l1) ** 2)
        psnr = 20 * np.log10(255) - 10 * np.log10(mse)
        ssim_val = ssim(A, X_l1, data_range=255)

        axes2[j + 1].imshow(X_l1, cmap='gray')
        axes2[j + 1].set_title(f"alpha={alpha}\nPSNR={psnr:.1f}dB\nSSIM={ssim_val:.3f}")
        axes2[j + 1].axis('off')
        print(f"{alpha:>7}  {wl1:>12.2f}  {psnr:>10.2f}  {ssim_val:>8.4f}  {len(errs):>10}")

    plt.suptitle(f"Взвешенная L1: {label}, сравнение по alpha (k=30)")
    plt.tight_layout()
    plt.show()
