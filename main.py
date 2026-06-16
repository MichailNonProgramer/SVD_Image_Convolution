import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def find_image(name):
    for ext in ('jpg', 'jpeg', 'png', 'bmp', 'webp'):
        path = f"{name}.{ext}"
        if os.path.exists(path):
            return path
    return None


def compact_svd(U, s, Vt):
    r = int(np.sum(s > 0))
    r = max(r, 1)
    return U[:, :r], s[:r], Vt[:r, :], r


def trunc_svd_from(U, s, Vt, k):
    k = min(k, len(s))
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :], k


def compact_trunc_svd(U, s, Vt, A, k):
    U_c, s_c, Vt_c, r_compact = compact_svd(U, s, Vt)
    k_actual = min(k, r_compact)
    A_k, _ = trunc_svd_from(U_c, s_c, Vt_c, k_actual)
    A_k = np.clip(A_k, 0, 255)
    m, n = A.shape
    ratio = (m * n) / (k_actual * (m + n + 1))
    energy = np.sum(s_c[:k_actual] ** 2) / np.sum(s ** 2) * 100
    mse = np.mean((A - A_k) ** 2)
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    ssim_val = ssim(A, A_k, data_range=255)
    return A_k, ratio, energy, psnr, ssim_val, r_compact, k_actual


def plain_trunc_svd(U, s, Vt, A, k):
    k_actual = min(k, len(s))
    A_k = np.clip(U[:, :k_actual] @ np.diag(s[:k_actual]) @ Vt[:k_actual, :], 0, 255)
    m, n = A.shape
    ratio = (m * n) / (k_actual * (m + n + 1))
    energy = np.sum(s[:k_actual] ** 2) / np.sum(s ** 2) * 100
    mse = np.mean((A - A_k) ** 2)
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    ssim_val = ssim(A, A_k, data_range=255)
    return A_k, ratio, energy, psnr, ssim_val


_image_configs = [("test", "JPEG"), ("base", "BMP"), ("portrait", "Портрет"), ("text", "Текст")]
image_sources = [(p, lbl) for name, lbl in _image_configs if (p := find_image(name))]
k_values = [10, 20, 30, 50, 100]

for img_path, label in image_sources:
    A = np.array(Image.open(img_path).convert('L'), dtype=np.float64)
    print(f"\n[{label}] Вычисление SVD...")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    _, _, _, r_compact = compact_svd(U, s, Vt)
    r_full = len(s)
    print(f"  Полный ранг (economy): {r_full}")
    print(f"  Компактный ранг (tol): {r_compact}")

    fig, axes = plt.subplots(3, len(k_values) + 1, figsize=(22, 12))

    axes[0, 0].imshow(A, cmap='gray')
    axes[0, 0].set_title(f"Оригинал\n({label})")
    axes[0, 0].axis('off')

    axes[1, 0].plot(s, 'b-', linewidth=1)
    axes[1, 0].axvline(x=r_compact - 1, color='r', linestyle='--', label=f'compact r={r_compact}')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title("Сингулярные\nзначения")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True)

    axes[2, 0].axis('off')

    print(f"\n{'k':>6}  {'r_compact':>10}  {'k_actual':>10}  "
          f"{'PSNR Plain':>12}  {'PSNR Compact':>14}  "
          f"{'SSIM Plain':>12}  {'SSIM Compact':>14}")

    for j, k in enumerate(k_values):
        A_plain, ratio_p, energy_p, psnr_p, ssim_p = plain_trunc_svd(U, s, Vt, A, k)
        A_comp, ratio_c, energy_c, psnr_c, ssim_c, r_c, k_act = compact_trunc_svd(U, s, Vt, A, k)

        axes[0, j + 1].imshow(A_plain, cmap='gray')
        axes[0, j + 1].set_title(f"TruncSVD k={k}\nPSNR={psnr_p:.1f}dB\nSSIM={ssim_p:.3f}")
        axes[0, j + 1].axis('off')

        axes[1, j + 1].imshow(A_comp, cmap='gray')
        axes[1, j + 1].set_title(f"Compact+Trunc k={k}\n(k_eff={k_act})\nPSNR={psnr_c:.1f}dB\nSSIM={ssim_c:.3f}")
        axes[1, j + 1].axis('off')

        diff = np.abs(A_plain.astype(float) - A_comp.astype(float))
        axes[2, j + 1].imshow(diff, cmap='hot')
        axes[2, j + 1].set_title(f"Разность\nmax={diff.max():.1f}")
        axes[2, j + 1].axis('off')

        print(f"{k:>6}  {r_c:>10}  {k_act:>10}  "
              f"{psnr_p:>12.2f}  {psnr_c:>14.2f}  "
              f"{ssim_p:>12.4f}  {ssim_c:>14.4f}")

    plt.suptitle(f"TruncatedSVD vs CompactSVD+TruncatedSVD: {label}", fontsize=13)
    plt.tight_layout()
    plt.show()

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    psnr_plain_list = [plain_trunc_svd(U, s, Vt, A, k)[3] for k in k_values]
    psnr_comp_list  = [compact_trunc_svd(U, s, Vt, A, k)[3] for k in k_values]
    ssim_plain_list = [plain_trunc_svd(U, s, Vt, A, k)[4] for k in k_values]
    ssim_comp_list  = [compact_trunc_svd(U, s, Vt, A, k)[4] for k in k_values]

    ax1.plot(k_values, psnr_plain_list, 's--', linewidth=2, label='TruncSVD')
    ax1.plot(k_values, psnr_comp_list,  'o-',  linewidth=2, label='Compact+Trunc')
    ax1.set_xlabel('k'); ax1.set_ylabel('PSNR (дБ)')
    ax1.set_title(f'PSNR vs k  ({label})'); ax1.legend(); ax1.grid(True)

    ax2.plot(k_values, ssim_plain_list, 's--', linewidth=2, label='TruncSVD')
    ax2.plot(k_values, ssim_comp_list,  'o-',  linewidth=2, label='Compact+Trunc')
    ax2.set_xlabel('k'); ax2.set_ylabel('SSIM')
    ax2.set_title(f'SSIM vs k  ({label})'); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.show()
