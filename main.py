import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def find_image(name):
    for ext in ('jpg', 'jpeg', 'png', 'bmp'):
        path = f"{name}.{ext}"
        if os.path.exists(path):
            return path
    return None


def compress_svd_precomputed(U, S, Vt, A, k):
    k = min(k, len(S))
    A_k = np.clip(U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :], 0, 255)
    m, n = A.shape
    ratio = (m * n) / (k * (m + n + 1))
    energy = np.sum(S[:k] ** 2) / np.sum(S ** 2) * 100
    mse = np.mean((A - A_k) ** 2)
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return A_k, ratio, energy, psnr


_image_configs = [("test", "JPEG"), ("base", "BMP"), ("portrait", "Портрет"), ("text", "Текст")]
image_sources = [(p, lbl) for name, lbl in _image_configs if (p := find_image(name))]
k_values = [10, 20, 30, 50, 100]

for img_path, label in image_sources:
    A = np.array(Image.open(img_path).convert('L'), dtype=np.float64)
    print(f"\n[{label}] Вычисление SVD...")
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    fig, axes = plt.subplots(2, len(k_values) + 1, figsize=(22, 8))

    axes[0, 0].imshow(A, cmap='gray')
    axes[0, 0].set_title(f"Оригинал\n({label})")
    axes[0, 0].axis('off')

    axes[1, 0].plot(S, 'b-', linewidth=1)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title("Сингулярные\nзначения")
    axes[1, 0].grid(True)

    print(f"{'k':>6}  {'Ratio':>8}  {'Energy%':>8}  {'PSNR дБ':>10}")

    for j, k in enumerate(k_values):
        A_k, ratio, energy, psnr = compress_svd_precomputed(U, S, Vt, A, k)
        axes[0, j + 1].imshow(A_k, cmap='gray')
        axes[0, j + 1].set_title(f"k={k}\nratio={ratio:.1f}x\nPSNR={psnr:.1f}dB")
        axes[0, j + 1].axis('off')
        axes[1, j + 1].text(0.5, 0.5, f"Энергия\n{energy:.1f}%", ha='center', va='center', fontsize=12)
        axes[1, j + 1].axis('off')
        print(f"{k:>6}  {ratio:>8.2f}  {energy:>8.2f}  {psnr:>10.2f}")

    plt.suptitle(f"SVD-сжатие: {label}")
    plt.tight_layout()
    plt.show()
