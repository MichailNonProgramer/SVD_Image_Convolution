import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compress_image_svd(image_path, k):
    # Загрузка и преобразование в полутоновое
    img = Image.open(image_path).convert('L')
    A = np.array(img, dtype=np.float64)
    m, n = A.shape

    # 1. Полное SVD
    U_full, S_full, Vt_full = np.linalg.svd(A, full_matrices=False)
    r = len(S_full)  # Ранг матрицы (все ненулевые сингулярные значения)

    # 2 & 3. Compact + Truncated SVD (k должно быть <= r)
    k = min(k, r)
    U_k = U_full[:, :k]
    S_k = S_full[:k]
    Vt_k = Vt_full[:k, :]

    # Восстановление изображения
    A_compressed = U_k @ np.diag(S_k) @ Vt_k
    A_compressed = np.clip(A_compressed, 0, 255).astype(np.uint8)

    # Расчет коэффициента сжатия
    original_size = m * n
    compressed_size = k * (m + n + 1)
    compression_ratio = original_size / compressed_size

    return A_compressed, compression_ratio, S_full

# Использование
image_path = "./test.jpg"
k = 50  # Количество сохраняемых сингулярных значений
compressed_img, ratio, singular_values = compress_image_svd(image_path, k)

print(f"Коэффициент сжатия: {ratio:.2f}")
print(f"Сохраняемая информация: {np.sum(singular_values[:k]**2) / np.sum(singular_values**2) * 100:.2f}%")

# Визуализация
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(Image.open(image_path).convert('L'), cmap='gray')
axes[0].set_title('Оригинал')
axes[0].axis('off')

axes[1].imshow(compressed_img, cmap='gray')
axes[1].set_title(f'Сжатое (k={k})')
axes[1].axis('off')

axes[2].plot(singular_values, 'o-', markersize=3)
axes[2].set_title('Сингулярные значения')
axes[2].axvline(x=k, color='r', linestyle='--', label=f'k={k}')
axes[2].set_xlabel('Номер')
axes[2].set_ylabel('Значение')
axes[2].legend()
plt.tight_layout()
plt.show()