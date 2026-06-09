import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim


def trunc_svd(X, k):
    """Аппроксимирует X матрицей ранга k (усеченное SVD)."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def weighted_lp_prox_grad(A, W, k, p, max_iter=100, alpha=0.1, tol=1e-5):
    """
    Обобщенный метод для аппроксимации в Lp-норме.
    Минимизирует Σ w_ij |A_ij - X_ij|^p.
    Для 0 < p ≤ 2.
    """
    X = trunc_svd(A, k)
    errors = []

    for i in range(max_iter):
        R = A - X
        # Для p=1 используется sign, для p≠1 используется градиент |r|^(p-1) * sign(r)
        if p == 1:
            grad = W * np.sign(R)
        elif p < 2:
            # Для p<2 градиент включает p|r|^(p-1)sign(r)
            grad = p * W * (np.abs(R) ** (p - 1)) * np.sign(R)
        else:
            # Для p>2 градиент растет быстрее, используем адаптивный шаг
            grad = p * W * (np.abs(R) ** (p - 1)) * np.sign(R)
            alpha = max(0.01, alpha * 0.99)  # уменьшаем шаг

        # Антиградиентный шаг
        Y = X + alpha * grad
        # Проекция на ранг k
        Y = trunc_svd(Y, k)
        # Ошибка
        err = np.sum(W * (np.abs(A - Y) ** p))
        errors.append(err)

        if i > 0 and abs(errors[-2] - errors[-1]) < tol:
            break
        X = Y

    return X, errors


# Загрузка и подготовка данных
image_path = "test.jpg"  # Укажите путь к вашему файлу
img = Image.open(image_path).convert('L')
A = np.array(img, dtype=np.float64)

# Построение весов
edges = np.abs(sobel(A, axis=0)) + np.abs(sobel(A, axis=1))
W = 1 + 10 * (edges / edges.max())

# Параметры
k = 30
p_values = [0.5, 1, 2]
results = []

plt.figure(figsize=(20, 18))

for idx, p in enumerate(p_values):
    print(f"Расчет для p = {p}...")
    X_p, errors = weighted_lp_prox_grad(A, W, k, p, max_iter=150, alpha=0.3 if p <= 2 else 0.05)
    mse = np.mean((A - X_p) ** 2)
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    ssim_val = ssim(A, X_p, data_range=255)
    results.append({'p': p, 'psnr': psnr, 'ssim': ssim_val, 'X': X_p, 'errors': errors})

    plt.subplot(2, 4, idx + 1)
    plt.imshow(X_p, cmap='gray')
    plt.title(f'Lp norm, p={p}\nPSNR={psnr:.2f}dB, SSIM={ssim_val:.3f}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Сравнение метрик
print("\n=== Сводная таблица результатов ===")
print("p\tPSNR (дБ)\tSSIM")
for res in results:
    print(f"{res['p']}\t{res['psnr']:.2f}\t\t{res['ssim']:.4f}")

# График зависимости качества от p
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot([r['p'] for r in results], [r['psnr'] for r in results], 'bo-', label='PSNR')
ax1.set_xlabel('p')
ax1.set_ylabel('PSNR (дБ)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_xscale('log')
ax2 = ax1.twinx()
ax2.plot([r['p'] for r in results], [r['ssim'] for r in results], 'ro-', label='SSIM')
ax2.set_ylabel('SSIM', color='r')
ax2.tick_params(axis='y', labelcolor='r')
plt.title('Зависимость качества восстановления от параметра p')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()