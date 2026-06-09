import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import sobel, gaussian_filter

def trunc_svd(X, k):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

def weighted_l2_prox_grad(A, W, k, max_iter=100, alpha=0.1, tol=1e-5, verbose=False):
    X = trunc_svd(A, k)
    errors = []
    for it in range(max_iter):
        R = A - X
        grad = -2 * W * R          # градиент по X
        Y = X - alpha * grad       # шаг антиградиента
        Y = trunc_svd(Y, k)
        err = np.sum(W * (A - Y)**2)
        errors.append(err)
        if it > 0 and abs(errors[-2] - errors[-1]) < tol:
            if verbose:
                print(f"L2 сошлось на итерации {it+1}")
            break
        X = Y
        if verbose and (it+1) % 20 == 0:
            print(f"L2 iter {it+1}, err={err:.2f}")
    return X, errors

def weighted_l1_prox_grad(A, W, k, max_iter=100, alpha=0.2, tol=1e-5, verbose=False):

    X = trunc_svd(A, k)
    errors = []
    for it in range(max_iter):
        R = A - X
        # антиградиент для L1: W * sign(R)
        grad = W * np.sign(R)
        Y = X + alpha * grad      # шаг (знак +, т.к. это уже антиградиент)
        Y = trunc_svd(Y, k)
        err = np.sum(W * np.abs(A - Y))
        errors.append(err)
        if it > 0 and abs(errors[-2] - errors[-1]) < tol:
            if verbose:
                print(f"L1 сошлось на итерации {it+1}")
            break
        X = Y
        if verbose and (it+1) % 20 == 0:
            print(f"L1 iter {it+1}, err={err:.2f}")
    return X, errors

# ========== 1. Загрузка изображения ==========
image_path = "test.jpg"   # Укажите путь к вашему файлу
try:
    img = Image.open(image_path).convert('L')
    A = np.array(img, dtype=np.float64)
    print(f"Изображение загружено: размер {A.shape}")
except FileNotFoundError:
    print(f"Файл {image_path} не найден.")
    exit()

# Высокие веса на границах (детектор Собеля)
edges = np.abs(sobel(A, axis=0)) + np.abs(sobel(A, axis=1))
# Нормализуем и задаём диапазон [1, 11]
W = 1 + 10 * (edges / edges.max())
# Дополнительно можно сгладить веса, чтобы не было слишком резких переходов
W = gaussian_filter(W, sigma=1.0)

k = 30                      # ранг аппроксимации (степень сжатия)
max_iter = 100
alpha_l2 = 0.1
alpha_l1 = 0.3

print("\n--- Обычное SVD ---")
X_svd = trunc_svd(A, k)

print("--- Взвешенная L2 ---")
X_l2, err_l2 = weighted_l2_prox_grad(A, W, k, max_iter, alpha_l2, verbose=True)

print("--- Взвешенная L1 ---")
X_l1, err_l1 = weighted_l1_prox_grad(A, W, k, max_iter, alpha_l1, verbose=True)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes[0,0].imshow(A, cmap='gray')
axes[0,0].set_title("Оригинал")
axes[0,0].axis('off')

axes[0,1].imshow(X_svd, cmap='gray')
axes[0,1].set_title(f"SVD (L2, без весов)\nrank={k}")
axes[0,1].axis('off')

axes[0,2].imshow(X_l2, cmap='gray')
axes[0,2].set_title(f"Взвешенная L2\nrank={k}")
axes[0,2].axis('off')

axes[0,3].imshow(X_l1, cmap='gray')
axes[0,3].set_title(f"Взвешенная  L1\nrank={k}")
axes[0,3].axis('off')

diff_svd = np.abs(A - X_svd)
diff_l2 = np.abs(A - X_l2)
diff_l1 = np.abs(A - X_l1)

axes[1,0].imshow(diff_svd, cmap='hot')
axes[1,0].set_title("Ошибка SVD")
axes[1,0].axis('off')

axes[1,1].imshow(diff_l2, cmap='hot')
axes[1,1].set_title("Ошибка L2")
axes[1,1].axis('off')

axes[1,2].imshow(diff_l1, cmap='hot')
axes[1,2].set_title("Ошибка L1")
axes[1,2].axis('off')

# График сходимости
axes[1,3].plot(err_l2, label='Веса L2', linewidth=2)
axes[1,3].plot(err_l1, label='Веса L1', linewidth=2)
axes[1,3].set_yscale('log')
axes[1,3].set_title("Сходимость целевых функций")
axes[1,3].set_xlabel("Итерация")
axes[1,3].set_ylabel("Ошибка")
axes[1,3].legend()
axes[1,3].grid(True)

plt.tight_layout()
plt.show()

def weighted_l2_err(A, X, W):
    return np.sum(W * (A - X)**2)

def weighted_l1_err(A, X, W):
    return np.sum(W * np.abs(A - X))

w2_svd = weighted_l2_err(A, X_svd, W)
w2_l2 = weighted_l2_err(A, X_l2, W)
w2_l1 = weighted_l2_err(A, X_l1, W)

w1_svd = weighted_l1_err(A, X_svd, W)
w1_l2 = weighted_l1_err(A, X_l2, W)
w1_l1 = weighted_l1_err(A, X_l1, W)

print("\n=== Взвешенная L2 ошибка (чем меньше, тем лучше для L2) ===")
print(f"SVD:       {w2_svd:.2f}")
print(f"Weighted L2: {w2_l2:.2f}")
print(f"Weighted L1: {w2_l1:.2f}")

print("\n=== Взвешенная L1 ошибка (чем меньше, тем лучше для L1) ===")
print(f"SVD:       {w1_svd:.2f}")
print(f"Веса L2: {w1_l2:.2f}")
print(f"Веса L1: {w1_l1:.2f}")

# Обычный MSE и PSNR
mse_svd = np.mean((A - X_svd)**2)
mse_l2 = np.mean((A - X_l2)**2)
mse_l1 = np.mean((A - X_l1)**2)
psnr_svd = 20*np.log10(255) - 10*np.log10(mse_svd)
psnr_l2 = 20*np.log10(255) - 10*np.log10(mse_l2)
psnr_l1 = 20*np.log10(255) - 10*np.log10(mse_l1)

print(f"\nPSNR (дБ): SVD = {psnr_svd:.2f}, L2 = {psnr_l2:.2f}, L1 = {psnr_l1:.2f}")

# SSIM
try:
    from skimage.metrics import structural_similarity as ssim
    # Для SSIM значения в диапазоне 0..255
    ssim_svd = ssim(A, X_svd, data_range=255)
    ssim_l2 = ssim(A, X_l2, data_range=255)
    ssim_l1 = ssim(A, X_l1, data_range=255)
    print(f"SSIM: SVD = {ssim_svd:.4f}, L2 = {ssim_l2:.4f}, L1 = {ssim_l1:.4f}")
except ImportError:
    print("Для SSIM установите scikit-image: pip install scikit-image")