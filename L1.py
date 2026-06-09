import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import sobel, gaussian_filter


def trunc_svd(X, k):
    """Возвращает наилучшую аппроксимацию X ранга k в норме Фробениуса (SVD усечение)."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def weighted_l1_proximal_gradient(A, W, k, max_iter=100, alpha=0.01, tol=1e-5, verbose=False):
    """
    Итерационная аппроксимация матрицы A матрицей ранга k,
    минимизирующая взвешенную L1-норму ||W * (A - X)||_1.

    Параметры:
    A : исходная матрица (m x n)
    W : неотрицательные веса (m x n)
    k : целевой ранг
    max_iter : число итераций
    alpha : шаг градиента
    tol : допуск для остановки
    verbose : печать прогресса

    Возвращает:
    X : аппроксимирующая матрица ранга k
    errors : список значений целевой функции на каждой итерации
    """
    X = trunc_svd(A, k)  # инициализация SVD-аппроксимацией
    errors = []
    for it in range(max_iter):
        R = A - X
        # Субградиент взвешенной L1-нормы: W * sign(R)
        grad = W * np.sign(R)
        # Шаг градиента
        X_new = X + alpha * grad
        # Проекция на множество матриц ранга k (через SVD усечение)
        X_new = trunc_svd(X_new, k)
        # Вычисление значения целевой функции
        error = np.sum(W * np.abs(A - X_new))
        errors.append(error)
        # Проверка сходимости
        if it > 0 and abs(errors[-2] - errors[-1]) < tol:
            if verbose:
                print(f"Сошлось на итерации {it + 1}")
            break
        X = X_new
        if verbose and (it + 1) % 20 == 0:
            print(f"Итерация {it + 1}, ошибка = {error:.2f}")
    return X, errors


# Загрузка и предобработка изображения
img_path = "test.jpg"  # укажите свой путь
try:
    img = Image.open(img_path).convert('L')
except:
    # Создадим тестовое изображение, если файла нет
    img = Image.fromarray((np.random.rand(256, 256) * 255).astype(np.uint8))
A = np.array(img, dtype=np.float64)

# Построение весов W: высокие веса на границах (детектор Собеля)
edges = np.abs(sobel(A, axis=0)) + np.abs(sobel(A, axis=1))
W = 1 + 10 * (edges / edges.max())  # веса от 1 до 11, чем резче граница – тем выше вес

# Параметры
k = 30  # ранг аппроксимации
max_iter = 150
alpha = 0.2

# Запуск взвешенного L1 метода
X_l1, errors = weighted_l1_proximal_gradient(A, W, k, max_iter=max_iter, alpha=alpha, verbose=True)

# Классическое SVD усечение
X_svd = trunc_svd(A, k)

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(A, cmap='gray')
axes[0, 0].set_title("Оригинал")
axes[0, 0].axis('off')

axes[0, 1].imshow(X_svd, cmap='gray')
axes[0, 1].set_title(f"SVD, ранг={k}")
axes[0, 1].axis('off')

axes[0, 2].imshow(X_l1, cmap='gray')
axes[0, 2].set_title(f"Взвешенная L1, ранг={k}")
axes[0, 2].axis('off')

# Разности
diff_svd = np.abs(A - X_svd)
diff_l1 = np.abs(A - X_l1)
axes[1, 0].imshow(diff_svd, cmap='hot')
axes[1, 0].set_title("Ошибка SVD")
axes[1, 0].axis('off')

axes[1, 1].imshow(diff_l1, cmap='hot')
axes[1, 1].set_title("Ошибка L1 (взв.)")
axes[1, 1].axis('off')

# График сходимости
axes[1, 2].semilogy(errors)
axes[1, 2].set_title("Сходимость целевой функции")
axes[1, 2].set_xlabel("Итерация")
axes[1, 2].set_ylabel("Взвешенная L1 ошибка")
axes[1, 2].grid(True)

plt.tight_layout()
plt.show()

# Сравнительные метрики (по взвешенной L1, которая минимизируется – она должна быть лучше у L1-метода)
err_svd = np.sum(W * np.abs(A - X_svd))
err_l1 = np.sum(W * np.abs(A - X_l1))
print(f"Взвешенная L1-ошибка SVD: {err_svd:.2f}")
print(f"Взвешенная L1-ошибка L1 метода: {err_l1:.2f}")

# Также обычные метрики
from sklearn.metrics import mean_squared_error

mse_svd = mean_squared_error(A.ravel(), X_svd.ravel())
mse_l1 = mean_squared_error(A.ravel(), X_l1.ravel())
print(f"MSE SVD: {mse_svd:.2f}, MSE L1: {mse_l1:.2f}")
psnr_svd = 20 * np.log10(255) - 10 * np.log10(mse_svd)
psnr_l1 = 20 * np.log10(255) - 10 * np.log10(mse_l1)
print(f"PSNR (дБ): SVD = {psnr_svd:.2f}, L1 = {psnr_l1:.2f}")

# SSIM (если есть scikit-image)
try:
    from skimage.metrics import structural_similarity as ssim

    ssim_svd = ssim(A, X_svd, data_range=255)
    ssim_l1 = ssim(A, X_l1, data_range=255)
    print(f"SSIM: SVD = {ssim_svd:.4f}, L1 = {ssim_l1:.4f}")
except ImportError:
    print("Для SSIM установите scikit-image: pip install scikit-image")