import numpy as np
import time
import cProfile

def generate_image(size):
    return np.random.rand(size, size).astype(np.float32)


def gaussian_kernel(size=5, sigma=1.0):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)



# 3. Naive реализация (бавна)
def gaussian_blur_naive(image, kernel):
    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(image, pad, mode='constant')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + k, j:j + k]
            output[i, j] = np.sum(region * kernel)

    return output



# 4. Vectorized / optimized версия
def gaussian_blur_vectorized(image, kernel):
    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(image, pad, mode='constant')

    output = np.zeros_like(image)

    for i in range(k):
        for j in range(k):
            output += kernel[i, j] * padded[i:i + image.shape[0], j:j + image.shape[1]]

    return output


def run_naive_test(size):
    image = generate_image(size)
    kernel = gaussian_kernel(5, 1.0)

    start = time.time()
    result = gaussian_blur_naive(image, kernel)
    end = time.time()

    print(f"Size: {size}x{size}, Time: {end - start:.4f} sec")
    return result


def run_vectorized_test(size):
    image = generate_image(size)
    kernel = gaussian_kernel(5, 1.0)

    start = time.time()
    result = gaussian_blur_vectorized(image, kernel)
    end = time.time()

    print(f"Size: {size}x{size}, Time: {end - start:.4f} sec")
    return result


if __name__ == "__main__":
    print("Naive tests:")
    run_naive_test(512)
    run_naive_test(1024)

    print("\nVectorized Tests:")
    run_vectorized_test(512)
    run_vectorized_test(1024)

    print("\nProfiling (Navie 512x512):")
    cProfile.run("run_naive_test(512)")

    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from skimage.color import rgb2gray

    raw = imageio.imread("text.png")
    if raw.shape[2] == 4:
        raw = raw[:, :, :3]
    image = rgb2gray(raw)

    kernel = gaussian_kernel(5, 1.0)

    blur_naive = gaussian_blur_naive(image, kernel)
    blur_vectorized = gaussian_blur_vectorized(image, kernel)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Naive Blur")
    plt.imshow(blur_naive, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Optimized Blur")
    plt.imshow(blur_vectorized, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    plt.show()