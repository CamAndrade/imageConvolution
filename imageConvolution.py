from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

KERNEL_VERTICAL = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

KERNEL_HORIZONTAL = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]])


def filter_size(sigma):
    size = ((3 * int(sigma)) + 1)
    if size % 2 == 0:
        size += 1
    return size


def gaussian_distribuition_kernel(size, sigma):
    ind_x = 0
    kernel = np.zeros((size, size))
    for x in range(-(size // 2), (size // 2) + 1):
        ind_y = 0
        for y in range(-(size // 2), (size // 2) + 1):
            part1 = (1.0 / (2 * np.pi * sigma * sigma))
            part2 = (np.exp(- (x * x + y * y) / (2 * sigma * sigma)))
            kernel[ind_x][ind_y] = part1 * part2
            ind_y += 1
        ind_x += 1
    return kernel


def kernel_normalization(kernel):
    sum = np.sum(kernel)
    kernel /= sum
    return kernel


def convolution(kernel, img):
    row, column = img.shape
    row_kernel, column_kernel = kernel.shape
    conv = np.zeros((row, column))
    conv_aux = np.zeros((row_kernel, column_kernel))
    img = np.pad(img, row_kernel // 2)

    for i_img in tqdm(range(0, row)):
        for j_img in range(0, column):
            for i_kernel in range(0, row_kernel):
                for j_kernel in range(0, column_kernel):
                    conv_aux[i_kernel][j_kernel] = kernel[i_kernel, j_kernel] * img[i_img + i_kernel][j_img + j_kernel]
            conv[i_img][j_img] = np.sum(conv_aux)

    return conv


def magnitude(vertical, horizontal):
    return np.sqrt(np.power(vertical, 2) + np.power(horizontal, 2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageIn', '-i', required=True, help='Diretório da imagem de entrada')
    parser.add_argument('--filter', '-f', required=True, choices=['gaussian', 'sobel'], help='Filtro a ser utilizado = gaussian ou sobel.')
    parser.add_argument('--sigma', '-s',  default=False, choices=['1','2','3'], help='Valores de sigma para calcular o kernel.')

    args = parser.parse_args()
    if not args.sigma and args.filter == 'gaussian':
        raise argparse.ArgumentError("O parâmetro -f com a opção gaussian requer o parâmetro -s.")

    imageIn = args.imageIn
    filter = args.filter

    try:
        img = io.imread(imageIn)
    except:
        print('ERRO AO ABRIR A IMAGEM!')

    if filter == 'gaussian':
        sigma = int(args.sigma)
        size_kernel = filter_size(sigma)
        kernel = gaussian_distribuition_kernel(size_kernel, sigma)
        kernel = kernel_normalization(kernel)
        conv = convolution(kernel, img)

        fig, x = plt.subplots(1, 2)
        x[0].imshow(img, cmap="gray")
        x[1].imshow(conv, cmap="gray")
        plt.show()
    elif filter == 'sobel':
        vertical = convolution(KERNEL_VERTICAL, img)
        horizontal = convolution(KERNEL_HORIZONTAL, img)
        conv = magnitude(vertical, horizontal)

        fig, x = plt.subplots(1, 2)
        x[0].imshow(img, cmap="gray")
        x[1].imshow(conv, cmap="gray")
        plt.show()