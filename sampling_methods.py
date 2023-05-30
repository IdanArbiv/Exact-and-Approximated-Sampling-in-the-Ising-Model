import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def G(row_s, temp):
    return np.exp(1 / temp * np.sum(row_s[:-1] * row_s[1:]))


def F(row_s, row_t, temp):
    return np.exp(1 / temp * np.sum(row_s * row_t))


def c_e_5():
    z_temp = []
    for temp in [1, 1.5, 2]:
        sum_ = 0
        for y_1, y_2 in product(range(4), repeat=2):
            y_1_row = y2row(y_1, 2)
            y_2_row = y2row(y_2, 2)
            sum_ += G(y_2_row, temp) * G(y_1_row, temp) * F(y_1_row, y_2_row, temp)
        z_temp.append(sum_)
    return z_temp


def c_e_6():
    z_temp = []
    for temp in [1, 1.5, 2]:
        sum_ = 0
        for y_1, y_2, y_3 in product(range(8), repeat=3):
            y_1_row = y2row(y_1, 3)
            y_2_row = y2row(y_2, 3)
            y_3_row = y2row(y_3, 3)
            sum_ += G(y_2_row, temp) * G(y_1_row, temp) * G(y_3_row, temp) * F(y_1_row, y_2_row, temp) * F(y_2_row,
                                                                                                           y_3_row,
                                                                                                           temp)
        z_temp.append(sum_)
    return z_temp


def dynamic_forward(lattice_size, t_values, temp):
    for i in range(1, lattice_size + 1):
        t_i = []
        if i == 1:
            for y_2 in range(2 ** lattice_size):
                t_i.append(
                    sum([G(y2row(y_1, lattice_size), temp) * F(y2row(y_1, lattice_size), y2row(y_2, lattice_size), temp)
                         for y_1 in range(2 ** lattice_size)]))
        elif i == lattice_size:
            t_i.append(sum([G(y2row(y_last, lattice_size), temp) * t_values[i - 2][y_last] for y_last in
                            range(2 ** lattice_size)]))
        else:
            for y_i_plus_one in range(2 ** lattice_size):
                t_i.append(sum([G(y2row(y_i, lattice_size), temp) * F(y2row(y_i, lattice_size),
                                                                      y2row(y_i_plus_one, lattice_size), temp) *
                                t_values[i - 2][y_i] for y_i in range(2 ** lattice_size)]))

        t_values.append(t_i)
    return t_values


def dynamic_backward(lattice_size, t_values, p_values, temp):
    for i in range(lattice_size, 0, -1):
        p_i = []
        if i == lattice_size:
            for y_last in range(2 ** lattice_size):
                p_i.append(t_values[i - 2][y_last] * G(y2row(y_last, lattice_size), temp) / t_values[-1][0])
        elif i == 1:
            for y_2 in range(2 ** lattice_size):
                p_i_row = []
                for y_1 in range(2 ** lattice_size):
                    p_i_row.append(
                        F(y2row(y_1, lattice_size), y2row(y_2, lattice_size), temp) * G(y2row(y_1, lattice_size),
                                                                                        temp) / t_values[0][y_2])
                p_i.append(p_i_row)
        else:
            for y_k_plus_one in range(2 ** lattice_size):
                p_i_row = []
                for y_k in range(2 ** lattice_size):
                    p_i_row.append(
                        t_values[i - 2][y_k] * F(y2row(y_k, lattice_size), y2row(y_k_plus_one, lattice_size), temp) * G(
                            y2row(y_k, lattice_size), temp) / t_values[i - 1][y_k_plus_one])
                p_i.append(p_i_row)

        p_values.insert(0, p_i)
    return p_values


def sample_y(lattice_size, p_values):
    y_values = []
    for i in range(lattice_size, 0, -1):
        if i == lattice_size:
            y_i = np.random.choice(a=[i for i in range(2 ** lattice_size)], p=p_values[-1])
        else:
            y_i = np.random.choice(a=[i for i in range(2 ** lattice_size)], p=p_values[i - 1][y_values[0]])
        y_values.insert(0, y_i)
    return y_values


def convert_y_to_image(y_values, lattice_size):
    return [y2row(y, lattice_size) for y in y_values]


def generate_distribution(lattice_size, temp):
    t_values = []
    p_values = []
    t_values = dynamic_forward(lattice_size, t_values, temp)
    return dynamic_backward(lattice_size, t_values, p_values, temp)


def sample_image(lattice_size, p_values):
    y_values = sample_y(lattice_size, p_values)
    return convert_y_to_image(y_values, lattice_size)


def c_e_7():
    temp_list = [1, 1.5, 2.0]
    num_samples = 10
    lattice_size = 8
    plot_multiple_images(num_samples, generate_images(lattice_size, temp_list, num_samples), temp_list)


def generate_images(lattice_size, temp_list, num_samples):
    images = []
    for temp in temp_list:
        p_values = generate_distribution(lattice_size, temp)
        for i in range(num_samples):
            images.append(sample_image(lattice_size, p_values))
    return images


def plot_multiple_images(num_samples, images, temp_list):
    # Plot the images
    fig, axs = plt.subplots(nrows=3, ncols=num_samples, figsize=(12, 6))
    for i in range(len(images)):
        row = i // num_samples
        col = i % num_samples
        axs[row, col].imshow(images[i], interpolation='none', cmap='gray', vmin=-1, vmax=1)
        axs[row, col].axis('off')
        axs[row, col].set_title(f"Temp={temp_list[row]}")
    plt.tight_layout()
    plt.show()


def y2row(y, width=8):
    """
    y: an integer in (0,...,(2**width)-1)
    """
    if not 0 <= y <= (2 ** width) - 1:
        raise ValueError(y)
    my_str = np.binary_repr(y, width=width)
    my_list = list(map(int, my_str))
    my_array = np.asarray(my_list)
    my_array[my_array == 0] = -1
    row = my_array
    return row


def compute_ztemp_3x3():
    z_temp = {}
    for temp in [1, 1.5, 2]:
        values = product([1, -1], repeat=9)
        sum_ = sum([np.exp(1 / temp * (
                x1_1 * x1_2 + x1_2 * x1_3 + x2_1 * x2_2 + x2_2 * x2_3 + x3_1 * x3_2 + x3_2 * x3_3 + x2_1 * x1_1 + x2_1 * x3_1 + x2_2 * x1_2 + x2_2 * x3_2 + x2_3 * x1_3 + x2_3 * x3_3))
                    for x1_1, x1_2, x1_3, x2_1, x2_2, x2_3, x3_1, x3_2, x3_3 in values])
        z_temp[temp] = sum_
    return z_temp


def compute_ztemp_2x2():
    z_temp = {}
    for temp in [1, 1.5, 2]:
        values = product([1, -1], repeat=4)
        sum_ = sum(
            [np.exp(1 / temp * (x1_1 * x1_2 + x1_1 * x2_1 + x1_2 * x2_2 + x2_1 * x2_2)) for x1_1, x1_2, x2_1, x2_2 in
             values])
        z_temp[temp] = sum_
    return z_temp


def c_e_8(lattice_size):
    results = {}
    for temp in [1, 1.5, 2]:
        curr_temp_results = []
        e_mean_11_22 = 0
        e_mean_11_88 = 0
        p_values = generate_distribution(lattice_size, temp)
        for i in range(1, 10_001):
            image = sample_image(lattice_size, p_values)
            e_mean_11_22 += image[0][0] * image[1][1]
            e_mean_11_88 += image[0][0] * image[7][7]
        e_mean_11_22 /= 10_000
        e_mean_11_88 /= 10_000
        results[temp] = curr_temp_results
        print("Method: {}, Temp: {} , E(X_11, X_22) = {} , E(X_11, X_88) = {}".format("Dynamic Programming", temp,
                                                                                      e_mean_11_22, e_mean_11_88))
    return results


def sample_8x8_lattice_using_fair_coin():
    return np.random.randint(low=0, high=2, size=(8, 8)) * 2 - 1


def sample_100x100_lattice_using_fair_coin():
    return np.random.randint(low=0, high=2, size=(100, 100)) * 2 - 1


def sweep_gibbs_sampling(lattice_x, Temp, lattice_y=np.zeros((8,8)), sigma=0):
    rows, cols = lattice_x.shape
    distribution_matrix = distribution_matrix = np.empty((rows, cols), dtype=np.ndarray)
    for i in range(rows):
        for j in range(cols):
            neighbor_sum = 0
            if i > 0:
                neighbor_sum += lattice_x[i - 1][j]  # upper neighbor
            if j > 0:
                neighbor_sum += lattice_x[i][j - 1]  # left neighbor
            if i < rows - 1:
                neighbor_sum += lattice_x[i + 1][j]  # lower neighbor
            if j < cols - 1:
                neighbor_sum += lattice_x[i][j + 1]  # right neighbor
            if sigma == 0:
                prob1 = np.exp(1 / Temp * neighbor_sum)
                prob2 = np.exp(-1 / Temp * neighbor_sum)
            else:
                prob1 = np.exp((1 / Temp * neighbor_sum) - 1 / sigma ** 2 * (lattice_y[i][j] - 1) ** 2)
                prob2 = np.exp((-1 / Temp * neighbor_sum) - 1 / sigma ** 2 * (lattice_y[i][j] + 1) ** 2)

            new_entry_value = np.random.choice([1, -1], p=[prob1 / (prob1 + prob2), prob2 / (prob1 + prob2)])
            lattice_x[i][j] = new_entry_value
            distribution_matrix[i][j] = np.array([prob1, prob2])

    return lattice_x, distribution_matrix


def gibbs_sampling(lattice, temp, sweep_num, lattice_y=np.zeros((8, 8)), sigma=0):
    for sweep in range(sweep_num):
        lattice, distribution = sweep_gibbs_sampling(lattice, temp, lattice_y, sigma )
    return lattice, distribution


def c_e_9_independent_samples():
    results = {}
    for temp in [1, 1.5, 2]:
        curr_temp_results = []
        x11_x22_sum = 0
        x11_x88_sum = 0
        for sample in range(1, 10_001):
            lattice, _ = gibbs_sampling(sample_8x8_lattice_using_fair_coin(), temp, 25)
            x11_x22_sum += lattice[0][0] * lattice[1][1]
            x11_x88_sum += lattice[0][0] * lattice[7][7]
        e_mean_11_22 = x11_x22_sum / 10_000
        e_mean_11_88 = x11_x88_sum / 10_000
        curr_temp_results.append(e_mean_11_22)
        curr_temp_results.append(e_mean_11_88)
        results[temp] = curr_temp_results
        print("Method: {}, Temp: {} , E(X_11, X_22) = {} , E(X_11, X_88) = {}".format("Independent Samples", temp,
                                                                                      e_mean_11_22, e_mean_11_88))
    return results


def create_e_digits_num(a,b,c):
    return  -1 if (a.isdigit() and b.isdigit() and c.isdigit()) else  a+b+c

def gibbs_sampling_with_sweep_calculations(temp, sweep_num):
    e_mean_11_22 = 0
    e_mean_11_88 = 0
    lattice = sample_8x8_lattice_using_fair_coin()
    for sweep in range(sweep_num):
        lattice, _ = sweep_gibbs_sampling(lattice, temp)
        if sweep > 100:
            e_mean_11_22 += lattice[0][0] * lattice[1][1]
            e_mean_11_88 += lattice[0][0] * lattice[7][7]
    return lattice, e_mean_11_22 / 24_900, e_mean_11_88 / 24_900


def c_e_9_ergodicity():
    results = {}
    for temp in [1, 1.5, 2]:
        curr_temp_results = []
        _, e_mean_11_22, e_mean_11_88 = gibbs_sampling_with_sweep_calculations(temp, 25_000)
        curr_temp_results.append(e_mean_11_22)
        curr_temp_results.append(e_mean_11_88)
        results[temp] = curr_temp_results
        print("Method: {}, Temp: {} , E(X_11, X_22) = {} , E(X_11, X_88) = {}".format("Ergodicity", temp, e_mean_11_22,
                                                                                      e_mean_11_88))
    return results


def c_e_9():
    c_e_9_independent_samples()
    c_e_9_ergodicity()


def ICM(lattice, distribution):
    rows, cols = lattice.shape
    continue_flag = True
    while continue_flag:
        continue_flag = False
        for i in range(rows):
            for j in range(cols):
                new_value = [1, -1][np.argmax(distribution[i][j])]
                if new_value != lattice[i][j]:
                    continue_flag = True
                    lattice[i][j] = new_value
    return lattice


def MLE(lattice_y):
    new_lattice = np.sign(lattice_y)
    zero_indices = np.where(new_lattice == 0)
    num_zeros = len(zero_indices[0])
    zero_signs = np.random.choice([-1, 1], size=num_zeros, p=[0.5, 0.5])
    new_lattice[zero_indices] = zero_signs
    return new_lattice


def plot_images(temp, img_array, label_array):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(img_array[i], cmap='gray', vmin=-1, vmax=1) if label_array[i] != 'y' else axes[i].imshow(
            img_array[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(label_array[i])
    plt.subplots_adjust(wspace=1)
    plt.suptitle("Temp: {}".format(temp))
    plt.show()


def c_e_10():
    init = sample_100x100_lattice_using_fair_coin()
    for temp in [1, 1.5, 2]:
        lattice_x, _ = gibbs_sampling(init, temp, 50)
        gauss_noise = 2 * np.random.standard_normal(size=(100, 100))
        lattice_y = lattice_x + gauss_noise
        lattice_x_from_gibbs_sample, distribution = gibbs_sampling(sample_100x100_lattice_using_fair_coin(), temp, 50, lattice_y, 4)
        lattice_x_from_ICM = ICM(sample_100x100_lattice_using_fair_coin(), distribution)
        lattice_x_from_MLE = MLE(lattice_y)
        img_array = [lattice_x, lattice_y, lattice_x_from_gibbs_sample, lattice_x_from_ICM, lattice_x_from_MLE]
        label_array = ["Original Lattice", "Lattice with Gaussian noise", "Recon using Gibbs Sampling\nwith noisy image", "Recon using ICM", "Recon using MLE"]
        plot_images(temp, img_array, label_array)


if __name__ == '__main__':
    c_e_10()
