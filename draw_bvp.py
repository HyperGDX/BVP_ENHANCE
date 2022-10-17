import scipy.io as scio
import matplotlib.pyplot as plt

file_path = "testdata/user1/user1-1-1-1-1-1-1e-07-100-20-100000-L0.mat"
data_file_name = "user1-1-1-1-1-1-1e-07-100-20-100000-L0.mat"
data_1 = scio.loadmat(file_path)['velocity_spectrum_ro']
label_1 = int(data_file_name.split('-')[1])


def draw_bvp(data, per_row_image=10):
    t_len = data.shape[-1]
    for i in range(t_len):
        plt.subplot(t_len // per_row_image + 1, per_row_image, i + 1)
        # plt.imshow(data[:, :, i], cmap='gray')
        plt.imshow(data[:, :, i])
        plt.axis('off')
    plt.show()


draw_bvp(data_1)
