from model.cvae_tf import CVAE, compute_loss, log_normal_pdf, train_step
from model.myrnn import assemble_model
from read_bvp import load_data, onehot_encoding
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from IPython import display
import imageio.v2 as imageio
import glob

LATENT_DIM = 32
data_dir = 'testdata'
ALL_MOTION = [1, 2, 3, 4, 5, 6]
N_MOTION = len(ALL_MOTION)
T_MAX = 0
fraction_for_test = 0.1

data, label, t_max = load_data(path_to_data=data_dir, motion_sel=ALL_MOTION)
label_train = onehot_encoding(label, N_MOTION)
test_size = int(label.shape[0]*fraction_for_test)
train_size = label.shape[0] - test_size
batch_size = 32
[data_train, data_test] = train_test_split(data, test_size=fraction_for_test)

train_dataset = (tf.data.Dataset.from_tensor_slices(data_train)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(data_test)
                .shuffle(test_size).batch(batch_size))

num_examples_to_generate = 8
cvae_model = CVAE(LATENT_DIM, t_max)
epochs = 200
optimizer = tf.keras.optimizers.Adam(1e-3)


# def generate_and_save_images(model, epoch, test_sample, t_max):
#     mean, logvar = model.encode(test_sample)
#     z = model.reparameterize(mean, logvar)
#     predictions = model.sample(z)  # 8,T,20,20,1
#     for i in range(predictions.shape[0]):
#         for j in range(predictions.shape[1]):
#             plt.subplot(8, t_max, i*t_max + j+1)
#             plt.imshow(predictions[i, j, :, :], cmap='gray')
#             plt.axis('off')

#     # tight_layout minimizes the overlap between 2 sub-plots
#     plt.savefig('cvae_res/image_at_epoch_{:04d}.png'.format(epoch))
#     # plt.show()


# def draw_raw(test_sample, t_max):
#     for i in range(test_sample.shape[0]):
#         for j in range(test_sample.shape[1]):
#             plt.subplot(8, t_max, i*t_max + j+1)
#             plt.imshow(test_sample[i, j, :, :], cmap='gray')
#             plt.axis('off')
#     plt.savefig('cvae_res/image_at_epoch_0000.png')


# # Pick a sample of the test set for generating output images
# assert batch_size >= num_examples_to_generate
# for test_batch in test_dataset.take(1):
#     test_sample = test_batch[0:num_examples_to_generate, :, :, :, :]

# draw_raw(test_sample, t_max)
best_elbo = -100000
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        train_step(cvae_model, train_x, optimizer)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(compute_loss(cvae_model, test_x))
    elbo = -loss.result()

    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
          .format(epoch, elbo, end_time - start_time))
    if elbo > best_elbo & epoch > 50:
        tf.saved_model.save(cvae_model, "model/cvae_model_save.pt")
    # generate_and_save_images(model=cvae_model, epoch=epoch, test_sample=test_sample, t_max=t_max)


# anim_file = 'cvae_res/cvae.gif'

# with imageio.get_writer(anim_file, mode='I') as writer:
#     filenames = glob.glob('cvae_res/image*.png')
#     filenames = sorted(filenames)
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)
#     image = imageio.imread(filename)
#     writer.append_data(image)
