train_dir = 'data/train_32x32.mat'
test_dir = 'data/test_32x32.mat'
model_weights = 'data/svhn_weights.h5'

size = 32 
n_channels = 3 
input_shape = (size, size, n_channels)

seed = 1234
val_split = 0.15
batch_size = 128
epochs = 1
verbose = 1
rescale = 255.0

dense1 = 1000
dense2 = 512
dense3 = 256
dense4 = 64