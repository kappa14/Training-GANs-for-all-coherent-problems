"""An implementation of the improved WGAN described in https://arxiv.org/abs/1704.00028
The improved WGAN has a term in the loss function which penalizes the network if its gradient
norm moves away from 1. This is included because the Earth Mover (EM) distance used in WGANs is only easy
to calculate for 1-Lipschitz functions (i.e. functions where the gradient norm has a constant upper bound of 1).
The original WGAN paper enforced this by clipping weights to very small values [-0.01, 0.01]. However, this
drastically reduced network capacity. Penalizing the gradient norm is more natural, but this requires
second-order gradients. These are not supported for some tensorflow ops (particularly MaxPool and AveragePool)
in the current release (1.0.x), but they are supported in the current nightly builds (1.1.0-rc1 and higher).
To avoid this, this model uses strided convolutions instead of Average/Maxpooling for downsampling. If you wish to use
pooling operations in your discriminator, please ensure you update Tensorflow to 1.1.0-rc1 or higher. I haven't
tested this with Theano at all.
The model saves images using pillow. If you don't have pillow, either install it or remove the calls to generate_images.
"""
import argparse
import os, pickle
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
# from tensorflow.keras.layers.merge import Concatenate
from keras.layers import Conv2D, Conv2DTranspose
# from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras.datasets import mnist, cifar10
from keras.optimizers import Adam, RMSprop, SGD, Adagrad
from keras.layers.merge import _Merge
from keras import backend as K
from functools import partial
from optimizer import *
from os.path import exists
from os import makedirs
#import tensorflow as tf


try:
    from PIL import Image
except ImportError:
    print('This script depends on pillow! Please install it (e.g. with pip install pillow)')
    exit()

BATCH_SIZE = 64


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty


def make_generator():
    """Creates a generator model that takes a 100-dimensional noise vector as a "seed", and outputs images
    of size 28x28x1."""
    model = Sequential()
    model.add(Dense(1024, input_dim=100))
    model.add(LeakyReLU())
    model.add(Dense(128 * 8 * 8))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    if K.image_data_format() == 'channels_first':
        model.add(Reshape((128, 8, 8), input_shape=(128 * 8 * 8,)))
        bn_axis = 1
    else:
        model.add(Reshape((8, 8, 128), input_shape=(128 * 8 * 8,)))
        bn_axis = -1
    model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    # Because we normalized training inputs to lie in the range [-1, 1],
    # the tanh function should be used for the output of the generator to ensure its output
    # also lies in this range.
    model.add(Conv2D(3, (5, 5), padding='same', activation='tanh'))
    return model


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
    the input is real or generated. Unlike normal GANs, the output is not sigmoid and does not represent a probability!
    Instead, the output should be as large and negative as possible for generated inputs and as large and positive
    as possible for real inputs.
    Note that the improved WGAN paper suggests that BatchNormalization should not be used in the discriminator."""
    model = Sequential()
    if K.image_data_format() == 'channels_first':
        model.add(Conv2D(64, (5, 5), padding='same', input_shape=(3, 32, 32)))
    else:
        model.add(Conv2D(64, (5, 5), padding='same', input_shape=(32, 32, 3)))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer='he_normal'))
    model.add(LeakyReLU())
    model.add(Dense(1, kernel_initializer='he_normal'))
    return model


def relu_block(inputs):
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(downsample, filters, kernel_size: int = 3):
    
    y = Conv2D(kernel_size=kernel_size,
               strides= 2,
               filters=filters,
               padding="same")(x)
    y = relu_block(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=2,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(y)
    out = Add()([x, y])
    out = relu_block(out)
    return out


def make_resnet_generator():
    
    model = Sequential()
    if K.image_data_format() == 'channels_first':
        model.add(Conv2D(64, (5, 5), padding='same', input_shape=(3, 32, 32)))
    else:
        model.add(Conv2D(64, (5, 5), padding='same', input_shape=(32, 32, 3)))
    
    model.add(residual_block(downsample=False, filters=128))
    model.add(residual_block(downsample=False, filters=128))
    model.add(residual_block(downsample=False, filters=128))
    model.add(LeakyReLU())
    model.add(Conv2d(filters=128, 3, 3, padding=1))
    model.add(LeakyReLU())

    return model


def make_resnet_discriminator():
    
    model = Sequential()
    if K.image_data_format() == 'channels_first':
        model.add(Conv2D(64, (5, 5), padding='same', input_shape=(3, 32, 32)))
    else:
        model.add(Conv2D(64, (5, 5), padding='same', input_shape=(32, 32, 3)))
    
    model.add(residual_block(downsample=True, filters=128))
    model.add(residual_block(downsample=True, filters=128))
    model.add(residual_block(downsample=True, filters=128))
    model.add(LeakyReLU())
    model.add(Conv2d(filters=128, 3, 3, padding=1))
    model.add(LeakyReLU())

    return model


def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for display."""
    assert len(image_stack.shape) == 4
    image_list = [image_stack[i, :, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        #weights = K.random_uniform(K.shape(inputs)[:1])
        #weights = K.reshape(weights, (-1,1,1,1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def generate_images(generator_model, output_dir, epoch):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG file."""
    test_image_stack = generator_model.predict(np.random.rand(100, 100))
    test_image_stack = (test_image_stack * 127.5) + 127.5
    test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))
    tiled_output = tile_images(test_image_stack)
    tiled_output = Image.fromarray(tiled_output)  # L specifies greyscale
    outfile = os.path.join(output_dir, 'epoch_{}.png'.format(epoch))
    tiled_output.save(outfile)
    outfile = os.path.join(output_dir, 'epoch_{}.pkl'.format(epoch))
    with open(outfile, 'wb') as f:
        pickle.dump(test_image_stack, f)



parser = argparse.ArgumentParser(description="Improved Wasserstein GAN implementation for Keras.")
parser.add_argument("--output_dir", "-o", required=True, help="Directory to output generated files to")
parser.add_argument("--lr", dest='optimizer_lr', default=1e-4, type=float, help="Directory to output generated files to")
parser.add_argument("--momentum", default=0, type=float)
parser.add_argument("--nesterov", action='store_true')
parser.add_argument("--optimizer", required=True)
parser.add_argument('--model', choices=('resnet', 'std_conv'), default='std_conv')
parser.add_argument("--schedule", default=None)
parser.add_argument("--gp", default=10.0, type=float)
parser.add_argument("--beta_1", default=0.5, type=float)
parser.add_argument("--beta_2", default=0.9, type=float)
parser.add_argument("-v", dest='version', type=int)
parser.add_argument("--retrain_dir", default=None)
parser.add_argument("--retrain_epoch", default=-1, type=int)
parser.add_argument("--training_ratio", default=5, type=int)

args = parser.parse_args()
if not exists(args.output_dir):
    makedirs(args.output_dir)

GRADIENT_PENALTY_WEIGHT = args.gp  # As per the paper
TRAINING_RATIO = args.training_ratio  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
# First we load the image data, reshape it and normalize it to the range [-1, 1]
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = np.concatenate((X_train, X_test), axis=0)
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape((X_train.shape[0], 3, X_train.shape[1], X_train.shape[2]))
else:
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 3))
X_train = (X_train.astype(np.float32) - 127.5) / 127.5

# Now we initialize the generator and discriminator.
if args.model == 'std_conv':
    generator = make_generator()
    discriminator = make_discriminator()
elif args.model == 'resnet':
    generator = make_resnet_generator()
    discriminator = make_resnet_discriminator()
    
if args.retrain_dir:
    generator.load_weights(os.path.join(args.retrain_dir, 'epoch_{}_g.h5'.format(args.retrain_epoch)))
    discriminator.load_weights(os.path.join(args.retrain_dir, 'epoch_{}_d.h5'.format(args.retrain_epoch)))

# The generator_model is used when we want to train the generator layers.
# As such, we ensure that the discriminator layers are not trainable.
# Note that once we compile this model, updating .trainable will have no effect within it. As such, it
# won't cause problems if we later set discriminator.trainable = True for the discriminator_model, as long
# as we compile the generator_model first.
for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
generator_input = Input(shape=(100,))
generator_layers = generator(generator_input)
discriminator_layers_for_generator = discriminator(generator_layers)
generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])

if args.schedule not in [None, 'adagrad', 'adam']:
    raise ValueError('schedule {} not recognized!'.format(args.schedule))

if args.optimizer == 'SGD':
    if args.schedule == 'adam':
        d_optim = Adam(0.0001, beta_1=0.5, beta_2=0.9)
        g_optim = Adam(0.0001, beta_1=0.5, beta_2=0.9)
    else:
        d_optim = Adagrad(lr=args.optimizer_lr) if args.schedule == 'adagrad' else SGD(lr=args.optimizer_lr, momentum=args.momentum, nesterov=args.nesterov)
        g_optim = Adagrad(lr=args.optimizer_lr) if args.schedule == 'adagrad' else SGD(lr=args.optimizer_lr, momentum=args.momentum, nesterov=args.nesterov)
else:
    optim_mapper = {
            'OFRL': OFRL,
            'OMDA': OMDA,
            'optimAdam': optimAdam,
            'extraOptimAdam': ExtraoptimAdam,
            'optimAdagrad': optimAdagrad,
            }
    if args.optimizer == 'optimAdam':
        d_optim = optim_mapper[args.optimizer](lr=args.optimizer_lr, beta_1=args.beta_1, beta_2=args.beta_2)
        g_optim = optim_mapper[args.optimizer](lr=args.optimizer_lr, beta_1=args.beta_1, beta_2=args.beta_2)
        print("Pass3")
    elif args.optimizer == 'extraOptimAdam':
        d_optim = optim_mapper[args.optimizer](lr=args.optimizer_lr, beta_1=args.beta_1, beta_2=args.beta_2)
        g_optim = optim_mapper[args.optimizer](lr=args.optimizer_lr, beta_1=args.beta_1, beta_2=args.beta_2)
        print("Pass4")
    elif args.optimizer == 'optimAdagrad':
        d_optim = optim_mapper[args.optimizer](lr=args.optimizer_lr)
        g_optim = optim_mapper[args.optimizer](lr=args.optimizer_lr)
    else:
        d_optim = optim_mapper[args.optimizer](lr=args.optimizer_lr, version=args.version, schedule=args.schedule)
        g_optim = optim_mapper[args.optimizer](lr=args.optimizer_lr, version=args.version, schedule=args.schedule)



# We use the Adam paramaters from Gulrajani et al.
generator_model.compile(optimizer=g_optim, loss=wasserstein_loss)

# Now that the generator_model is compiled, we can make the discriminator layers trainable.
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

# The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
# The noise seed is run through the generator model to get generated images. Both real and generated images
# are then run through the discriminator. Although we could concatenate the real and generated images into a
# single tensor, we don't (see model compilation for why).
real_samples = Input(shape=X_train.shape[1:])
generator_input_for_discriminator = Input(shape=(100,))
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)

# We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
averaged_samples = RandomWeightedAverage()(inputs = [real_samples, generated_samples_for_discriminator])
# We then run these samples through the discriminator as well. Note that we never really use the discriminator
# output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
averaged_samples_out = discriminator(averaged_samples)

# The gradient penalty loss function requires the input averaged samples to get gradients. However,
# Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
# of the function with the averaged samples here.
partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

# Keras requires that inputs and outputs have the same number of samples. This is why we didn't concatenate the
# real samples and generated samples before passing them to the discriminator: If we had, it would create an
# output with 2 * BATCH_SIZE samples, while the output of the "averaged" samples for gradient penalty
# would have only BATCH_SIZE samples.

# If we don't concatenate the real and generated samples, however, we get three outputs: One of the generated
# samples, one of the real samples, and one of the averaged samples, all of size BATCH_SIZE. This works neatly!
discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])
# We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
# samples, and the gradient penalty loss for the averaged samples.
discriminator_model.compile(optimizer=d_optim,
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])
# We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
# negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
# gradient_penalty loss function and is not used.
positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

for epoch in range(50):
    np.random.shuffle(X_train)
    print("Epoch: ", epoch)
    print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))
    discriminator_loss = []
    generator_loss = []
    minibatches_size = BATCH_SIZE * TRAINING_RATIO
    
    counter = 0
    
    for i in range(int(X_train.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
        discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]
        counter += 1
        print(f"counter: {counter}")
        for j in range(TRAINING_RATIO):
            image_batch = discriminator_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            noise = np.random.rand(BATCH_SIZE, 100).astype(np.float32)
            discriminator_loss.append(discriminator_model.train_on_batch(x=[image_batch, noise],y=[positive_y, negative_y, dummy_y]))
        generator_loss.append(generator_model.train_on_batch(np.random.rand(BATCH_SIZE, 100), positive_y))
    
    # Still needs some code to display losses from the generator and discriminator, progress bars, etc.
    
    if((epoch+1) % 2 == 0):
        print(f"Generating and saving images for epoch: {epoch}")
        generate_images(generator, args.output_dir, epoch)
        generator.save_weights(os.path.join(args.output_dir, 'epoch_{}_g.h5'.format(epoch)))
        discriminator.save_weights(os.path.join(args.output_dir, 'epoch_{}_d.h5'.format(epoch)))
