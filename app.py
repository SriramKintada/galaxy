import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import matplotlib.pyplot as plt

import h5py
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Dense, LeakyReLU, Flatten
import matplotlib
matplotlib.use("Agg")

# Initialize Flask app
app = Flask(__name__, template_folder="app/templates")

# Paths
OUTPUT_DIR = "generated"
MODEL_PATH = "models/generator_wgan_gp.h5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, Activation


with h5py.File("models/generator_wgan_gp.h5", "r") as f:
    print("Keys in the HDF5 file:", list(f.keys()))
    for key in f.keys():
        print(f"{key}: {list(f[key])}")


def create_generator_model(input_z_shape=100):
    """
    Creates a generator model for a Conditional Generative Adversarial Network (CGAN) using a DCGAN-like architecture.

    The generator takes a noise vector (usually a random vector) as input and generates synthetic images.
    The architecture is designed to progressively upsample the noise vector to produce an image with the desired dimensions.

    Args:
        input_z_shape (int): The shape of the input noise vector. Default is NOISE_DIM, which represents the dimensionality of the noise vector.

    Returns:
        tensorflow.keras.Model: A Keras Model object that represents the generator network.

    The generator architecture consists of:
    1. A Dense layer that transforms the noise vector into a high-dimensional tensor.
    2. A Reshape layer that reshapes this tensor into a 4x4 spatial resolution with 512 channels.
    3. A series of Conv2DTranspose layers that progressively upsample the image. Each layer increases the spatial resolution (height and width) and reduces the number of channels.
    4. BatchNormalization layers are applied after each Conv2DTranspose layer to normalize the activations, which helps stabilize and speed up training.
    5. LeakyReLU activation functions are used to introduce non-linearity and avoid dead neurons.
    6. The final Conv2DTranspose layer produces an image with 3 channels (RGB) using the 'tanh' activation function to scale pixel values to the range [-1, 1].
    """
    # Define the input layer with the shape of the noise vector.
    # This is the input to the generator network.
    input_z_layer = Input((input_z_shape,))

    # First, fully connect the noise vector to a high-dimensional tensor.
    # This tensor will be reshaped into a 4x4 image with 512 channels.
    z = Dense(4 * 4 * 512, use_bias=False)(input_z_layer)
    z = Reshape((4, 4, 512))(z)

    # Apply a series of Conv2DTranspose layers to upsample the image.
    # Each Conv2DTranspose layer increases the spatial dimensions of the image.

    # First transposed convolution layer
    x = Conv2DTranspose(512, (4, 4), strides=(1, 1), padding='same', use_bias=False,
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(z)
    x = BatchNormalization()(x)  # Normalize the output of the convolution to help stabilize training
    x = LeakyReLU()(x)  # Apply Leaky ReLU activation function for non-linearity

    # Second transposed convolution layer
    x = Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False,
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Third transposed convolution layer
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False,
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Fourth transposed convolution layer
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False,
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Fifth transposed convolution layer
    x = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', use_bias=False,
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Final transposed convolution layer to generate the output image
    # The output image has 3 channels (RGB) and uses 'tanh' activation to ensure pixel values are between -1 and 1
    output = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation="tanh",
                             kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)

    # Create a model with the input and output layers defined above
    model = Model(inputs=input_z_layer, outputs=output)
    return model


generator_model = create_generator_model()


generator_model.load_weights("models/generator_wgan_gp.h5")

print("Model and weights loaded successfully!")


def image_grid(images, fig, images_to_generate):

    for i in range(images_to_generate):
        axs = fig.add_subplot(int(np.sqrt(images_to_generate)), int(np.sqrt(images_to_generate)), i + 1)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.imshow(images[i] * 0.5 + 0.5)  # Scale from [-1, 1] to [0, 1]

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate")
def generate_galaxy():

    NUM_IMAGES = 1
    NOISE_DIM = 100


    if NUM_IMAGES < 0:
        raise ValueError("Number must be non-negative")
    if not np.sqrt(NUM_IMAGES).is_integer():
        raise ValueError(f"{NUM_IMAGES} is not a perfect square")


    test_noise = tf.random.normal([NUM_IMAGES, 100])


    try:
        prediction = generator_model.predict(test_noise)
    except Exception as e:
        raise RuntimeError(f"Error during image generation: {e}")


    fig = plt.figure(figsize=(12, 12))
    image_grid(prediction, fig, NUM_IMAGES)


    generated_image_path = os.path.join(OUTPUT_DIR, "generated_galaxy.png")
    plt.savefig(generated_image_path)
    plt.close(fig)

    return send_from_directory(OUTPUT_DIR, os.path.basename(generated_image_path))


if __name__ == "__main__":
    app.run(debug=True)
