#COPY MAT KARNA NA PLEASE, UMMMM...... NHI NHI COPY KARLENA PAR EK BAAR BATA DENA CONNECT KARKE IF YOU WISH TO PLEASE PLEASE, THIS IS THE LEAST I CAN EXPECT!

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

# Load and preprocess the MNIST dataset
(train_images, _), (_, _) = mnist.load_data()
train_images = train_images / 127.5 - 1.0
train_images = np.expand_dims(train_images, axis=-1)

# Define the generator model
def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(128 * 7 * 7, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((7, 7, 128)))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(1, (7, 7), activation='tanh', padding='same'))
    return model

# Define the discriminator model
def build_discriminator(img_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Build and compile the discriminator
img_shape = (28, 28, 1)
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

# Build the generator
latent_dim = 100
generator = build_generator(latent_dim)

# Combine the generator and discriminator into a GAN model
discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = models.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# Function to train the GAN
def train_gan(epochs=10000, batch_size=128, save_interval=1000):
    batch_count = train_images.shape[0] // batch_size

    for e in range(epochs + 1):
        for _ in range(batch_count):
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            generated_images = generator.predict(noise)
            image_batch = train_images[np.random.randint(0, train_images.shape[0], size=batch_size)]

            X = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9  # One-sided label smoothing

            d_loss = discriminator.train_on_batch(X, y_dis)

            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            y_gen = np.ones(batch_size)
            g_loss = gan.train_on_batch(noise, y_gen)

        if e % save_interval == 0:
            print(f"Epoch {e}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")
            save_generated_images(e, generator)

# Function to save generated images
def save_generated_images(epoch, generator, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, size=[examples, latent_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images * 0.5 + 0.5  # Rescale pixel values to [0, 1]

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i, :, :, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"gan_generated_image_epoch_{epoch}.png")

# Train the GAN with fewer epochs
train_gan(epochs=1000, batch_size=64)

#IMPORTING AND SHOWING IMAGE

import IPython.display as display
from PIL import Image

# Specify the path to the generated image
image_path = "gan_generated_image_epoch_0.png"  # Replace XXX with the epoch number

# Display the generated image
img = Image.open(image_path)
display.display(img)

#ACCURACY EVALUATION

# Evaluate the discriminator accuracy on real images
real_labels = np.ones((train_images.shape[0], 1))
real_accuracy = discriminator.evaluate(train_images, real_labels, verbose=0)
print(f"Discriminator Accuracy on Real Images: {real_accuracy[1]*100:.2f}%")

# Evaluate the discriminator accuracy on generated images
noise = np.random.normal(0, 1, size=[100, latent_dim])
generated_images = generator.predict(noise)
fake_labels = np.zeros((100, 1))
fake_accuracy = discriminator.evaluate(generated_images, fake_labels, verbose=0)
print(f"Discriminator Accuracy on Generated Images: {fake_accuracy[1]*100:.2f}%")

#HYPERPARAMETER TUNING

# Adjust the learning rate for both the discriminator and generator
discriminator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5)
generator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5)

# Compile the discriminator with the new optimizer
discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])

# Compile the GAN model with the new optimizer
gan.compile(loss='binary_crossentropy', optimizer=generator_optimizer)

#MODEL OPTIMISATION

# Add gradient clipping to the discriminator optimizer
discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, clipvalue=0.5)

# Compile the discriminator with the new optimizer
discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])

# Evaluate the discriminator accuracy on real images
real_labels = np.ones((train_images.shape[0], 1))
real_accuracy = discriminator.evaluate(train_images, real_labels, verbose=0)
print(f"Discriminator Accuracy on Real Images: {real_accuracy[1]*100:.2f}%")

# Evaluate the discriminator accuracy on generated images
noise = np.random.normal(0, 1, size=[100, latent_dim])
generated_images = generator.predict(noise)
fake_labels = np.zeros((100, 1))
fake_accuracy = discriminator.evaluate(generated_images, fake_labels, verbose=0)
print(f"Discriminator Accuracy on Generated Images: {fake_accuracy[1]*100:.2f}%")

#FINE TUNING MODEL

# Fine-tune the generator by adding more layers
def fine_tune_generator(latent_dim):
    model = models.Sequential()
    # Add more layers or change existing layers as needed
    model.add(layers.Dense(256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((7, 7, 128)))
    # ... rest of the generator architecture
    return model

# Rebuild the generator with the fine-tuned architecture
generator = fine_tune_generator(latent_dim)

#END OF CODE
#COPY MAT KARNA NA PLEASE, UMMMM...... NHI NHI COPY KARLENA PAR EK BAAR BATA DENA CONNECT KARKE IF YOU WISH TO PLEASE PLEASE, THIS IS THE LEAST I CAN EXPECT!
