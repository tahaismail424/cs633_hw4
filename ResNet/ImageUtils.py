import numpy as np
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import torch

""" This script implements the functions for data augmentation and preprocessing.
"""

def radio_process(batch, training=True, device='mps', seed=42):
    batch = torch.tensor(batch, dtype=torch.float32, device=device).unsqueeze(-1) / 255

    # do randomized transforms for training data
    if training:
        torch.manual_seed(seed)
        random = T.Compose([
            T.RandomCrop(26),
            T.RandomHorizontalFlip(0.5)
        ])
        batch = torch.stack([random(img) for img in batch])

    # resize to 32
    batch = F.resize(batch, (32, 32))
    return batch
    
def parse_record(record, training, dim=32, channels=3, seed=42):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [chan, dim, dim].
    """
    # Reshape from [batch, depth * height * width] to [batch, depth, height, width].
    B = record.shape[0]

    depth_major = record.reshape((B, channels, dim, dim))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [0, 2, 3, 1])

    image = preprocess_image(image, training, dim=dim, seed=seed)

    # Convert from [batch, height, width, depth] to [batch, depth, height, width]
    image = np.transpose(image, [0, 3, 1, 2])

    return image

def preprocess_image(image, training, dim=32, seed=42):
    """ Preprocess an image batch of shape [batch, height, width, depth].

    Args:
        image: An array of shape [batch, 32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [batch, 32, 32, 3].
    """
    B = image.shape[0]
    if training:
        rng = np.random.default_rng(seed=seed)

        crop = np.zeros_like(image)
        # Resize the image to add four extra pixels on each side.
        image = np.pad(
            image,
            pad_width=((0, 0), (4, 4), (4, 4), (0, 0)), 
            mode='constant', 
            constant_values=0
        )
        

        ### YOUR CODE HERE
        # Randomly crop a [dim, dim] section of the image.
        # HINT: randomly generate the upper left point of the image
        top = rng.integers(0, 9, size=B)
        left = rng.integers(0, 9, size=B)

        b = np.arange(B)
        rows = top[:, None] + np.arange(dim)
        cols = left[:, None] + np.arange(dim)

        crop = image[
            b[:, None, None], 
            rows[:, :, None], 
            cols[:, None, :]
            ]
        
        # Randomly flip the images horizontally.
        p_mask = rng.uniform(size=B) < 0.5
        crop = np.where(
            p_mask[:, None, None, None],
            crop[:, :, ::-1], crop
        )

        image = crop
        
    ### YOUR CODE HERE
    # Subtract off the mean and divide by the standard deviation of the pixels.
    channel_means = image.mean(axis=(1, 2))[:, None, None, :]
    channel_stds = image.std(axis=(1, 2))[:, None, None, :]
    image_std = (image - channel_means) / channel_stds
    return image_std