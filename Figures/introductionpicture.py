# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:55:38 2024

@author: demib
"""

import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
from scipy.ndimage import zoom
from tensorly.decomposition import parafac, tucker, matrix_product_state
from PIL import Image

plt.close('all')
random_state = 12345

# Load the image
image_path = 'IMG_0245.jpg' 
image = Image.open(image_path)

# Convert the image to RGB (if not already in RGB mode)
image = image.convert('RGB')

# Convert the image to a numpy array
rgb_matrix = np.array(image)

image = tl.tensor(zoom(rgb_matrix, (0.3, 0.3, 1)), dtype='float64')

def to_image(tensor):
    """A convenience function to convert from a float dtype back to uint8"""
    im = tl.to_numpy(tensor)
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)

# Rank of the CP decomposition
cp_rank = 32
# Rank of the Tucker decomposition
tucker_rank = [30, 30, 3]
# Rank of the TT decomposition
tt_rank = [1, 17, 17, 1]

# Perform the CP decomposition
weights, factors = parafac(image, rank=cp_rank, init='random', tol=10e-4)
# Reconstruct the image from the factors
cp_reconstruction = tl.cp_to_tensor((weights, factors))

# Tucker decomposition
core, tucker_factors = tucker(image, rank=tucker_rank, init='random', tol=10e-5, random_state=random_state)
tucker_reconstruction = tl.tucker_to_tensor((core, tucker_factors))

# TT decomposition
tt_cores = matrix_product_state(image, rank=tt_rank)
tt_reconstruction = tl.tt_to_tensor(tt_cores)

#%%
# Plotting the original and reconstruction from the decompositions
fig, axs = plt.subplots(1, 4, figsize=(15, 5))
images = [image, cp_reconstruction, tucker_reconstruction, tt_reconstruction]
titles = ['Original', 'CP', 'Tucker', 'TT']

for ax, img, title in zip(axs, images, titles):
    ax.imshow(to_image(img))
    ax.set_title(title, fontsize=15)
    ax.axis('off')

plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02, wspace=0.05, hspace=0.02)
plt.show()

# Image dimensions
I, J, K = image.shape

# Original number of parameters
original_params = I * J * K

# CP Decomposition
cp_params = (I + J + K) * cp_rank

# Tucker Decomposition
core_params = np.prod(core.shape)
tucker_params = core_params + sum([np.prod(factor.shape) for factor in tucker_factors])

# TT Decomposition
tt_params = sum([np.prod(core.shape) for core in tt_cores])

# Calculate percentages
def calculate_percentage(original, decomposed):
    retained = decomposed / original * 100
    lost = 100 - retained
    return retained, lost

cp_retained, cp_lost = calculate_percentage(original_params, cp_params)
tucker_retained, tucker_lost = calculate_percentage(original_params, tucker_params)
tt_retained, tt_lost = calculate_percentage(original_params, tt_params)

# Print the results
print(f"Original parameters: {original_params}")
print(f"CP parameters: {cp_params}, Retained: {cp_retained:.2f}%, Lost: {cp_lost:.2f}%")
print(f"Tucker parameters: {tucker_params}, Retained: {tucker_retained:.2f}%, Lost: {tucker_lost:.2f}%")
print(f"TT parameters: {tt_params}, Retained: {tt_retained:.2f}%, Lost: {tt_lost:.2f}%")
