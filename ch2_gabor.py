# Code
# Gabor Filtering

import matplotlib.pyplot as plt
import numpy as np
from skimage import data, filters, transform

# Load four texture-like images
tex1 = data.brick()       # bricks texture
tex2 = data.grass()       # grass texture
tex3 = data.coins()       # coins texture
tex4 = data.binary_blobs(length=256, seed=1).astype(float)  # synthetic blobs

# Ensure grayscale
if tex1.ndim == 3: tex1 = tex1[...,0]
if tex2.ndim == 3: tex2 = tex2[...,0]
if tex3.ndim == 3: tex3 = tex3[...,0]

# Resize all to the same shape
tex1 = transform.resize(tex1, (256, 256), anti_aliasing=True)
tex2 = transform.resize(tex2, (256, 256), anti_aliasing=True)
tex3 = transform.resize(tex3, (256, 256), anti_aliasing=True)
tex4 = transform.resize(tex4, (256, 256), anti_aliasing=True)

# Build 4-quadrant texture image
top = np.hstack((tex1, tex2))
bottom = np.hstack((tex3, tex4))
image = np.vstack((top, bottom))

# Parameters for Gabor filters
frequency = 0.2
orientations = [0, np.pi/4, np.pi/2]  # 0°, 45°, 90°

# Apply Gabor filters
responses = []
for theta in orientations:
    real, imag = filters.gabor(image, frequency=frequency, theta=theta)
    responses.append(real)

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0,0].imshow(image, cmap='gray')
axes[0,0].set_title("Original 4-Quadrant Texture Image")
axes[0,0].axis('off')

axes[0,1].imshow(responses[0], cmap='gray')
axes[0,1].set_title("Gabor Filter (0°)")
axes[0,1].axis('off')

axes[1,0].imshow(responses[1], cmap='gray')
axes[1,0].set_title("Gabor Filter (45°)")
axes[1,0].axis('off')

axes[1,1].imshow(responses[2], cmap='gray')
axes[1,1].set_title("Gabor Filter (90°)")
axes[1,1].axis('off')

plt.tight_layout()
plt.show()
