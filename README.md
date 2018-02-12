# PyDense2
## A Pythonic wrapper for the PyDenseCRF package by Lucas Beyer

The [PyDenseCRF](https://github.com/lucasb-eyer/pydensecrf) package by [Lucas Beyer](https://github.com/lucasb-eyer/) is unfortunately not designed in a particularly pythonic fashion. My goal with this project is to make his wrapper for the [DenseCRF C++ code](http://graphics.stanford.edu/projects/densecrf/) by [Philip Krähenbühl](http://www.philkr.net/) easier to use.

This is a tool for post-processing of image segmentation masks, and before using it you should have precomputed class probabilities for each label. Training is unfortunately not supported by this tool.

### Example code
Here is a simple example where the probability mask and image is stored as numpy array files.
```python
import numpy as np
import matplotlib.pyplot as plt
from pydense2 import crf_model, potentials

# Create unary potential
unary = potentials.UnaryPotentialFromProbabilities(one_class_probabilities=True)
# Create pairwise potentials
bilateral_pairwise = potentials.BilateralPotential(
    spatial_sigma=10,
    colour_sigma=1,
    compatibility=4
)
gaussian_pairwise = potentials.GaussianPotential(
    sigma=10,
    compatibility=2
)

# Create CRF model and add potentials
crf = crf_model.DenseCRF(
    num_classes=2,
    unary_potential=unary,
    binary_potentials=[bilateral_pairwise, gaussian_pairwise]
)

# Load image and probabilities
image = np.load('image.npy')
probabilities = np.load('image.npy')

# Set the image for the CRF model to start refining the mask
crf.set_image(
    image=image,
    probabilities=probabilities,
    colour_axis=-1,
    class_axis=-1
)
# Refine the mask, doing 10 iterations
crf.perform_inference(10)
refined_mask10 = crf.segmentation_map

# Refine the mask some more, performing another 10 iterations
crf.perform_inference(10)  # The CRF model will continue where it last left off.
refined_mask20 = crf.segmentation_map

# Plot the results
plt.subplot(121)
plt.title('Segmentation mask after 10 iterations')
plt.imshow(refined_mask10)

plt.subplot(122)
plt.title('Segmentation mask after 20 iterations')
plt.imshow(refined_mask20)
```

## Import errors:
If you get a cryptic error on import, then you are probably missing libgcc as part of your Python installation. If you are using Anaconda this can be fixed by typing ```conda install libgcc``` in a terminal window.
