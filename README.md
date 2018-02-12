# PyDense2
## A Pythonic wrapper for the PyDenseCRF package by Lucas Beyer

The [PyDenseCRF](https://github.com/lucasb-eyer/pydensecrf) package by [Lucas Beyer](https://github.com/lucasb-eyer/) is unfortunately not designed in a particularly pythonic fashion. My goal with this project is to make his wrapper for the [DenseCRF C++ code](http://graphics.stanford.edu/projects/densecrf/) by [Philip Krähenbühl](http://www.philkr.net/) easier to use.

This is a tool for post-processing of image segmentation masks, and before using it you should have precomputed class probabilities for each label. Training is unfortunately not supported by this tool.

### Installation guide:
Open a terminal window and navigate to the folder that you want to download PyDense2 into. Write ```git clone https://github.com/yngvem/PyDense2.git``` in the terminal window, followed by ```cd PyDense2``` and at last ```python setup.py install```. PyDense2 can then be imported into any Python file.

#### Import errors:
If you get a cryptic error on import, then you are probably missing libgcc as part of your Python installation. If you are using Anaconda this can be fixed by typing ```conda install libgcc``` in a terminal window.

### Supported potentials:
The unary potentials must come from precomputed class probabilities. In addition to this, two different pairwise potentials are supported. Gaussian potentials and bilateral potentials. Gaussian are given by the function ![\mu(y_i, y_j) exp((x_i - x_j)^T\Sigma^{-1}(x_i - x_j))](http://latex.codecogs.com/gif.latex?\mu(y_i,&space;y_j)&space;exp((x_i&space;-&space;x_j)^T\Sigma_x^{-1}(x_i&space;-&space;x_j))), where i and j are two pixel indices, the y-variables are the pixel labels, the x-variables are the pixel positions, ![\Sigma_x](http://latex.codecogs.com/gif.latex?\Sigma_x) is a hyperparameter deciding how fast the potential should decline in the different spatial directions and ![\mu](http://latex.codecogs.com/gif.latex?\mu) is a function (called the compatibility function) that signify how big the potential for a pair of pixel should be given their labels. Oddily enough, two labels with a high compatibility function value yield a high potential, a better name for this would therefore be incompatibility function. Bilateral potentials are given by a similar function, namely ![\mu(y_i, y_j) exp((x_i - x_j)^T\Sigma_x^{-1}(x_i - x_j)-(c_i - c_j)^T\Sigma_c^{-1}(c_i - c_j)](http://latex.codecogs.com/gif.latex?\mu(y_i,&space;y_j)&space;exp((x_i&space;-&space;x_j)^T\Sigma^{-1}(x_i&space;-&space;x_j)-(c_i&space;-&space;c_j)^T\Sigma_c^{-1}(c_i&space;-&space;c_j))). The only new variables in this equation is the c-values, which are the colour values of the corresponding pixels and the ![\Sigma_c](http://latex.codecogs.com/gif.latex?\Sigma_c), which plays the same role as ![\\Sigma_x](http://latex.codecogs.com/gif.latex?\Sigma_x), but for colour distance instead of spatial distance.


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

