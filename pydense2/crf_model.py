"""
A pythonic object oriented wrapper for the DenseCRF optimizer supplied
by the PyDenseCRF package.
"""

__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'

import numpy as np
import pydensecrf.densecrf as dense_crf
from . import potentials


class DenseCRF:
    def __init__(self, num_classes, unary_potential, pairwise_potentials):
        """A wrapper for the PyDenseCRF functions. 
        
        Images of any dimensionality is supported.
        
        Arguments
        ---------
        num_classes : int
        unary_potential : UnaryPotential
        pairwise_potentials : PairwisePotential or array like
            A collection of PairwisePotential instances.
        """
        if isinstance(pairwise_potentials, potentials.PairwisePotential):
            pairwise_potentials = [pairwise_potentials]

        for pairwise_potential in pairwise_potentials:
            self.check_potential(pairwise_potential, potentials.PairwisePotential)
        self.check_potential(unary_potential, potentials.UnaryPotential)

        self.num_classes = num_classes
        self.pairwise_potentials = pairwise_potentials
        self.unary_potential = unary_potential

        self.crf_model = None
        self._image = None
        self._colour_axis = None

        # Variables needed for the CRF model to run
        self._Q = None
        self._tmp1 = None
        self._tmp2 = None

    def set_image(self, image, probabilities, colour_axis=None, class_axis=None):
        """Set the image for the CRF model to perform inference on.

        Arguments
        ---------
        image : numpy.ndarray
            The image to segment.
        probabilities : numpy.ndarray
            Class probabilities for each pixel.
        colour_axis : int
            Which axis of the image array the colour information lies in.
            Usually the last axis (-1), but can also be the first axis (0).
            If `image` is a grayscale image, this should be set to None.
        class_axis : int
            Which axis of the probabilities array the class information lies.
            If the probabilites are on the form
                ``[image_height, image_width, class]``,
            then class axis should be set to either `3` or `-1`.
        """
        if colour_axis is None:
            colour_axis = -1
            image = image.reshape(*image.shape, 1)

        self._image = image
        self._colour_axis = self.fix_negative_index(colour_axis)
        self._class_axis = class_axis
        self._probabilities = probabilities
        self._image_shape = \
            image.shape[:self._colour_axis] + image.shape[self._colour_axis+1:]

        self._create_model()
        self._set_potentials()
        self._start_inference()        

    def set_image_and_predict(self, image, probabilities, colour_axis=None,
                              class_axis=None, num_steps=50):
        self.set_image(image, probabilities, colour_axis, class_axis)
        self.perform_inference(num_steps)
        return self.segmentation_map

    def _create_model(self):
        self.crf_model = dense_crf.DenseCRF(np.prod(self.image_shape),
                                            self.num_classes)

    def _set_potentials(self):
        """Apply the potentials to current image.
        """
        self.crf_model.setUnaryEnergy(
            self.unary_potential.apply(self._probabilities, self._class_axis)
        )

        for pairwise_potential in self.pairwise_potentials:
            self.crf_model.addPairwiseEnergy(
                pairwise_potential.apply(self._image, self._colour_axis),
                compat=pairwise_potential.compatibility
            )

    def _start_inference(self):
        """Prepare the model for inference."""
        self._Q, self._tmp1, self._tmp2 = self.crf_model.startInference()

    def inference_step(self):
        """Performs one iteration in the CRF energy minimisation.
        """
        self.crf_model.stepInference(self._Q, self._tmp1, self._tmp2)

    def perform_inference(self, num_steps):
        """Perform `num_steps` iterations in the CRF energy minimsation.

        The minimisation continues where it previously left off. So calling
        this function twice with `num_steps=10` is the same as calling it
        once with `num_steps=20`.
        """
        for _ in range(num_steps):
            self.inference_step()
        
        return self.segmentation_map

    def fix_negative_index(self, idx):
        if idx < 0:
            idx = len(self._image.shape) + idx
        
        return idx

    @staticmethod
    def check_potential(potential, potential_type=None):
        """Checks `potential` is of correct type and has an apply function.
        """
        potential_type = potentials.Potential if potential_type is None \
                                              else potential_type
        potential_name = potential_type.__name__
        if not isinstance(potential, potential_type):
            raise ValueError(
                f'{potential} is not a {potential_name}'
            )
        elif not hasattr(potential, 'apply'):
            raise ValueError(
                f'{potential} is not yet implemented.'
            )

    @property
    def kl_divergence(self):
        """The KL divergence"""
        if self._Q is None:
            raise RuntimeWarning('No image is set')
            return None
        return self.crf_model.klDivergence(Q)/np.prod(self.image_shape)
    
    @property
    def segmentation_map(self):
        if self._Q is None:
            raise RuntimeWarning('No image is set')
            return None
        return np.argmax(self._Q, axis=0).reshape(self.image_shape)

    @property
    def image(self):
        return self._image.copy()
    
    @property
    def image_shape(self):
        return self._image_shape
