"""
Pythonic object oriented wrappersfor the potentials supplied by the
PyDenseCRF package.
"""

import numpy as np
import pydensecrf.utils as crf_utils

__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


class Potential:
    pass


class UnaryPotential:
    pass


class PairwisePotential:
    pass


class UnaryPotentialFromProbabilities(UnaryPotential):
    def __init__(self, one_class_probabilities=False):
        """Creates a unary potential from a probability mask.

        If the probabilities are for more than one class, then the first
        axis of the probabilitymask should specify which class it is a
        probability for. The shape of the probability mask for a 2D image
        should therefore be 
            [num_classes, height, width],
        the shape of 3D images should be
            [num_classes, height, width, depth],
        etc.

        In the case of only two classes, images with only one probability
        is accepted. I.e. the axis specifying which class the probabilities
        signify can be skipped.
        """
        self.one_class_probabilities = one_class_probabilities
    
    def apply(self, probability_mask, colour_axis=0):
        if self.one_class_probabilities:
            colour_axis = 0
            probability_mask = np.stack((probability_mask, 1-probability_mask),
                                        axis=0)

        probability_mask = np.moveaxis(probability_mask, colour_axis, 0)
        self.probability_mask = probability_mask
        self.colour_axis = colour_axis
        print(probability_mask.shape)
        return crf_utils.unary_from_softmax(probability_mask)


class AnisotropicBilateralPotential(PairwisePotential):
    def __init__(self, spatial_sigmas, colour_sigmas, compatibility):
        r"""Creates a bilateral potential for image segmentation with DenseCRF.

        The bilateral energy-function is on the form:
        .. math::
           \mu exp(-(x_i-x_j)^T\Sigma_x^{-1}(x_i-x_j)
                   -(c_i-c_j)^T\Sigma_c^{-1}(c_i-c_j)),
        
        where :math:`x_i` and :math:`c_i` is respectively the position
        and colour of pixel i. :math:`\mu` is the (inverse) compatibility
        between the label of pixel i and pixel j. The :math:`\Sigma` matrices
        are created by `diag(spatial_sigmas)` and `diag(colour_sigma)`
        respectively.

        Arguments
        ---------
        spatial_sigmas : numpy.ndarray
            Specifies how fast the energy should decline with spatial distance.
        colour_sigmas : numpy.ndarray
            Specifies how fast the energy should decline with colour distance.
        compatibility : float or numpy.ndarray
            The (inverse) compatibility function. If constant, a Potts-like
            potential is used, if it is a matrix, then the element
            compatibility[i, j] specifies the cost of having label i adjacent
            to a pixel with label j. High `compatibility` values yields a
            strong potential.
        """
        self.spatial_sigmas = spatial_sigmas
        self.colour_sigmas = colour_sigmas
        self.compatibility = compatibility
    
    def apply(self, image, colour_axis):
        return crf_utils.create_pairwise_bilateral(
            img=image,
            sdims=self.spatial_sigmas,
            schan=self.colour_sigmas,
            chdim=colour_axis
        )


class BilateralPotential(PairwisePotential):
    def __init__(self, spatial_sigma, colour_sigma, compatibility):
        r"""Creates a bilateral potential for image segmentation with DenseCRF.

        The bilateral energy-function is on the form:
        .. math::
           \mu exp(-\frac{||x_i-x_j||^2}{\sigma_x}
                   -\frac{||c_i-c_j||^2}{\sigma_c}),
        
        where :math:`x_i` and :math:`c_i` is respectively the position
        and colour of pixel i. :math:`\mu` is the (inverse) compatibility
        between the label of pixel i and pixel j.

        Arguments
        ---------
        spatial_sigma : float
            Specifies how fast the energy should decline with spatial distance.
        colour_sigma : float
            Specifies how fast the energy should decline with colour distance.
        compatibility : float or numpy.ndarray
            The (inverse) compatibility function. If constant, a Potts-like
            potential is used, if it is a matrix, then the element
            compatibility[i, j] specifies the cost of having label i adjacent
            to a pixel with label j. High `compatibility` values yields a
            strong potential.
        """
        self.spatial_sigma = spatial_sigma
        self.colour_sigma = colour_sigma
        self.compatibility = compatibility
    
    def apply(self, image, colour_axis):
        spatial_sigmas = [self.spatial_sigma for _ in range(len(image.shape)-1)]
        colour_sigmas = [self.colour_sigma 
                            for _ in range(image.shape[colour_axis])]
        return crf_utils.create_pairwise_bilateral(
            img=image,
            sdims=spatial_sigmas,
            schan=colour_sigmas,
            chdim=colour_axis
        )


class AnisotropicGaussianPotential(PairwisePotential):
    def __init__(self, sigmas, compatibility):
        r"""Creates an anisotropic Gaussian potential for image segmentation.

        The anisotropic Gaussian energy-function is on the form:
        .. math::
           \mu exp(-(x_i-x_j)^T\Sigma^{-1}(x_i-x_j)),
        
        where :math:`x_i` is the position pixel i. :math:`\mu` is the 
        (inverse) compatibility between the label of pixel i and pixel j. The
        :math:`\Sigma` matrix is given by diag(`sigmas`).

        Arguments
        ---------
        sigmas : numpy.ndarray
            Specifies how fast the energy should decline with spatial distance
            along each of the axes.
        compatibility : float or numpy.ndarray
            The (inverse) compatibility function. If constant, a Potts-like
            potential is used, if it is a matrix, then the element
            compatibility[i, j] specifies the cost of having label i adjacent
            to a pixel with label j. High `compatibility` values yields a
            strong potential.
        """
        self.sigmas = sigmas
        self.compatibility = compatibility
    
    def apply(self, image, colour_axis):
        shape = [
            image.shape[i] for i in range(len(image.shape)) if i != colour_axis
        ]
        return crf_utils.create_pairwise_gaussian(
            sdims=self.sigmas,
            shape=shape
        )


class GaussianPotential(PairwisePotential):
    def __init__(self, sigma, compatibility):
        r"""Creates a Gaussian potential for image segmentation with DenseCRF.

        The Gaussian energy-function is on the form:
        .. math::
           \mu exp(-\frac{||x_i-x_j||^2}{\sigma}}),
        
        where :math:`x_i` is the position pixel i. :math:`\mu` is the 
        (inverse) compatibility between the label of pixel i and pixel j.

        Arguments
        ---------
        sigma : float
            Specifies how fast the energy should decline with spatial distance.
        compatibility : float or numpy.ndarray
            The (inverse) compatibility function. If constant, a Potts-like
            potential is used, if it is a matrix, then the element
            compatibility[i, j] specifies the cost of having label i adjacent
            to a pixel with label j. High `compatibility` values yields a
            strong potential.
        """
        self.sigma = sigma
        self.compatibility = compatibility
    
    def apply(self, image, colour_axis):
        spatial_sigmas = [self.sigma for _ in range(len(image.shape)-1)]
        shape = [
            image.shape[i] for i in range(len(image.shape)) if i != colour_axis
        ]
        return crf_utils.create_pairwise_gaussian(
            sdims=spatial_sigmas,
        )