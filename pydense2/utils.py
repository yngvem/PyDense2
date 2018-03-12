from . import crf_model
from . import potentials
import numpy as np
try:
    import joblib
except ModuleNotFoundError:
    print('Joblib not found, parallel computations not possible.\n'
          '    To resolve this, run `pip install joblib`.next')
from sklearn import metrics


class CRFEvaluator:
    def __init__(self, model, images, probabilities, true_masks, colour_axis,
                 class_axis):
        """Used to evaluate a specific CRF model.

        Be careful when setting colour_axis and class_axis to anything but -1,
        as the first axis should specify which image we are looking at no matter
        what. However, if the shape of images is
            [num_images, colour, height, width, depth],
        then `colour_axis` should be zero, because the colour information lies
        in the zeroth axis of each image. The same also holds for the
        `class_axis` argument.

        Arguments
        ---------
        model : PyCRF2.CRFModel
            The CRF model to evaluate.
        images : Array like
            Contains all the images that should be post processed with CRF.
        probabilities : numpy.ndarray
            Contains the predicted probability masks for each image in the
            `images` array/list.
        true_masks : numpy.ndarray
            Contains the true class masks for each image in the `images`
            array/list.
        colour_axis : numpy.ndarray
            The axis in which the colour information of the images lies.
            Note the warning further up.
        class_axis : np.ndarray
            The axis in which the class information of the masks and
            probabilities lies. Note the warning further up.
        """
        self.model = model
        self.images = images
        self.probabilities = probabilities
        self.true_masks = true_masks
        self.pred_masks = None

        self.colour_axis = colour_axis
        self.class_axis = class_axis
        self.num_images = len(images)

        self.evaluation_metrics = {}

    def perform_inference(self, num_steps, parallel=False, n_jobs=-1):
        """Perform inference for a given number of iterations.

        Joblib is used for parallel computation, if `n_jobs` is negative,
        then `n_jobs` are set to (num_cpus+1+n_jobs).

        Arguments
        ---------
        num_steps : int
            The number of mean field iterations to perform.
        parallel : bool
            Whether the jobs should be ran in parallel.
        n_jobs : int
            How many jobs that shall run in parallel. If this is negative,
            it is set to (n_cpus + 1 + n_jobs).
        """
        if not parallel:
            self.pred_masks = [self.compute_segmentation_mask(i, num_steps)
                               for i in range(self.num_images)]
        else:
            self.pred_masks = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.compute_segmentation_mask)(i, num_steps)
                    for i in range(num_steps)
            )

    def compute_segmentation_mask(self, image_num, num_steps):
        """Compute a single segmentation mask

        Arguments
        ---------
        image_num : int
            Which image to segment
        num_steps : int
            Number of mean field iterations to run.
        Returns
        -------
        numpy.ndarray
            The segmentation mask of the specified image
        """
        return self.model.set_image_and_predict(
            image=self.images[image_num],
            probabilities=self.probabilities[image_num],
            colour_axis=self.colour_axis,
            class_axis=self.class_axis,
            num_steps=num_steps
        )

    def compute_evaluation_metrics(self, predicted_masks, true_masks,
                                   metric='dice_coefficient',
                                   metric_params=None,
                                   parallel=False, n_jobs=-1):
        """Compute the specified evaluation metric for all the predicted masks.

        Arguments
        ---------
        predicted_masks : Array like
            Iterable. Each index is the predicted segmentation mask.
        true_masks : Array like
            Iterable. Each index is the true mask corresponding to the predicted
            mask at the same index.
        metric : str
            Which evaluation metric to use. Should be a function of this
            instance.
        metric_params : dict
            Keyword arguments for the metric. e.g. if the `average` parameter
            is to be changed to 'micro' for the dice coefficient, then
            `metric_params` should be set to {'average': 'micro'},
        parallel : bool
            Whether the jobs should be ran in parallel.
        n_jobs : int
            How many jobs that shall run in parallel. If this is negative,
            it is set to (n_cpus + 1 + n_jobs).
        """
        metric_params = metric_params if metric_params is not None else {}
        metric = getattr(self, metric)

        if not parallel:
            return np.array([
                metric(predicted_masks[i], true_masks[i], **metric_params)
                    for i in range(self.num_images)
            ])
        else:
            return np.array(joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(metric)(
                    predicted_masks,
                    true_masks,
                    **metric_params
                ) for i in range(self.num_images)
            ))

    def evaluate_model(self, metric='dice_coefficient', metric_params=None,
                       parallel=False, n_jobs=-1):
        """Compute the specified evaluation metric for all the images.

        Arguments
        ---------
        metric : str
            Which evaluation metric to use. Should be a function of this
            instance.
        metric_params : dict
            Keyword arguments for the metric. e.g. if the `average` parameter
            is to be changed to 'micro' for the dice coefficient, then
            `metric_params` should be set to {'average': 'micro'},
        parallel : bool
            Whether the jobs should be ran in parallel.
        n_jobs : int
            How many jobs that shall run in parallel. If this is negative,
            it is set to (n_cpus + 1 + n_jobs).
        """
        self.evaluation_metrics[metric] = self.compute_evaluation_metrics(
            predicted_masks=self.pred_masks,
            true_masks=self.true_masks,
            metric=metric,
            metric_params=metric_params,
            parallel=parallel,
            n_jobs=n_jobs
        )
        return self.evaluation_metrics[metric]

    def perform_inference_and_evaluate(self, num_steps=50,
                                       metric='dice_coefficient',
                                       metric_params=None,
                                       parallel=False, n_jobs=-1):
        """Perform inference and evaluate the model afterwards.

        Arguments
        ---------
        num_steps : int
            Number of mean field iterations to perform
        metric : str
            Which evaluation metric to use. Should be a function of this
            instance.
        metric_params : dict
            Keyword arguments for the metric. e.g. if the `average` parameter
            is to be changed to 'micro' for the dice coefficient, then
            `metric_params` should be set to {'average': 'micro'},
        parallel : bool
            Whether the jobs should be ran in parallel.
        n_jobs : int
            How many jobs that shall run in parallel. If this is negative,
            it is set to (n_cpus + 1 + n_jobs).
        """
        self.perform_inference(num_steps, parallel, n_jobs)
        return self.evaluate_model(
            metric=metric,
            metric_params=metric_params,
            parallel=parallel,
            n_jobs=n_jobs
        )

    # --------------------Evaluation metrics------------------------ #
    @staticmethod
    def dice_coefficient(mask1, mask2, average='weighted'):
        return metrics.f1_score(mask1.reshape(-1, mask1.shape[-1]),
                                mask2.reshape(-1, mask2.shape[-1]),
                                average='weighted')

