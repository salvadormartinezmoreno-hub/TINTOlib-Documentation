Clusters
========

The Clusters method transforms tabular data into synthetic grayscale or RGB-like images by applying unsupervised learning techniques such as clustering, density estimation, or factor decomposition. In a first phase, the data are transformed into a latent representation based on the selected algorithm, and subsequently, images are generated from that representation.

Import Clusters
---------------
To import Clusters model use:

>>> from TINTOlib.clusters import Clusters
>>> model = Clusters()

Hyperparameters & Configuration
-------------------------------

When creating the :py:class:`Clusters` class, some parameters can be modified. The parameters are:

.. list-table::
   :widths: 20 45 15 20
   :header-rows: 1

   * - Parameters
     - Description
     - Default value
     - Valid values
   * - :py:data:`problem`
     - The type of problem, defining how the generated images are grouped.
     - None
     - ['supervised', 'unsupervised', 'regression']
   * - :py:data:`normalize`
     - If True, normalizes input data using MinMaxScaler through the parent class configuration.
     - None
     - [True, False]
   * - :py:data:`verbose`
     - Show execution details in the terminal.
     - None
     - [True, False]
   * - :py:data:`algorithm`
     - Algorithm / technique to be applied for generating the synthetic image representation.
     - 'kmeans'
     - ['kmeans', 'gaussianMix', 'aggloKNN', 'mixMethod', 'kde', 'kmedoids', 'factor']
   * - :py:data:`n_clusters`
     - Number of clusters/components used to represent the data. The image size is determined from the square root of this value. It can also be set to ``'auto'`` or to a list of candidate integers, in which case the optimal value is selected based on image stability (SSIM). This parameter does not apply to KDE.
     - 16
     - integer, ``'auto'``, or list of integers
   * - :py:data:`random_seed`
     - Seed for reproducibility.
     - 1
     - integer
   * - :py:data:`n_init`
     - Number of initializations for the ``kmeans`` and ``gaussianMix`` algorithms.
     - 'auto'
     - integer or 'auto'
   * - :py:data:`max_iter`
     - Maximum number of iterations for ``kmeans``, ``gaussianMix``, and ``kmedoids``.
     - 300
     - integer
   * - :py:data:`algorithmMethod`
     - Variant of the k-means algorithm.
     - 'lloyd'
     - ['lloyd', 'elkan']
   * - :py:data:`covariance_type`
     - Covariance type used by the Gaussian Mixture model.
     - 'full'
     - ['full', 'tied', 'diag', 'spherical']
   * - :py:data:`ensamMethod`
     - List of algorithms used when ``algorithm='mixMethod'``. Each selected method is mapped to one image channel. Up to three different methods are allowed and cannot be repeated.
     - []
     - ['kmeans', 'gaussianMix', 'aggloKNN', 'kmedoids', 'factor']
   * - :py:data:`bandwidth`
     - Bandwidth used in kernel density estimation.
     - 1.0
     - float
   * - :py:data:`kernel`
     - Type of kernel used in kernel density estimation.
     - 'gaussian'
     - ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
   * - :py:data:`metric`
     - Distance metric used by ``aggloKNN``, ``kmedoids``, and ``kde``. The valid options depend on the selected algorithm.
     - 'euclidean'
     - For ``kmedoids`` and ``kde``: ['euclidean', 'manhattan', 'chebyshev']; for ``aggloKNN``: ['euclidean', 'manhattan', 'cosine']; for ``mixMethod``: ['euclidean', 'manhattan']
   * - :py:data:`RBFKmeans`
     - If True, transforms k-means distances using an RBF function before converting them into image intensities.
     - False
     - [True, False]

Code example:

>>> model = Clusters(
...     problem='supervised',
...     algorithm='kmeans',
...     n_clusters=25,
...     random_seed=1,
...     max_iter=300,
...     RBFKmeans=True
... )

All the parameters that aren't specifically set will have their default values.

Algorithm Notes
---------------

- ``kmeans``: builds the image representation from distances to centroids, optionally transformed with RBF.
- ``gaussianMix``: uses posterior probabilities of Gaussian mixture components.
- ``aggloKNN``: first performs agglomerative clustering with a k-NN connectivity graph, then uses a KNN classifier to estimate class probabilities.
- ``kde``: estimates one-dimensional kernel density distributions for each feature and uses the resulting densities as image values.
- ``kmedoids``: uses distances to medoids obtained with a PAM-like iterative procedure.
- ``factor``: applies factor analysis and uses the transformed latent representation.
- ``mixMethod``: combines up to three different methods into a 3-channel image.

Additional constraints:

- ``n_clusters='auto'`` and list-based cluster search are only available for ``kmeans``, ``gaussianMix``, ``mixMethod``, ``kmedoids``, and ``factor``.
- ``kde`` and ``aggloKNN`` do not support automatic cluster selection.
- When using ``mixMethod``, ``ensamMethod`` must contain between 1 and 3 non-repeated valid methods.
- When using ``factor``, the number of clusters/components must be smaller than the number of features minus one.

Functions
---------
Clusters has the following functions:

.. list-table::
   :widths: 20 60 20
   :header-rows: 1

   * - Function
     - Description
     - Output
   * - :py:data:`saveHyperparameters(filename)`
     - Allows to save the defined parameters.
     - .pkl file with the configuration
   * - :py:data:`loadHyperparameters(filename)`
     - Load Clusters configuration previously saved with :py:data:`saveHyperparameters(filename)`

       - filename: .pkl file path
     -
   * - :py:data:`fit(data)`
     - Trains the model on the tabular data. Depending on the selected algorithm, this step learns cluster centroids, medoids, mixture components, factor projections, KDE distributions, or multi-channel representations.

       - data: A path to a CSV file or a Pandas DataFrame containing the features and targets. The target column must be the last column.
     -
   * - :py:data:`transform(data, folder)`
     - Generates and saves synthetic images in a specified folder using the representation learned during fitting.

       - data: A path to a CSV file or a Pandas DataFrame containing the features and targets. The target column must be the last column.
       - folder: Path to the folder where the synthetic images will be saved.
     - Folders with synthetic images
   * - :py:data:`fit_transform(data, folder)`
     - Combines the training and image generation steps. Fits the model to the data and generates synthetic images in one step.

       - data: A path to a CSV file or a Pandas DataFrame containing the features and targets. The target column must be the last column.
       - folder: Path to the folder where the synthetic images will be saved.
     - Folders with synthetic images

- **The model must be fitted** before using the ``transform`` method. If the model isn't fitted, a ``RuntimeError`` will be raised.

Generated Output
----------------

Depending on the selected problem type, the method saves:

- Supervised:
  
  - Images grouped into subfolders by class.
  - A ``supervised.csv`` file with image paths and class labels.

- Unsupervised:
  
  - Images inside an ``images`` folder.
  - An ``unsupervised.csv`` file with image paths.

- Regression:
  
  - Images inside an ``images`` folder.
  - A ``regression.csv`` file with image paths and target values.