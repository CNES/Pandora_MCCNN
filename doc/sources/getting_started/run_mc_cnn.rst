Run MCCNN neural network
========================

Cost volume can be computed using the API, see example below.

`run_mc_cnn_fast` and `run_mc_cnn_accurate` functions take as input the images, the disparity range and the path of the pre-trained network.

Before inference, images are normalized. Functions will return a 3-dimension np.array cost volume with size: W, H, D.

Pre-trained weights are available in this package and can be accessed with the function `mc_cnn.weights.get_weights`, but it is also possible to use any other pre-trained weights file (.pt or .pth).

To post-process the cost volume (aggregation, optimization, computing the disparity map ..) please use `Pandora <https://github.com/CNES/Pandora>`_ and its `plugins <https://pandora.readthedocs.io/userguide/plugin.html>`_

.. sourcecode:: python

    from mc_cnn.run import run_mc_cnn_fast, run_mc_cnn_accurate
    from mc_cnn.weights import get_weights
    import rasterio


    if __name__ == '__main__':

        # Read images (only grayscale images are accepted )
        left_image = rasterio.open('Path_to_left_image').read(1)
        right_image = rasterio.open('Path_to_right_image').read(1)

        # Path to the pretrained network
        mccnn_fast_model_path = str(get_weights(arch="fast", training_dataset="middlebury")) # Or custom weights filepath
        mccnn_accurate_model_path = str(get_weights(arch="accurate", training_dataset="middlebury")) # Or custom weights filepath

        # Cost volume using mccnn fast similarity measure
        cost_volume_fast = run_mc_cnn_fast(left_image, right_image, disparity_min, disparity_max, mccnn_fast_model_path)

        # Cost volume using mccnn accurate similarity measure
        cost_volume_accurate = run_mc_cnn_accurate(left_image, right_image, disparity_min, disparity_max, mccnn_accurate_model_path)

