Run MCCNN neural network
========================

Cost volume can be computed using the API, see example below.

`run_mc_cnn_fast` and `run_mc_cnn_accurate` functions take as input the images, the disparity range and the path of the pre-trained network.
Before inference, images are normalized. Functions will return a 3-dimension np.array cost volume with size: W, H, D.

To post-process the cost volume (aggregation, optimization, computing the disparity map ..) please use `Pandora <https://github.com/CNES/Pandora>`_ and its `plugins <https://pandora.readthedocs.io/userguide/plugin.html>`_

.. sourcecode:: python

    from mccnn.run import run_mc_cnn_fast, run_mc_cnn_accurate
    import rasterio


    if __name__ == '__main__':

        # Read images (only grayscale images are accepted )
        left_image = rasterio.open('Path_to_left_image').read(1)
        right_image = rasterio.open('Path_to_right_image').read(1)

        # Path to the pretrained network
        mccnn_fast_model_path = 'Path_to_pretrained_mccnn_fast_network'
        mccnn_accurate_model_path = 'Path_to_pretrained_mccnn_accuarte_network'

        # Cost volume using mccnn fast similarity measure
        cost_volume_fast = run_mc_cnn_fast(left_image, right_image, disparity_min, disparity_max, mccnn_fast_model_path)

        # Cost volume using mccnn accurate similarity measure
        cost_volume_accurate = run_mc_cnn_accurate(left_image, right_image, disparity_min, disparity_max, mccnn_accurate_model_path)

