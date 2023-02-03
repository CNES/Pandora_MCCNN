Quick overview
==============

The Pandora_MCNN is Pytorch implementation of MCCNN [1]_ neural network which computes a similarity measure on pair of small image patches.
This similarity measure is computed between each possible patch to initialize the cost volume.
There are two architectures: mc-cnn fast and mc-cnn accurate, figures  below detail the networks. Both networks  have the same input and output, the mc-cnn fast network is faster than the mc-cnn accurate network.
The fast architecture uses a fixed similarity measure (dot product) while the accurate architecture attempts to learn a similarity measure.


   .. figure:: ../Images/mc_cnn_architectures.svg

      Left : mc-cnn fast architecture. Right : mc-cnn fast architecture

.. [1] Zbontar, J., & LeCun, Y. (2016). Stereo matching by training a convolutional neural network to compare image patches. J. Mach. Learn. Res., 17(1), 2287-2318.