# MC-CNN

MC-CNN est un réseau de neurones qui produit un coût de mise en correspondance entre deux imagettes.

## Installation pour les utilisateurs sur le cluster HAL du CNES

Un environnement conda est mis à disposition dans /work/OT/siaa/3D/Development/rt_corr_deep/conda_environment/mc_cnn_env :


```sh
u@m $ module purge
u@m $ module load python/3.7.2 gdal/2.1.1
u@m $ virtualenv myEnv --no-site-packages
u@m $ source myEnv/bin/activate
(myEnv) u@m $ git clone git@gitlab.cnes.fr:OutilsCommuns/CorrelateurChaine3D/mc-cnn.git
(myEnv) u@m $ pip install -e mc-cnn
```

## Utilisation

### Entrainement des réseaux mc-cnn fast et accurate

```bash
    usage: train.py [-h] [-data_augmentation {True,False}]
                    {accurate,fast} training testing image output_dir

    positional arguments:
      {accurate,fast}       Type of the network : accurate or fast
      training              Path to a hdf5 file containing the training sample
      testing               Path to a hdf5 file containing the testing sample
      image                 Path to a hdf5 file containing the image sample
      output_dir            Output directory

    optional arguments:
      -h, --help            show this help message and exit
      -data_augmentation {True,False}
                            Apply data augmentation ?
```

Les bases d'apprentissages training, testing, image, sont disponibles dans l'espace /work/OT/siaa/3D/Development/rt_corr_deep/.

### Utilisation des réseaux mc-cnn fast et accurate

L'utilisation des réseaux mc-cnn fast et accurate se fait via Pandora, avec le plugin [plugin_MC-CNN](https://gitlab.cnes.fr/OutilsCommuns/CorrelateurChaine3D/pandora_plugins/plugin_mc-cnn).