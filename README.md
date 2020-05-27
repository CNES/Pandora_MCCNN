# MC-CNN

MC-CNN [1] est un réseau de neurones qui produit un coût de mise en correspondance entre deux imagettes.

## Installation pour les utilisateurs sur le cluster HAL du CNES


```sh
u@m $ module purge
u@m $ module load python/3.7.2 gdal/2.1.1
u@m $ virtualenv myEnv --no-site-packages
u@m $ source myEnv/bin/activate
(myEnv) u@m $ git clone git@gitlab.cnes.fr:OutilsCommuns/CorrelateurChaine3D/mc-cnn.git
(myEnv) u@m $ pip install -e mc-cnn
```

## Utilisation

### Création des bases d'apprentissage

Les scripts du dossier preprocessing, permettent de créer des bases d'apprentissage hdf5.

### Entrainement des réseaux mc-cnn fast et accurate

```bash
    python mc_cnn/train.py -h
    usage: train.py [-h] injson outdir
    
    positional arguments:
      injson      Input json file
      outdir      Output directory
    
    optional arguments:
      -h, --help  show this help message and exit
```

Le fichier injson contient les paramètres d'entrainement, il est de la forme : 

 ```json
     {
        "network": "accurate",
        "dataset": "middlebury",
        "training_sample": "training_dataset.hdf5",
        "training_image": "images.hdf5",
        "testing_sample": "testing_dataset.hdf5",
        "testing_image": "images.hdf5",
        "dataset_neg_low": 1.5,
        "dataset_neg_high": 18,
        "dataset_pos": 0.5,
        "data_augmentation": false,
    
        "augmentation_param":{
          "scale": 0.8,
          "hscale": 0.8,
          "hshear": 0.1,
          "trans": 0,
          "rotate": 28,
          "brightness": 1.3,
          "contrast": 1.1,
          "d_hscale": 0.9,
          "d_hshear": 0.3,
          "d_vtrans": 1,
          "d_rotate": 3,
          "d_brightness": 0.7,
          "d_contrast": 1.1
        }
    }
 ```

Des exemples sont disponibles dans le dossier training_config.

### Utilisation des réseaux mc-cnn fast et accurate

L'utilisation des réseaux mc-cnn fast et accurate se fait via Pandora, avec le plugin [plugin_MC-CNN](https://gitlab.cnes.fr/OutilsCommuns/CorrelateurChaine3D/pandora_plugins/plugin_mc-cnn).


[1][ŽBONTAR, Jure et LECUN, Yann. Stereo matching by training a convolutional neural network to compare image patches. The journal of machine learning research, 2016, vol. 17, no 1, p. 2287-2318.]