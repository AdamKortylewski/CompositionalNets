# Compositional Convolutional Neural Networks: A Deep Architecture with Innate Robustness to Partial Occlusion [CVPR-2020]
![alt text](demo/17029_0_val2017_predclass_car_and_occluder_map.jpg "Title")

```
Compositional Convolutional Neural Networks: A Deep Architecture with Innate Robustness to Partial Occlusion
Adam Kortylewski, Ju He, Qing Liu, Alan Yuille
CVPR 2020
```

### Release Notes

This is a port of our original code from Tensorflow to PyTorch. 
The code is a lot faster and cleaner compared to the original code base. 
The results are a little different from the ones reported in the paper. 
In particular, the performance is a little lower for low occlusion and higher for stronger occlusion.
On average the results are slightly better than reported in the paper.

For now, we provide pretrained models for CompositionalNets trained from the VGG-16 pool5 layer.
Training CompositionalNets for other backbones and layers should be possible but has not been extensively tested so far.
 

### Installation

The code uses **Python 3.6** and it is tested on PyTorch GPU version 1.2, with CUDA-10.0 and cuDNN-7.5.

### Setup CompNet Virtual Environment

```
virtualenv --no-site-packages <your_home_dir>/.virtualenvs/CompNet
source <your_home_dir>/.virtualenvs/CompNet/bin/activate
```

### Clone the project and install requirements

```
git clone https://github.com/AdamKortylewski/CompositionalNets.git
cd CompositionalNets
pip install -r requirements.txt
```

## Download models

* Download pretrained CompNet weights from [here](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/akortyl1_jh_edu/EYH4UDvQnQ9Ettu7cBQAfZoBFLU0gZeredTmfUssMJCrKg?e=HqxXAs) and copy them inside the **models** folder.

* The repositroy contains a few images for the demo script. If you want to evaluate on the full datasets used in our paper you need to download the data [here](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/akortyl1_jh_edu/ETsbJHY58hxDjjM-qL9KUU4BsTU1ZlieevTayqJPhFMj9Q?e=Mrf4LQ) and copy it inside the **data** folder.

## Demo

CompNets require a tight crop of the object in the image. We provide sample images in the **demo** folder 
which are taken from [MS-COCO](http://cocodataset.org/).

#### Run the demo code
```
python Code/demo.py 
```

Our demo script classifies the images from the **demo** folder, extracts the predicted location of occluders, and writes the results back into the **demo** folder.
 

#### Evaluate the classification performance of a model

Run the following command from the terminal to evaluate a model on the full test dataset:
```
python Code/test.py 
```

#### Evaluate the occluder localization performance of a model

If you want to test occluder localization run:
```
python Code/eval_occlusion_localization.py
``` 
This will output qualitative occlusion localization results for each image and a quantitative analysis over all images 
as ROC curve.

## Initializing CompositionalNet Parameters

We initialize CompositionalNets (i.e. the vMF kernels and mixture models) by clustering the training data. 
In particular, we initialize the vMF kernels by clustering the feature vectors:

```
python Initialization_Code/vMF_clustering.py
``` 

Furthermore, we initialize the mixture models by EM-type learning.
The initial cluster assignment for the EM-type learning is computed based on the similarity of the vMF encodings of the training images.
To compute the similarity matrices use:
 
```
python Initialization_Code/comptSimMat.py
``` 

As this process takes some time we provide precomputed similarity matrices [here](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/akortyl1_jh_edu/EU6OcwaW7l1IhpggHJBCjeIBB_xLd28bDUIcoPHKUOhxqg?e=5k34Nx), you need to copy them into the 'models/init_vgg/' folder.
Afterwards you can compute the initialization of the mixture models by executing:

```
python Initialization_Code/Learn_mix_model_vMF_view.py
```


## Referencing CompositionalNets

Please cite the following paper if you use the code directly or indirectly in your research/projects.
```
@inproceedings{CompNet:CVPR:2020,
title = {Compositional Convolutional Neural Networks: A Deep Architecture with Innate Robustness to Partial Occlusion},
author = {Kortylewski, Adam and He, Ju and Liu, Qing and and Yuille, Alan},
booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
month = jun,
year = {2020},
month_numeric = {6}
}
```

## Relation to our Prior Work

With very small modifications this code would also enable the learning and testing of CompositionalNets with
Bernoulli distributions as proposed in our previous work:
  ```
@inproceedings{Combining:WACV:2020,
title = {Combining Compositional Models and Deep Networks For Robust Object Classification under Occlusion},
author = {Kortylewski, Adam and Liu, Qing and Wang, Huiyu and Zhang, Zhishuai and Yuille, Alan},
booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
month = mar,
year = {2020},
month_numeric = {3}
}
```

## Contact

If you have any questions you can contact Adam Kortylewski.

## Acknowledgement

We thank Zhishuai Zhang for helping us speed up and clean the code for the release.
