# Constrained Neural Style Transfer for Decorated Logo Generation

This is the repository for the implementation of ["Constrained Neural Style Transfer for Decorated Logo Generation"](https://arxiv.org/pdf/1803.00686.pdf) by G. Atarsaikhan, B. K. Iwana and S. Uchida.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Running the code](#running-the-code)
4. [Citation](#citation)

## Introduction


## Requirements

* Python >= 3.5

* TensorFlow >= 1.8

* Numpy & Scipy

* OpenCV >= 3.x (cv2)

* Matplotlib

* Download the pre-trained weights for VGG network from [here](https://drive.google.com/open?id=1iF4oKdb-5-45AAmGIwaJyMNcjI9xJZ2i), and place on the main folder. (~500MB)

## Running the code

### Style transfer

```
python StyleTransfer.py -CONTENT_IMAGE <path_to_content_image> -STYLE_IMAGE <path_to_style_image> 
```
Other parser arguments:
```
alpha = 0.001 # Override with -alpha
beta  = 0.8   # Override with -beta
gamma = 0.001 # Override with -gamma
IMAGE_WIDTH = 400 # Override with -width
w1~w5 = 1 # Override with -w1 ~ -w5
```

```
python StyleTransfer.py -CONTENT_IMAGE input/contents/animal.jpg -STYLE_IMAGE input/styles/bells.jpg
```

## Citation

B. K. Iwana, M. Mori, A. Kimura, and S. Uchida, "Introducing Local Distance-Based Features to Temporal Convolutional Neural Networks," in *Int. Conf. Frontiers in Handwriting Recognition*, 2018.

G. Atarsaikhan, B. K. Iwana and S. Uchida "Constrained Neural Style Transfer for Decorated Logo Generation" in the proceedings of DAS2018.

```
@article{iwana2018introducing,
  title={Introducing Local Distance-Based Features to Temporal Convolutional Neural Networks
},
  author={Iwana, Brian Kenji and Mori, Minoru and Kimura, Akisato and Uchida, Seiichi},
  booktitle={Int. Conf. Frontiers in Handwriting Recognition},
  year={2018}
}
```
