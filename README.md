# Certifying Confidence via Randomized Smoothing

This repository contains code for the paper [Certifying Confidence via Randomized Smoothing](https://arxiv.org/abs/2009.08061) by Aounon Kumar,
Alexander Levine, Soheil Feizi and Tom Goldstein. It is built on the publically-availabe code repository
for the paper [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/abs/1902.02918) by (Cohen et al. 2019).

Randomized smoothing has been shown to provide good certified-robustness guarantees for high-dimensional classification problems.
It uses the probabilities of predicting the top two most-likely classes around an input point under a smoothing distribution
to generate a certified radius for a classifier's prediction. However, most smoothing methods do not give us any information
about the confidence with which the underlying classifier (e.g., deep neural network) makes a prediction. This work proposes
a method to generate certified radii for the prediction confidence of the smoothed classifier.

Follow the instructions listed by Cohen et al. 2019 in their [GitHub repository](https://github.com/locuslab/smoothing)
under [Getting started](https://github.com/locuslab/smoothing#getting-started) to set up the project. Obtain a copy of
the pre-trained models from [here](https://drive.google.com/file/d/1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4/view?usp=sharing)
and extract them into a directory.

To certify average prediction score of a ResNet-110 model on the CIFAR-10 dataset, run:
```
python code/certify.py cifar10 [path-to-models]/models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.25 log_files/cifar10/pred_score/noise_0.25 --skip 20 --batch 400 
python code/certify.py cifar10 [path-to-models]/models/cifar10/resnet110/noise_0.50/checkpoint.pth.tar 0.50 log_files/cifar10/pred_score/noise_0.50 --skip 20 --batch 400 
```

To repeat the same experiments with margin as the confidence measure, use:
```
python code/certify.py cifar10 [path-to-models]/models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.25 log_files/cifar10/margin/noise_0.25 --skip 20 --batch 400 --confidence_measure margin
python code/certify.py cifar10 [path-to-models]/models/cifar10/resnet110/noise_0.50/checkpoint.pth.tar 0.50 log_files/cifar10/margin/noise_0.50 --skip 20 --batch 400 --confidence_measure margin
```

To run the same experiments for ImageNet with a ResNet-50 model, download ImageNet and preprocess the `val` directory to look
like the `train` directory by running [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).
Set the `IMAGENET_DIR` environment variable to the location of the ImageNet dataset. Modify the above commands used for
the CIFAR-10 experiments as shown below. 
```
export IMAGENET_DIR=[path-to-imagenet]
python code/certify.py imagenet [path-to-models]/models/imagenet/resnet50/noise_0.25/checkpoint.pth.tar 0.25 log_files/imagenet/pred_score/noise_0.25 --skip 100 --batch 400
python code/certify.py imagenet [path-to-models]/models/imagenet/resnet50/noise_0.50/checkpoint.pth.tar 0.50 log_files/imagenet/pred_score/noise_0.50 --skip 100 --batch 400
python code/certify.py imagenet [path-to-models]/models/imagenet/resnet50/noise_0.25/checkpoint.pth.tar 0.25 log_files/imagenet/margin/noise_0.25 --skip 100 --batch 400 --confidence_measure margin
python code/certify.py imagenet [path-to-models]/models/imagenet/resnet50/noise_0.50/checkpoint.pth.tar 0.50 log_files/imagenet/margin/noise_0.50 --skip 100 --batch 400 --confidence_measure margin
```

Finally, to compute certified accuracies use the file cert_acc.py:
```
python cert_acc.py [path to log file] [score to certify]
```
For example:
```
python cert_acc.py log_files/imagenet/pred_score/noise_0.25 0.5
```