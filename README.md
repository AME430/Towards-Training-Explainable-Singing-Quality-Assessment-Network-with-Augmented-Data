# Introduction

These codes are for the paper -- Towards Training Explainable Singing Quality Assessment Network with Augmented Data

Jinhu Li, Chitralekha Gupta, and Haizhou Li, "Towards Training Explainable Singing Quality Assessment Network with Augmented Data", in Proceedings of APSIPA ASC 2021, Japan.


# Requirement

dill
numpy
librosa
torch
sklearn
scipy
wave

# Usage

If you want to run our code, you need to get the DAMP subset, Databaker and NHSS dataset.

DAMP subset is provided in this link: https://github.com/chitralekha18/SingEval.git

**How to generate augmented data**

[generate_allsamples](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/tree/master/generate_allsamples "generate_allsamples") is used for generating augmented data based on our Chinese(Databaker) and English(NHSS) Datasets. You can use it by running [bad_Chinese.py](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/blob/master/generate_allsamples/Chinese/bad_Chinese.py "bad_Chinese.py") or [bad_English.py](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/blob/master/generate_allsamples/English/bad_English.py "bad_English.py")

**Objective verification experiment**

[model_for_augmented_method](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/tree/master/model_for_augmented_method "model_for_augmented_method") is the model we used in verification experiment. At first we prepare the audios and pitch files, then use [CreateDill_ph.py](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/blob/master/model_for_augmented_method/hybrid_CRNN/create_data/CreateDill_ph.py "CreateDill_ph.py") to dump all files into one file which will be loaded in training and testing precess. Then [train_cqt_ph.py](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/blob/master/model_for_augmented_method/hybrid_CRNN/train/train_cqt_ph.py "train_cqt_ph.py") is used for training the model and [test_cqt_ph.py](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/blob/master/model_for_augmented_method/hybrid_CRNN/test/test_cqt_ph.py "test_cqt_ph.py") for test the model.

**Explainable freamwork**

[multi_model](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/tree/master/multi_model "multi_model") is our multitask-framework. It can provide overall score and pitch score. The codes are similar as [model_for_augmented_method](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/tree/master/model_for_augmented_method "model_for_augmented_method") in addition to the structure of neural network. So you can use the same steps to run this code.
