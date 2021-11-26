# Introduction
These codes are for the paper -- Towards Training Explainable Singing Quality Assessment Network with Augmented Data

[generate_allsamples](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/tree/master/generate_allsamples "generate_allsamples") is used for generating augmented data based on our Chinese(Databaker) and English(NHSS) Datasets. You can use it by running [bad_Chinese.py](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/blob/master/generate_allsamples/Chinese/bad_Chinese.py "bad_Chinese.py") or [bad_English.py](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/blob/master/generate_allsamples/English/bad_English.py "bad_English.py")

[model_for_augmented_method](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/tree/master/model_for_augmented_method "model_for_augmented_method") is the model we used in verification experiment. At first we prepare the audios and pitch files, then use [CreateDill_ph.py](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/blob/master/model_for_augmented_method/hybrid_CRNN/create_data/CreateDill_ph.py "CreateDill_ph.py") to dump all files into one file which will be loaded in training and testing precess. Then [train_cqt_ph.py](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/blob/master/model_for_augmented_method/hybrid_CRNN/train/train_cqt_ph.py "train_cqt_ph.py") is used for training the model and [test_cqt_ph.py](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/blob/master/model_for_augmented_method/hybrid_CRNN/test/test_cqt_ph.py "test_cqt_ph.py") for test the model.

[multi_model](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/tree/master/multi_model "multi_model") is our multitask-framework. It can provide overall score and pitch score. The codes are similar as [model_for_augmented_method](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/tree/master/model_for_augmented_method "model_for_augmented_method") in addition to the structure of neural network.

# Requirement
dill  
numpy  
librosa  
torch  
sklearn  
scipy  
wave  
