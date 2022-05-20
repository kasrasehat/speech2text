#Speech to text
In this project different speech to text models including hubert, wave2vec base and large, facebook s2t and also XLSR which is multi lingual model are implemented and class named s2t was created. In data_prepare .py file, all of these models implemented on dataset.
###each file with the name of model: 
There is an implementation of that model. 
###train.py
contains codes including data loader, train and validation parts. This script is used to train or fine_tuning model.
###data_prepare_pickle.py: 
implements all models on the name of drugs and creates 
###test_models.py:
It is used in order to calculate model accuracy from hint to top five on specific dataset.
###Data_preprocess.py:
It is used for preparing raw speech data in a way that could be fed to the model in dictionary type including label ids that model.loss could compute loss just from this dictionary.
###speech2text.py:
There is a class named s2t including speech2text models.
###text2speech.py:
It is used to creat audio files from text.

##Explanation of different methods of classes:
###self.process:
It returns normalized speech data that can be fed to the model.