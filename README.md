







# Speech Recognition 
- **Kasra Sehhat** [[Email](kasra.sehat@sharif.edu)] [[LinkedIn](https://www.linkedin.com/in/kasra-sehat/)] 
[[GitHub](https://github.com/kasrasehat)]



In this repository different models created for speech2text tested and the state of the art one has been chosen
to be fine tuned on custom dataset. The state of the art model at this time for speech2text purpose is hidden unit bert.
SO, this HUBERT model was chosen in order to be fine-tuned.
# Hubert
## Overview

Hubert was proposed in HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units by Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed.

The abstract from the paper is the following:

Self-supervised approaches for speech representation learning are challenged by three unique problems:
(1) there are multiple sound units in each input utterance, (2) there is no lexicon of input sound units during the pre-training phase, and (3) sound units have variable lengths with no explicit segmentation.
To deal with these three problems, we propose the Hidden-Unit BERT (HuBERT) approach for self-supervised speech representation learning, which utilizes an offline clustering step to provide aligned target labels for a BERT-like prediction loss.
A key ingredient of our approach is applying the prediction loss over the masked regions only, which forces the model to learn a combined acoustic and language model over the continuous inputs.
HuBERT relies primarily on the consistency of the unsupervised clustering step rather than the intrinsic quality of the assigned cluster labels. Starting with a simple k-means teacher of 100 clusters, and using two iterations of clustering, the HuBERT model either matches or improves upon the state-of-the-art wav2vec 2.0 performance on the Librispeech (960h) and Libri-light (60,000h) benchmarks with 10min, 1h, 10h, 100h, and 960h fine-tuning subsets.
Using a 1B parameter model, HuBERT shows up to 19% and 13% relative WER reduction on the more challenging dev-other and test-other evaluation subsets.

# Speech to text
In this project different speech to text models including hubert, wave2vec base and large, facebook s2t and also XLSR which is multi lingual model are implemented and class named s2t was created. In data_prepare .py file, all of these models implemented on dataset.
## each file with the name of model: 
It is an implementation of that model.
### s2t_large_speech_recognizer.py:
In this script facebook s2t model is implemented

### Wav2Vec2-Base-960h.py wav2vec2-large-960h-Lv60.py:
In this script wave2vec2 model is implemented

### xlsr_multilingual_56.py:
In this script xlsr_multilingual model is implemented

### Hubert.py:
It is a script to play with hubert model

### HubertforCTC.py:
In this script HUBERT model is implemented

### speech2text.py:
There is a class named s2t including all above speech2text models.

### train.py:
contains codes including data loader, train and validation parts. 
This script is used to train or fine_tuning model with the data which is created inside this script 
with the help of Data_preprocess.py .So, It will take long time.

###Data_preprocess.py:
It is used to read and preprocess audio data and generate new data with 
text2speech algorithm. 

### data_prepare_pickle.py: 
implements all models on the name of drugs and creates 

### test_models.py:
It is used in order to calculate model accuracy from hint to top five on specific dataset.

### Data_preprocess.py:
It is used for preparing raw speech data in a way that could be fed to the model in dictionary type including label ids that model.loss could compute loss just from this dictionary.

### train_offline.py:
It is used to train hubert model with generated data

### create_data_on_hard_disk1.py:
It is used to generate new audio files with the help of google text2speech algorithm from the name of drugs and store them in a file

### train_in_jupyter.ipynb:
This script is used to create wav2vec2 model and finetune it on custom dataset. But, it has not completed.

### text2speech.py:
It is used to creat audio files from text.

### create_original_data.py:
This script is used to create audio files from name of drugs.

### data_augmentation:
It is used to augment audio files resulted from create_original_data.py

## Explanation of different methods of classes:
### self.process:
It returns normalized speech data that can be fed to the model.

there is a problem in training hubert model. This is very huge model and 
it needs huge amount of ram for training and each loss.backwark process occupy a lot 
of ram. As a result, maybe it is good to use small model for fine tuning. another way used in this repo is train the last layers of the model. till now, best results are derived from training tha last 30 layers of model.

# Definition of built_in function of model:
CTC loss function is used in order to train the weights of the model
```python
        loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
 ```
