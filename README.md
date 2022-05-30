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

###create_data_on_hard_disk1.py:
It is used to generate new audio files from google text2speech algorithm and store them in a file

###train_offline.py:
It is used to train hubert model with generated data

##Explanation of different methods of classes:
###self.process:
It returns normalized speech data that can be fed to the model.

there is a problem in training hubert model. This is very huge model and 
it needs huge amount of ram for training and each loss.backwark process occupy a lot 
of ram. As a result, maybe it is good to use small model for fine tuning.

#issue in model built_in function:
With respect to the results of previous runs it was obvious that there is a problem in calculating amount of loss.
after investigation in the heart of model we found a problem in modeling_hubert.py which returns loss.
 
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
if 