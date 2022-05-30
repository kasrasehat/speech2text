import torch
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import tqdm
from torch import nn
from ast import literal_eval


def compare_levenstein(word, dataset):

    for j in range(data.shape[0]):

        if not pd.isna(data.iloc[j].drug):
            dataset.loc[j,('score')] = fuzz.ratio(word, dataset.iloc[j].drug)

    dataset1 = dataset.sort_values('score', ascending=False, inplace=False)
    dataset1 = dataset1.reset_index(drop=True)
    return list(dataset1['drug'][0:5]), list(dataset1['drug'][0:4]), list(dataset1['drug'][0:3]), list(dataset1['drug'][0:2]), list(dataset1['drug'][0:1])

def compare_ctc_loss(path, dataset):

    for j in range(data.shape[0]):

        if not pd.isna(data.iloc[j].drug):
            log_probs = torch.load(path)
            flattened_targets = torch.tensor(literal_eval(dataset.iloc[j].drug_codes))
            input_lengths = torch.tensor(log_probs.shape[0])
            target_lengths = torch.tensor(flattened_targets.shape[-1])
            with torch.no_grad():
                with torch.backends.cudnn.flags(enabled=False):
                    loss = nn.functional.ctc_loss(
                        log_probs,
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=0,
                        reduction='mean',
                        zero_infinity=False,
                    )

            dataset.loc[j, ('ctc_loss')] = loss.item()


    dataset1 = dataset.sort_values('ctc_loss', ascending=False, inplace=False)
    dataset1 = dataset1.reset_index(drop=True)
    return list(dataset1['drug'][0:5]), list(dataset1['drug'][0:4]), list(dataset1['drug'][0:3]), list(dataset1['drug'][0:2]), list(dataset1['drug'][0:1])




data = pd.read_csv('fine_tuned_hubert_120layer_loss4.csv')
data1 = data.copy()
data['score'] = pd.Series(np.nan, index=data.index, dtype=int)
data['ctc_loss'] = pd.Series(np.nan, index=data.index, dtype=int)
tot = 0
top_5 = 0
top_4 = 0
top_3 = 0
top_2 = 0
top_1 = 0
top_5_fine = 0
top_4_fine = 0
top_3_fine = 0
top_2_fine = 0
top_1_fine = 0
bugs_top_5, bugs_hint, bugs_top_5_fine, bugs_hint_fine = [], [], {}, {}

top_5_ctc = 0
top_4_ctc = 0
top_3_ctc = 0
top_2_ctc = 0
top_1_ctc = 0
top_5_fine_ctc = 0
top_4_fine_ctc = 0
top_3_fine_ctc = 0
top_2_fine_ctc = 0
top_1_fine_ctc = 0
bugs_top_5_ctc, bugs_hint_ctc, bugs_top_5_fine_ctc, bugs_hint_fine_ctc = [], [], [], []

for i in tqdm.tqdm(range(data.shape[0])):

    if not pd.isna(data.iloc[i].drug):

        tot += 1
        word1 = data.iloc[i].hubert
        path1 = data.iloc[i].hubert_log_probs
        word2 = data.iloc[i].fine_tuned_hubert
        path2 = data.iloc[i].fine_tuned_hubert_log_probs
        target = data.iloc[i].drug
        drugs = data['drug']
        best5, best4, best3, best2, best1 = compare_levenstein(word1, data)
        if not pd.isna(word2):
            best5_fine, best4_fine, best3_fine, best2_fine, best1_fine = compare_levenstein(word2, data)
        else:
            continue

        if target in best5:
            top_5 += 1
        else:
            bugs_top_5.append(target)

        if target in best4:
            top_4 += 1

        if target in best3:
            top_3 += 1

        if target in best2:
            top_2 += 1

        if target in best1:
            top_1 += 1
        else:
            bugs_hint.append(target)

        if target in best5_fine:
            top_5_fine += 1
        else:
            bugs_top_5_fine[target] = word2

        if target in best4_fine:
            top_4_fine += 1

        if target in best3_fine:
            top_3_fine += 1

        if target in best2_fine:
            top_2_fine += 1

        if target in best1_fine:
            top_1_fine += 1
        else:
            bugs_hint_fine[target] = word2

        ###best5_ctc, best4_ctc, best3_ctc, best2_ctc, best1_ctc = compare_ctc_loss(path1, data)
        # if not pd.isna(path2):
        #     ###best5_fine_ctc, best4_fine_ctc, best3_fine_ctc, best2_fine_ctc, best1_fine_ctc = compare_ctc_loss(path2, data)
        # else:
        #     continue

        # if target in best5_ctc:
        #     top_5_ctc +=1
        # else:
        #     bugs_top_5_ctc.append(target)
        #
        # if target in best4_ctc:
        #     top_4_ctc +=1
        #
        # if target in best3_ctc:
        #     top_3_ctc +=1
        #
        # if target in best2_ctc:
        #     top_2_ctc +=1
        #
        # if target in best1_ctc:
        #     top_1_ctc +=1
        # else:
        #     bugs_hint_ctc.append(target)
        #
        # if target in best5_fine_ctc:
        #     top_5_fine_ctc +=1
        # else:
        #     bugs_top_5_fine_ctc.append(target)
        #
        # if target in best4_fine_ctc:
        #     top_4_fine_ctc +=1
        #
        # if target in best3_fine_ctc:
        #     top_3_fine_ctc +=1
        #
        # if target in best2_fine_ctc:
        #     top_2_fine_ctc +=1
        #
        # if target in best1_fine_ctc:
        #     top_1_fine_ctc +=1
        # else:
        #     bugs_hint_fine_ctc.append(target)

print('top_5 score is {}'.format(top_5/tot))
print('top_4 score is {}'.format(top_4/tot))
print('top_3 score is {}'.format(top_3/tot))
print('top_2 score is {}'.format(top_2/tot))
print('top_1 score is {}'.format(top_1/tot))
print('top_5 fine tuned hubert score is {}'.format(top_5_fine/tot))
print('top_4 fine tuned hubert score is {}'.format(top_4_fine/tot))
print('top_3 fine tuned hubert score is {}'.format(top_3_fine/tot))
print('top_2 fine tuned hubert score is {}'.format(top_2_fine/tot))
print('top_1 fine tuned hubert score is {}'.format(top_1_fine/tot))
print('\nbugs of hint:{}'.format(bugs_hint))
print('\nbugs of top 5:{}'.format(bugs_top_5))
print('\nbugs of hint fine tuned:{}'.format(bugs_hint_fine))
print('\nbugs of top 5 fine tuned:{}'.format(bugs_top_5_fine))


# print('top_5_ctc score is {}'.format(top_5_ctc/tot))
# print('top_4_ctc score is {}'.format(top_4_ctc/tot))
# print('top_3_ctc score is {}'.format(top_3_ctc/tot))
# print('top_2_ctc score is {}'.format(top_2_ctc/tot))
# print('top_1_ctc score is {}'.format(top_1_ctc/tot))
# print('top_5_ctc fine tuned hubert score is {}'.format(top_5_fine_ctc/tot))
# print('top_4_ctc fine tuned hubert score is {}'.format(top_4_fine_ctc/tot))
# print('top_3_ctc fine tuned hubert score is {}'.format(top_3_fine_ctc/tot))
# print('top_2_ctc fine tuned hubert score is {}'.format(top_2_fine_ctc/tot))
# print('top_1_ctc fine tuned hubert score is {}'.format(top_1_fine_ctc/tot))
# print('\nbugs of hint:{}'.format(bugs_hint_ctc))
# print('\nbugs of top 5:{}'.format(bugs_top_5_ctc))
# print('\nbugs of hint fine tuned:{}'.format(bugs_hint_fine_ctc))
# print('\nbugs of top 5 fine tuned:{}'.format(bugs_top_5_fine_ctc))
