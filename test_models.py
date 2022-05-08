from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import tqdm


def compare(word, dataset):
    for j in range(data.shape[0]):
        if not pd.isna(data.iloc[j].drug):
            dataset.loc[j,('score')] = fuzz.ratio(word, dataset.iloc[j].drug)

    dataset1 = dataset.sort_values('score', ascending=False, inplace=False)
    dataset1 = dataset1.reset_index(drop=True)
    return list(dataset1['drug'][0:5]), list(dataset1['drug'][0:4]), list(dataset1['drug'][0:3]), list(dataset1['drug'][0:2]), list(dataset1['drug'][0:1])


data = pd.read_csv('final_file.csv')
data1 = data.copy()
data['score'] = pd.Series(np.nan, index=data.index, dtype=int)
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
bugs_top_5, bugs_hint, bugs_top_5_fine, bugs_hint_fine = [], [], [], []

for i in tqdm.tqdm(range(data.shape[0])):

    if not pd.isna(data.iloc[i].drug):

        tot += 1
        word1 = data.iloc[i].hubert
        word2 = data.iloc[i].fine_tuned_hubert
        target = data.iloc[i].drug
        drugs = data['drug']
        best5, best4, best3, best2, best1 = compare(word1, data)
        best5_fine, best4_fine, best3_fine, best2_fine, best1_fine = compare(word2, data)

        if target in best5:
            top_5 +=1
        else: bugs_top_5.append(target)

        if target in best4:
            top_4 +=1

        if target in best3:
            top_3 +=1

        if target in best2:
            top_2 +=1

        if target in best1:
            top_1 +=1
        else: bugs_hint.append(target)

        if target in best5_fine:
            top_5_fine +=1
        else: bugs_top_5_fine.append(target)

        if target in best4_fine:
            top_4_fine +=1

        if target in best3_fine:
            top_3_fine +=1

        if target in best2_fine:
            top_2_fine +=1

        if target in best1_fine:
            top_1_fine +=1
        else: bugs_hint_fine.append(target)


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
