import numpy as np 
import pandas as pd 
import json


pd_train = pd.read_csv('train.csv')
pd_test = pd.read_csv('test.csv')

train = np.array(pd_train)
test = np.array(pd_test)

def find_all(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1

# Convert training data

output = {}
output['version'] = 'v1.0'
output['data'] = []

for line in train:
    paragraphs = []
    
    context = line[1]
    
    qas = []
    question = line[-1]
    qid = line[0]
    answers = []
    answer = line[2]
    if type(answer) != str or type(context) != str or type(question) != str:
        print(context, type(context))
        print(answer, type(answer))
        print(question, type(question))
        continue
    answer_starts = find_all(context, answer)
    for answer_start in answer_starts:
        answers.append({'answer_start': answer_start, 'text': answer})
    qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
    
    paragraphs.append({'context': context, 'qas': qas})
    output['data'].append({'title': 'None', 'paragraphs': paragraphs})

with open('data/train.json', 'w') as outfile:
    json.dump(output, outfile)


output = {}
output['version'] = 'v1.0'
output['data'] = []

for line in test:
    paragraphs = []
    
    context = line[1]
    
    qas = []
    question = line[-1]
    qid = line[0]
    if type(context) != str or type(question) != str:
        print(context, type(context))
        print(answer, type(answer))
        print(question, type(question))
        continue
    answers = []
    answers.append({'answer_start': 1000000, 'text': '__None__'})
    qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
    
    paragraphs.append({'context': context, 'qas': qas})
    output['data'].append({'title': 'None', 'paragraphs': paragraphs})

with open('data/test.json', 'w') as outfile:
    json.dump(output, outfile)

predictions = json.load(open('results_roberta_large/predictions_.json', 'r'))
submission = pd.read_csv(open('sample_submission.csv', 'r'))
for i in range(len(submission)):
    id_ = submission['textID'][i]
    if pd_test['sentiment'][i] == 'neutral': # neutral postprocessing
        submission.loc[i, 'selected_text'] = pd_test['text'][i]
    else:
        submission.loc[i, 'selected_text'] = predictions[id_]

print(submission.head())

submission.to_csv('submission.csv', index=False)