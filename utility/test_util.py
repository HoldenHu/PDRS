import numpy as np
import json

kuword_kuid_mapping = np.load('Data/kuword_kuid_mapping.textrank.npy', allow_pickle=True)
kuword_kuid_mapping = kuword_kuid_mapping.item()
kuid_kuword_mapping = {v: k for k, v in kuword_kuid_mapping.items()}

courseid_kuid_mapping = np.load('Data/courseid_kuid_mapping.textrank.npy', allow_pickle=True)
courseid_kuid_mapping = courseid_kuid_mapping.item()

kuid_kuid_mapping = np.load('Data/demo/kuid_kuid_mapping.textrank.npy', allow_pickle=True)
kuid_kuid_mapping = kuid_kuid_mapping.item()

dataset_json = json.load(open('Data/dataset.filter4.json'))

def sigmoid(X):
  return 1.0 / (1 + np.exp(-float((X-1))));

kuid_kuid_rd = {}
for key in kuid_kuid_mapping:
  ku1 = key[0]
  ku2 = key[1]
  if (ku2,ku1) in kuid_kuid_mapping:
    score = float(kuid_kuid_mapping[key]) / float(kuid_kuid_mapping[(ku2,ku1)])
  else:
    score = float(kuid_kuid_mapping[key])
  kuid_kuid_rd[key] = sigmoid(score)

def get_courseSet_linkingScore(cset1, cset2):
  # 返回一个0-10的score
  if type(cset1) != list:
    cset1 = [cset1]
  if type(cset2) != list:
    cset2 = [cset2]
  ku_list1 = []
  ku_list2 = []
  for cid1 in cset1:
    ku_list1 = ku_list1 + courseid_kuid_mapping[cid1]
  for cid2 in cset2:
    ku_list2 = ku_list2 + courseid_kuid_mapping[cid2]
  a = 0.0
  b = 0.0
  if len(ku_list1) == 0 or len(ku_list2) == 0:
    # print("There is empty ku set")
    return 0
  for ku1 in ku_list1:
    for ku2 in ku_list2:
      if (ku1,ku2) in kuid_kuid_rd:
        a += kuid_kuid_rd[(ku1,ku2)]
      b += 1
  return int(round(float(a)/float(b), 1) * 10)

def get_knowledgeScore(uid, cid):
  # input: int, int
  # output: prior score, target score
  str_index = str(uid*2)
  hist_clist = dataset_json['hist_course_list'][str_index]
  target_cid = dataset_json['last_course'][str_index]
  prior_score = get_courseSet_linkingScore(hist_clist, 'c'+str(cid))
  target_score = get_courseSet_linkingScore('c'+str(cid), target_cid)
  return (prior_score, target_score)