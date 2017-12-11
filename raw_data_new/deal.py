# -*- coding: utf-8 -*-
import pdb
import pandas as pd
import re
import json
import random

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
Valid=[]
Invalid=[]

def aaa(left,right,num,relation,temp):
	if right-left==1:
		temp['tags'][left]=relation+'__E{}S'.format(num)
	else:
		temp['tags'][left]=relation+'__E{}B'.format(num)
		temp['tags'][right]=relation+'__E{}L'.format(num)
		for i in range(left+1,right):
			temp['tags'][i]=relation+'__E{}I'.format(num)
	return temp

def deal(file):
	for sentence,raw_relation,entity1_b,entity1_e,entity2_b,entity2_e in zip(file['sentences'],file['raw_relations'],file['entity1_b'],file['entity1_e'],file['entity2_b'],file['entity2_e']):
		temp={}
		sentence=re.sub('(\<e1\>|\</e1\>|\<e2\>|\</e2\>)','',sentence)
		text=sentence.split()
		# words.update(text)
		temp['tokens']=text
		temp['tags']=['O']*len(text)
		relation=re.sub('\(.*\)','',raw_relation)
		if 'Other' not in raw_relation:
			if raw_relation.find('1')<raw_relation.find('2'):
				temp=aaa(entity1_b,entity1_e,1,relation,temp)
				temp=aaa(entity2_b,entity2_e,2,relation,temp)
			else:
				temp=aaa(entity1_b,entity1_e,2,relation,temp)
				temp=aaa(entity2_b,entity2_e,1,relation,temp)
			Valid.append(temp)
		else:
			Invalid.append(temp)

deal(train)
deal(test)
train_list=[]
test_list=[]
length=len(Valid)+len(Invalid)
flag=34000/length
train_list=Valid[:int(len(Valid)*flag)]+Invalid[:int(len(Invalid)*flag)]
test_list=Valid[int(len(Valid)*flag):]+Invalid[int(len(Invalid)*flag):]
random.shuffle(train_list)
random.shuffle(test_list)
with open('train_tag.json','a') as f:
	for _ in train_list:
		f.write(json.dumps(_,ensure_ascii=False)+'\n')
with open('test_tag.json','a') as f:
	for _ in test_list:
		f.write(json.dumps(_,ensure_ascii=False)+'\n')