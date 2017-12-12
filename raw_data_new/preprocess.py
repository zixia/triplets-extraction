import pdb
import jieba
import pandas as pd
import re
from collections import defaultdict


labelsMapping = {'Other':0,
                 '担保抵押(e1,e2)':1,'担保抵押(e2,e1)':1,
                }
labelsMapping = defaultdict(lambda: 0, labelsMapping)

class Preprocess(object):
    def __init__(self, train_or_test="train"):
        self.train_or_test = train_or_test

    def load_data(self):
        if(self.train_or_test=="train"):
            ifile = open('financial.txt','r')
        else:
            raise NotImplemented
        data = ifile.readlines()
        ifile.close()
        self.raw_data = data

    def process(self):
        sentences = []
        raw_sentences = []
        relations = []
        raw_relations = []
        e11 = []
        e12 = []
        e21 = []
        e22 = []
        ids = []
        
        data = self.raw_data

        num = 1
        for i,l in enumerate(data):
            tmp = l.strip().split('\t')
            #############把子公司、母公司排除掉，因为数据噪音很大
            if(labelsMapping[tmp[1]]==2):
                continue
            raw_relations.append(tmp[1])
            relations.append(labelsMapping[tmp[1]])

            sentence = tmp[0]
            raw_sentences.append(sentence)

            sentence = re.sub(r"[0-9A-Za-z\-]{8-20}",r"CONTRACT_NUM", sentence)
            sentence = re.sub(r"(?<![</]e)([0-9]+?)，?[0-9]*",r"NUM",sentence)

            sentence = " ".join(jieba.cut(sentence))
            sentence = re.sub(r"< e1 >", r"<e1>", sentence)
            sentence = re.sub(r"< e2 >", r"<e2>", sentence)
            sentence = re.sub(r"< / e1 >", r"</e1>", sentence)
            sentence = re.sub(r"< / e2 >", r"</e2>", sentence)
            sentence = re.sub(r"(\s+)", r" ", sentence)
            sentences.append(sentence)

            for pos, w in enumerate(sentence.split()):
                if(w=='<e1>'):
                    e11.append(pos)
                elif(w=='</e1>'):
                    e12.append(pos-2)
                elif(w=='<e2>'):
                    e21.append(pos-2)
                elif(w=='</e2>'):
                    e22.append(pos-4)
                    break
            if(len(e11)!=len(e12) or len(e12)!=len(e21) or len(e21)!=len(e22)):
                pdb.set_trace()
            ids.append(num)
            num+=1
        


        self.data = pd.DataFrame()
        self.data['id'] = ids
        self.data['relations'] = relations
        self.data['raw_relations'] = raw_relations
        #self.data['raw_sentences'] = raw_sentences
        self.data['sentences'] = sentences
        self.data['entity1_b'] = e11
        self.data['entity1_e'] = e12  
        self.data['entity2_b'] = e21
        self.data['entity2_e'] = e22
        if(self.train_or_test=="train"):
            pdb.set_trace()
            self.data[:-2500].to_csv('train.csv',index=False)
            self.data[-2500:].to_csv("test.csv",index=False)

if __name__=='__main__':
    prep = Preprocess("train")
    prep.load_data()
    prep.process()
            
            

