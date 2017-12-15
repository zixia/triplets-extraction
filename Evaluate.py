# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-
import numpy as np
import cPickle
import pdb
import json
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

def evaluavtion_triple(testresult,index2tag,index2word,epoch):
    total_predict_right=0.
    total_predict=0.
    total_right = 0.
    pre_relation_right=0. #关系预测正确数
    pre_relation_total=0. #预测的关系数
    total_relation_right=0. #正确的关系数

    for sent in testresult: #每句话
        ptag = sent[0]  #每句话预测的标签
        ttag = sent[1]  #每句话真实的标签
        predictrightnum,predictnum,rightnum,relation_right,total_relation,predict_relation= count_sentence_triple_num(ptag,ttag,epoch)
        total_predict_right+=predictrightnum
        total_predict+=predictnum
        total_right += rightnum
        pre_relation_right+=relation_right
        pre_relation_total+=predict_relation
        total_relation_right+=total_relation

    P = total_predict_right /float(total_predict) if total_predict!=0 else 0
    R = total_predict_right /float(total_right)
    F = (2*P*R)/float(P+R) if P!=0 else 0

    P1 = pre_relation_right /float(pre_relation_total) if pre_relation_total!=0 else 0
    R1 = pre_relation_right /float(total_relation_right)
    F1 = (2*P1*R1)/float(P1+R1) if P1!=0 else 0

    if F>0.5:
        for sent in testresult: #每句话
            ptag = sent[0]  #每句话预测的标签
            ttag = sent[1]  #每句话真实的标签
            predictrightnum,predictnum,rightnum,relation_right,total_relation,predict_relation= count_sentence_triple_num(ptag,ttag,epoch)
            total_predict_right+=predictrightnum
            total_predict+=predictnum
            total_right += rightnum
            pre_relation_right+=relation_right
            pre_relation_total+=predict_relation
            total_relation_right+=total_relation
            sentence=[]
            for i in sent[2]:
                if i:
                    sentence.append(index2word[i])
            res={}
            res['sentence']=sentence
            res['prediction']=ptag
            res['ground_truth']=ttag
            if predictrightnum:  #这句话预测对了
                with open('allright.json','a') as f:
                    f.write(json.dumps(res,ensure_ascii=False)+'\n')
            elif relation_right:  #关系预测对了，实体错了
                with open('rightrelation.json','a') as f:
                    f.write(json.dumps(res,ensure_ascii=False)+'\n')
            elif relation_right!=total_relation: #关系预测错了
                with open('error.json','a') as f:
                    f.write(json.dumps(res,ensure_ascii=False)+'\n')

    return P,R,F,P1,R1,F1

def count_sentence_triple_num(ptag,ttag,epoch):
    #transfer the predicted tag sequence to triple index
    #if epoch>100
    #    pdb.set_trace()
    predict_rmpair= tag_to_triple_index(ptag,epoch) #模型预测的实体关系对
    right_rmpair = tag_to_triple_index(ttag,epoch) #正确的实体关系对
    predict_right_num = 0       # the right number of predicted triple
    predict_num = 0     # the number of predicted triples
    right_num = 0
    predict_relation=0 #总预测的关系数
    relation_right=0  #关系预测对了
    total_relation=0  #总正确的关系数
    for type in predict_rmpair:
        predict_relation+=1
        eelist = predict_rmpair[type]
        e1 = eelist[0]
        e2 = eelist[1]
        predict_num += min(len(e1),len(e2)) #这句话模型预测的实体对数

        if right_rmpair.__contains__(type):
            relation_right+=1
            reelist = right_rmpair[type]
            re1 = reelist[0]
            re2 = reelist[1]

            for i in range(0,min(min(len(e1),len(e2)),min(len(re1),len(re2)))):
                if e1[i][0]== re1[i][0] and e1[i][1]== re1[i][1]  \
                        and e2[i][0]== re2[i][0] and e2[i][1]== re2[i][1] :
                    predict_right_num+=1

    for type in right_rmpair:
        total_relation+=1
        eelist = right_rmpair[type]
        e1 = eelist[0]
        e2 = eelist[1]
        right_num += min(len(e1),len(e2))
    return predict_right_num,predict_num,right_num,relation_right,total_relation,predict_relation

def tag_to_triple_index(ptag,epoch):
    #if epoch>100:
    #    pdb.set_trace()
    rmpair={}
    for i in range(0,len(ptag)):
        tag = ptag[i] #每个词的标签
        if not tag.__eq__("O") and not tag.__eq__(""):
            type_e = tag.split("__") #标签包含的实体
            if not rmpair.__contains__(type_e[0]):
                eelist=[]
                e1=[]
                e2=[]
                if type_e[1].__contains__("1"): #这个词的标签包含1
                    if type_e[1].__contains__("S"):
                        e1.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("1") and \
                                ( ptag[j].__contains__("I") or  ptag[j].__contains__("L")):
                                j+=1
                            else:
                                break
                        e1.append((i, j)) #实体1在句子中的位置
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and \
                                ( ptag[j].__contains__("I") or  ptag[j].__contains__("L")):
                                j+=1
                            else:
                                break
                        e2.append((i, j)) #实体2在句子中的位置
                eelist.append(e1)
                eelist.append(e2)
                rmpair[type_e[0]] = eelist
            else:
                eelist=rmpair[type_e[0]]
                e1=eelist[0]
                e2=eelist[1]
                if type_e[1].__contains__("1"):
                    if type_e[1].__contains__("S"):
                        e1.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("1") and \
                                ( ptag[j].__contains__("I") or  ptag[j].__contains__("L")):
                                j+=1
                            else:
                                break
                        e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and \
                                ( ptag[j].__contains__("I") or  ptag[j].__contains__("L")):
                                j+=1
                            else:
                                break
                        e2.append((i, j))
                eelist[0]=e1
                eelist[1]=e2
                rmpair[type_e[0]] = eelist
    return rmpair


if __name__=="__main__":
    resultname = "./data/demo/result/biose-loss5-result-15"
    testresult = cPickle.load(open(resultname, 'rb'))
    P,R,F = evaluavtion_triple(testresult)
    print P,R,F