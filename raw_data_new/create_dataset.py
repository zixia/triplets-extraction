import pdb
import pandas as pd
import re

id2name = {-1:"Others",1:"商品交易类",2:"资产交易类",3:"提供或接受劳务",4:"代理委托",5:"资金交易",6:"担保抵押",7:"租赁",8:"托管经营（管理方面）",9:"赠与",10:"非货币交易",13:"股权交易",15:"债权债务类交易",17:"合作项目",18:"许可协议",19:"研究与开发成果",20:"关键管理人员报酬 ",21:"其他事项"}

df = pd.read_csv('direct_result_all.csv')

ofile = open('financial.txt','w')
for sentence,relation in zip(df['Notes'], df['Repat']):
    print(sentence)
    pdb.set_trace()
    relation = id2name[relation]
    #检测<e1>和<e2>出现的次数
    if(sentence.count('<e1>')!=1 or sentence.count('<e2>')!=1):
        continue
    if(sentence.find('<e1>')<sentence.find('<e2>')):
        relation = relation+'(e1,e2)'
    else:
        relation = relation+'(e2,e1)'
        sentence = re.sub(r'<e2>', r'<e3>',sentence)
        sentence = re.sub(r'</e2>', r'</e3>',sentence)
        sentence = re.sub(r'<e1>', r'<e2>',sentence)
        sentence = re.sub(r'</e1>', r'</e2>',sentence)
        sentence = re.sub(r'<e3>', r'<e1>',sentence)
        sentence = re.sub(r'</e3>', r'</e1>',sentence)

    if(sentence.find('<e2>')<sentence.find('</e1>')):
        continue
    ofile.write(sentence+'\t'+relation+'\n')
    
ofile.close()


