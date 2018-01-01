# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'
import os
import sys
import pickle
import os.path
import json
import pdb
import numpy as np
import tensorflow as tf
from PrecessEEdata import get_data_e2e
from Evaluate import evaluavtion_triple
from LSTM_layer import encoderLSTM, decoderLSTM
from tensorflow.contrib.rnn import LSTMCell
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# reload(sys)
# sys.setdefaultencoding( "utf-8" )

def data_type():
    return tf.float32

class JointModel(object):
    """Joint extraction Model"""
    def __init__(self,seq_length,target2id,id2target,
        embedding_matrix,keep_prob,bilstm_dim,
        init_lr=0.001,is_training=True,npochos=100,max_grad_norm=5):
        """
        npochos：训练轮数
        is_training：是否是训练状态
        max_grad_norm：控制最大梯度膨胀
        seq_length：每句话的最大长度
        target2id：每个tag对应的id
        id2target：每个id对应的tag
        embedding_matrix：embedding层的权重矩阵
        bilstm_dim：lstm层维度
        init_lr：起始learning rate
        """
        num_steps=seq_length
        self._inputs=tf.placeholder(tf.int32,[None,num_steps])
        self._y_label=tf.placeholder(tf.float32,[None,num_steps,len(target2id)+1])

        #embedding层
        with tf.device('/cpu:0'):
            embedding=tf.get_variable('embedding',embedding_matrix.shape,initializer=tf.constant_initializer(embedding_matrix),dtype=data_type())
            embedding_output=tf.nn.embedding_lookup(embedding,self._inputs)
        if is_training and keep_prob<1.0:
            embedding_output=tf.nn.dropout(embedding_output,keep_prob)

        #bilstm层
        bilstm_input=tf.unstack(embedding_output,num_steps,1)
        forward_lstm=encoderLSTM(bilstm_dim,initializer=tf.random_uniform_initializer(-0.01, 0.01),forget_bias=0.0)
        backward_lstm=encoderLSTM(bilstm_dim,initializer=tf.random_uniform_initializer(-0.01, 0.01), forget_bias=0.0)
        bilstm_output=tf.nn.static_bidirectional_rnn(forward_lstm,backward_lstm,bilstm_input,dtype=data_type())[0]
        
        #decoder_lstm层
        LSTMd_input=tf.stack(bilstm_output,axis=1)
        LSTMd=decoderLSTM(bilstm_dim*2,initializer=tf.random_uniform_initializer(-0.01, 0.01), forget_bias=0.0)
        LSTMd_output=tf.nn.dynamic_rnn(LSTMd,LSTMd_input,dtype=tf.float32)[0]

        #softmax层
        softmax_input=tf.reshape(LSTMd_output,[-1,LSTMd_output.shape[-1]])
        softmax_W=tf.get_variable('softmax_W',[bilstm_dim*2,len(target2id)+1],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=data_type())
        softmax_b=tf.get_variable('softmax_b',[len(target2id)+1],initializer=tf.random_uniform_initializer(-0.01, 0.01),dtype=data_type())
        y_pre=tf.nn.softmax(tf.matmul(softmax_input,softmax_W)+softmax_b)

        self._preditions=y_pre
        self._loss=loss=tf.reduce_mean(-tf.reduce_sum(tf.reshape(self._y_label,[-1,self._y_label.shape[-1]])*tf.log(y_pre),reduction_indices=[1]))
        if not is_training:
            return

        global_step=tf.Variable(0)
        learning_rate = tf.train.exponential_decay(init_lr, global_step, npochos*100, 0.98, staircase=True)

        #batch norm
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),max_grad_norm)

        optimizer=tf.train.RMSPropOptimizer(learning_rate)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=global_step)

    @property
    def predictions(self):
        return self._preditions

    @property
    def train_step(self):
        return self._train_op

    @property
    def loss(self):
        return self._loss

    @property
    def inputs(self):
        return self._inputs

    @property
    def y_label(self):
        return self._y_label

def get_training_batch_xy_bias(inputsX, inputsY, max_s, max_t,
                          batchsize, vocabsize, target_idex_word,lossnum,shuffle=False):
    assert len(inputsX) == len(inputsY)
    indices = np.arange(len(inputsX))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputsX) - batchsize + 1, batchsize):
        # pdb.set_trace()
        excerpt = indices[start_idx:start_idx + batchsize]
        x = np.zeros((batchsize, max_s)).astype('int32')
        y = np.zeros((batchsize, max_t, vocabsize + 1)).astype('int32')
        for idx, s in enumerate(excerpt):
            x[idx,] = inputsX[s]
            for idx2, word in enumerate(inputsY[s]):
                targetvec = np.zeros(vocabsize + 1)
                wordstr=''
                if word!=0:
                    wordstr = target_idex_word[word]
                if wordstr.__contains__("E"):
                    targetvec[word] = lossnum
                else:
                    targetvec[word] = 1
                y[idx, idx2,] = targetvec
        yield x, y

"""
def test_model(nn_model,testdata,index2tag,index2word,session,arg,epoch,flag='test on testdata'):
    print(flag)
    index2tag[0]=''
    testx = np.asarray(testdata[0],dtype="int32")
    testy = np.asarray(testdata[1],dtype="int32")

    batch_size=50
    testlen = len(testx)
    testlinecount=0
    if len(testx)%batch_size ==0:
        testnum = len(testx)/batch_size
    else:
        extra_test_num = batch_size - len(testx)%batch_size
        extra_data = testx[:extra_test_num]
        testx=np.append(testx,extra_data,axis=0)
        extra_data = testy[:extra_test_num]
        testy=np.append(testy,extra_data,axis=0)
        testnum = len(testx)/batch_size

    testresult=[]
    for n in range(0,testnum):
        xbatch = testx[n*batch_size:(n+1)*batch_size]
        ybatch = testy[n*batch_size:(n+1)*batch_size]
        predictions=nn_model.predictions.eval(feed_dict={nn_model.inputs:xbatch},session=session)
        predictions=predictions.reshape([batch_size,predictions.shape[0]/batch_size,predictions.shape[-1]])
        for si in range(0,len(predictions)):
            if testlinecount < testlen:
                sent = predictions[si]
                ptag = [] #预测的标签
                for word in sent:
                    next_index = np.argmax(word)
                    if next_index != 0:
                        next_token = index2tag[next_index]
                        ptag.append(next_token)
                senty = ybatch[si]
                ttag=[] #实际的标签
                for word in senty:
                    next_token = index2tag[word]
                    ttag.append(next_token)
                result = [] #result[0]=ptag result[1]=ttag
                result.append(ptag)
                result.append(ttag)
                testlinecount += 1
                testresult.append(result)
    # pickle.dump(testresult,open(resultfile,'wb'))
    P, R, F = evaluavtion_triple(testresult,epoch)
    print(P, R, F)
    with open('./data/prf{}.log'.format(arg),'a') as f:
        f.write(flag+'\n')
        f.write(str(P)+' '+str(R)+' '+str(F)+'\n')
    return P, R, F
"""

def test_model(nn_model,testdata,index2tag,index2word,session,arg,epoch,flag='test on testdata'):
    print(flag)
    index2tag[0]=''
    xbatch = np.asarray(testdata[0],dtype="int32")
    ybatch = np.asarray(testdata[1],dtype="int32")
    testresult=[]
    batch_size=xbatch.shape[0]
    predictions=nn_model.predictions.eval(feed_dict={nn_model.inputs:xbatch},session=session)
    predictions=predictions.reshape([
        batch_size,
        int(predictions.shape[0]/batch_size),
        predictions.shape[-1],
    ])
    for si in range(0,len(predictions)):
        sent = predictions[si]
        ptag = [] #预测的标签
        for word in sent:
            next_index = np.argmax(word)
            if next_index != 0:
                next_token = index2tag[next_index]
                ptag.append(next_token)
        senty = ybatch[si]
        ttag=[] #实际的标签
        for word in senty:
            next_token = index2tag[word]
            ttag.append(next_token)
        result = [] #result[0]=ptag result[1]=ttag
        result.append(ptag)
        result.append(ttag)
        result.append(xbatch[si])
        testresult.append(result)
    # pickle.dump(testresult,open(resultfile,'wb'))
    P, R, F, P1, R1, F1 = evaluavtion_triple(testresult,index2tag,index2word,epoch)
    print(P, R, F, P1, R1, F1)
    with open('./data/prf{}.log'.format(arg),'a') as f:
        f.write(flag+'\n')
        f.write(str(P)+' '+str(R)+' '+str(F)+'\n')
    return P, R, F, P1, R1, F1

def show_result(nn_model,testdata,index2tag,index2word,session):
    print('show the test')
    index2tag[0]=''
    xbatch = np.asarray(testdata[0],dtype="int32")
    ybatch = np.asarray(testdata[1],dtype="int32")
    batch_size=xbatch.shape[0]
    predictions=nn_model.predictions.eval(feed_dict={nn_model.inputs:xbatch},session=session)
    predictions=predictions.reshape([batch_size,predictions.shape[0]/batch_size,predictions.shape[-1]])
    for si in range(0,len(predictions)): #每句话
        sent = predictions[si]
        sentence=[]
        ptag = [] #预测的标签
        for i in range(len(sent)):
            tag=sent[i]
            next_index = np.argmax(tag)
            if next_index != 0:
                next_token = index2tag[next_index]
                sentence.append(index2word[xbatch[si][i]])
                ptag.append(next_token)
        senty = ybatch[si]
        ttag=[] #实际的标签
        for word in senty:
            next_token = index2tag[word]
            ttag.append(next_token)
        res={}
        res['sentence']=sentence
        res['predictions']=ptag
        res['ground_truth']=ttag
        with open('result.json','a') as f:
            f.write(json.dumps(res,ensure_ascii=False)+'\n')

def train_e2e_model(eelstmfile, modelfile,resultdir,arg,npochos,
                    lossnum=10,batchsize = 512,retrain=False):
    
    # load training data and test data
    traindata, testdata, showdata , source_W, source_vob, sourc_idex_word, target_vob, target_idex_word, max_s, k,\
        = pickle.load(open(eelstmfile, 'rb'))
    # import pdb; pdb.set_trace()

    # train model
    x_train = np.asarray(traindata[0], dtype="int32")
    # x_train = np.fliplr(x_train)
    y_train = np.asarray(traindata[1], dtype="int32")

    with tf.Session() as sess:
        # summary_writer = tf.summary.FileWriter('./graph', sess.graph)
        with tf.variable_scope("model", reuse=None):
            m=JointModel(seq_length=max_s,target2id=target_vob,id2target=target_idex_word,embedding_matrix=source_W,keep_prob=0.5,bilstm_dim=k)
        with tf.variable_scope("model", reuse=True):
            m_test=JointModel(is_training=False,seq_length=max_s,target2id=target_vob,id2target=target_idex_word,embedding_matrix=source_W,keep_prob=1.0,bilstm_dim=k)
        sess.run(tf.global_variables_initializer())
        for epoch in range(npochos):
            print('epoch:'+str(epoch))
            # for _ in tf.trainable_variables():
            #    print(_)
            for x_data, y_data in get_training_batch_xy_bias(x_train, y_train, max_s, max_s,
                                              batchsize, len(target_vob),
                                                target_idex_word,lossnum,shuffle=True):
                # pdb.set_trace()
                m.train_step.run(feed_dict={m.inputs:x_data,m.y_label:y_data})
                value=m.loss.eval(feed_dict={m.inputs:x_data,m.y_label:y_data})
                print('loss: {}'.format(value))
                if value>10:
                    with open('nan.log','a') as f:
                        f.write(str(value)+'\n')
            #rand=np.random.randint(low=0,high=20000)
            #tempdata=[traindata[0][rand:rand+2000],traindata[1][rand:rand+2000]]
            #test_model(m_test,tempdata,target_idex_word,sess,arg,epoch,'test on traindata')
            P, R, F, P1, R1, F1 = test_model(m_test,testdata,target_idex_word,sourc_idex_word,sess,arg,epoch)
            if F>0.5:
                show_result(m_test,showdata,target_idex_word,sourc_idex_word,sess)

def infer_e2e_model(eelstmfile, lstm_modelfile,resultfile):
    traindata, testdata, source_W, source_vob, sourc_idex_word, target_vob, \
    target_idex_word, max_s, k \
        = pickle.load(open(eelstmfile, 'rb'))

    nnmodel = creat_binary_tag_LSTM(sourcevocabsize=len(source_vob),targetvocabsize= len(target_vob),
                                    source_W=source_W,input_seq_lenth= max_s,output_seq_lenth= max_s,
                                    hidden_dim=k, emd_dim=k)

    nnmodel.load_weights(lstm_modelfile)
    P, R, F, pre1, rre1, fe1, pre2, rre2, fe2, tp1f, tp2f \
        = test_model(nnmodel, testdata, target_idex_word, resultfile)
    print(P, R, F)


if __name__=="__main__":

    alpha = 10
    maxlen = 150
    arg=''
    if len(sys.argv)>1:
        arg=sys.argv[1]
    trainfile = "./data/demo/{}train_tag.json".format(arg+'/')
    testfile = "./data/demo/{}test_tag.json".format(arg+'/')
    showfile = "./data/demo/{}show_tag.json".format(arg+'/')
    w2v_file = "./data/demo/{}w2v.pkl".format(arg+'/')
    e2edatafile = "./data/demo/{}model/e2edata.pkl".format(arg+'/')
    modelfile = "./data/demo/{}model/e2e_lstmb_model.pkl".format(arg+'/')
    resultdir = "./data/demo/{}result/".format(arg+'/')

    retrain = True
    valid = False
    if not os.path.exists(e2edatafile):
        print("Precess lstm data....")
        get_data_e2e(trainfile,testfile,showfile,w2v_file,e2edatafile,maxlen=maxlen)
    train_e2e_model(e2edatafile, modelfile,resultdir,arg,
                     npochos=1000,lossnum=alpha,retrain=False)



