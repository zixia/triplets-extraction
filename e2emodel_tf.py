# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'
import cPickle
import os.path
import pdb
import numpy as np
import tensorflow as tf
from PrecessEEdata import get_data_e2e
from Evaluate import evaluavtion_triple
from LSTM_decoder import decoder_layer

def weight_variable(shape):
    init=tf.truncated_normal(shape,stddev=0.1,dtype=tf.float32)
    return tf.Variable(init)

def bias_variable(shape):
    init=tf.constant(0.1,shape=shape,dtype=tf.float32)
    return tf.Variable(init)

def embedding_layer(embedding_weights,input_x,keep_prob):
    return tf.nn.dropout(tf.nn.embedding_lookup(tf.Variable(embedding_weights),input_x),keep_prob=keep_prob)

def Bi_LSTM(hidden_dim,bilstm_input,batchsize):
    n_steps=bilstm_input.shape[1]
    bilstm_input = tf.unstack(bilstm_input, n_steps, 1)
    with tf.variable_scope('forward'):
        forward_lstm=tf.contrib.rnn.LSTMCell(hidden_dim,initializer=tf.random_uniform_initializer(-0.01, 0.01),forget_bias=0.0, state_is_tuple=True)
    with tf.variable_scope('backward'):
        backward_lstm=tf.contrib.rnn.LSTMCell(hidden_dim,initializer=tf.random_uniform_initializer(-0.01, 0.01), forget_bias=0.0, state_is_tuple=True)
    Bi_LSTM=tf.nn.static_bidirectional_rnn(forward_lstm,backward_lstm,bilstm_input,dtype=tf.float32)
    return tf.stack(Bi_LSTM[0],axis=1)

def LSTMd_layer(hidden_dim,lstmd_input,batchsize):
    with tf.variable_scope('decode'):
        LSTMd=decoder_layer(hidden_dim*2,initializer=tf.random_uniform_initializer(-0.01, 0.01), forget_bias=0.0)
    return tf.nn.dynamic_rnn(LSTMd,lstmd_input,dtype=tf.float32)

def softmax_layer(layer_weights,softmax_input,layer_bias):
    # time_input=tf.unstack(time_input,time_input.shape[0],0)
    softmax_input=tf.reshape(softmax_input,[-1,softmax_input.shape[-1]])
    #res=[]
    #for _ in time_input:
    #    temp=tf.matmul(_,layer_weights)+layer_bias
    #    res.append(tf.nn.softmax(logits=temp))
    #return tf.stack(res)
    return tf.matmul(softmax_input,layer_weights)

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

def trans_mat(lossnum,targetvocabsize):
    res=np.eye(targetvocabsize+1,dtype=np.float32)
    for i in range(res.shape[0]):
        if i>1:
            res[i]*=lossnum
    return res


def train_e2e_model(eelstmfile, modelfile,resultdir,npochos,
                    lossnum=1,batchsize = 500,retrain=False):
    
    # load training data and test data
    traindata, testdata, source_W, source_vob, sourc_idex_word, target_vob, target_idex_word, max_s, k \
        = cPickle.load(open(eelstmfile, 'rb'))

    # train model
    x_train = np.asarray(traindata[0], dtype="int32")
    y_train = np.asarray(traindata[1], dtype="int32")
    #nn_model = creat_binary_tag_LSTM(sourcevocabsize=len(source_vob), targetvocabsize=len(target_vob),
#                                    source_W=source_W, input_seq_lenth=max_s, output_seq_lenth=max_s,
#                                  hidden_dim=k, emd_dim=k)
#    if retrain:
#        nn_model.load_weights(modelfile)
    #nn_model = CreatBinaryTagLSTM_Att(len(source_vob), len(target_vob), source_W, max_s, max_t, k, k)
    epoch = 0
    x=tf.placeholder(tf.int32,[None,max_s])
    y_label=tf.placeholder(tf.float32,[None,max_s,len(target_vob)+1])
    keep_prob=tf.placeholder(tf.float32)
    source_W=tf.to_float(source_W, name='ToFloat')
    word_embedding=embedding_layer(embedding_weights=source_W,input_x=x,keep_prob=keep_prob)
    bilstm_h=Bi_LSTM(hidden_dim=k,bilstm_input=word_embedding,batchsize=batchsize)
    lstmd_h=LSTMd_layer(hidden_dim=k,lstmd_input=bilstm_h,batchsize=batchsize)[0]
    softmax_W,softmax_b=weight_variable([k*2,len(target_vob)+1]),bias_variable([len(target_vob)+1])
    softmax_output=softmax_layer(softmax_W,lstmd_h,softmax_b)
    y_pre=tf.nn.softmax(logits=softmax_output)
    loss = tf.reduce_mean(-tf.reduce_sum(tf.reshape(y_label,[-1,y_label.shape[-1]])*tf.log(y_pre),reduction_indices=[1]))
    train_step=tf.train.RMSPropOptimizer(0.001).minimize(loss)
    with tf.Session() as sess:
        # summary_writer = tf.summary.FileWriter('./graph', sess.graph)
        sess.run(tf.global_variables_initializer())
        while (epoch < npochos):
            print 'epoch:'+str(epoch)
            epoch+=1
            for x_data, y_data in get_training_batch_xy_bias(x_train, y_train, max_s, max_s,
                                              batchsize, len(target_vob),
                                                target_idex_word,lossnum,shuffle=True):
                # pdb.set_trace()
                train_step.run(feed_dict={x:x_data,y_label:y_data,keep_prob:0.5})
                # print 'loss: {}'.format(loss.eval(feed_dict={x:x_data,y_label:y_data,keep_prob:1.0}))
            index2word=target_idex_word
            index2word[0]=''
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
                predictions=y_pre.eval(feed_dict={x:xbatch,keep_prob:1.0})
                predictions=predictions.reshape([batch_size,predictions.shape[0]/batch_size,predictions.shape[-1]])
                if epoch>5:
                    pdb.set_trace()
                for si in range(0,len(predictions)):
                    if testlinecount < testlen:
                        sent = predictions[si]
                        ptag = []
                        for word in sent:
                            next_index = np.argmax(word)
                            if next_index != 0:
                                next_token = index2word[next_index]
                                ptag.append(next_token)
                        senty = ybatch[si]
                        ttag=[]
                        for word in senty:
                            next_token = index2word[word]
                            ttag.append(next_token)
                        result = []
                        result.append(ptag)
                        result.append(ttag)
                        testlinecount += 1
                        testresult.append(result)
            # cPickle.dump(testresult,open(resultfile,'wb'))
            P, R, F = evaluavtion_triple(testresult)
            print P, R, F
            with open('./data/prf.log','a') as f:
                f.write(str(P)+' '+str(R)+' '+str(F)+'\n')
            #nn_model.fit(x, y, batch_size=batch_size,
#                         nb_epoch=1, show_accuracy=False, verbose=0)
            #if epoch > saveepoch:
                # saveepoch += save_inter
                # resultfile = resultdir+"result-"+str(saveepoch)
                # P, R, F, pre1, rre1, fe1, pre2, rre2, fe2, tp1f, tp2f\
                # P, R, F= test_model(nn_model, testdata, target_idex_word,resultfile)

#                if F > maxF:
#                    maxF=F
#                    save_model(nn_model, modelfile)

                #print P, R, F
    # return nn_model

def infer_e2e_model(eelstmfile, lstm_modelfile,resultfile):
    traindata, testdata, source_W, source_vob, sourc_idex_word, target_vob, \
    target_idex_word, max_s, k \
        = cPickle.load(open(eelstmfile, 'rb'))

    nnmodel = creat_binary_tag_LSTM(sourcevocabsize=len(source_vob),targetvocabsize= len(target_vob),
                                    source_W=source_W,input_seq_lenth= max_s,output_seq_lenth= max_s,
                                    hidden_dim=k, emd_dim=k)

    nnmodel.load_weights(lstm_modelfile)
    P, R, F, pre1, rre1, fe1, pre2, rre2, fe2, tp1f, tp2f \
        = test_model(nnmodel, testdata, target_idex_word, resultfile)
    print P, R, F


if __name__=="__main__":

    alpha = 10
    maxlen = 50
    trainfile = "./data/demo/train_tag.json"
    testfile = "./data/demo/test_tag.json"
    w2v_file = "./data/demo/w2v.pkl"
    e2edatafile = "./data/demo/model/e2edata.pkl"
    modelfile = "./data/demo/model/e2e_lstmb_model.pkl"
    resultdir = "./data/demo/result/"

    retrain = True
    valid = False
    if not os.path.exists(e2edatafile):
        print "Precess lstm data...."
        get_data_e2e(trainfile,testfile,w2v_file,e2edatafile,maxlen=maxlen)
    train_e2e_model(e2edatafile, modelfile,resultdir,
                     npochos=100,lossnum=alpha,retrain=False)



