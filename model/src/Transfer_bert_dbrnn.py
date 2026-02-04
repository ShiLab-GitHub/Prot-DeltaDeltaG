import tensorflow as tf

import sys
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional 

from keras.layers.merge import Concatenate, concatenate, subtract, multiply
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D

from keras.optimizers import Adam,  RMSprop

import keras.backend.tensorflow_backend as KTF

import numpy as np
from tqdm import tqdm

from keras.layers import Input, CuDNNGRU, GRU, LSTM, SimpleRNN

import scipy
from sklearn.model_selection import KFold, ShuffleSplit
import os
import pickle
from keras import backend as K


# 测试的时候不使用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

id2seq_file = '../../data/binding_affinity/SKP1102s.seq.txt'
ds_file = '../../data/binding_affinity/SKP1102s.ddg.txt'

label_index = int(4)
rst_file = 'results/new_test.txt'
use_emb = 3
sid1_index = 0
sid2_index = 1
sid3_index = 2
sid4_index = 3
model_dim = str(64)  # 这个model_dim其实是BILM的模型维度，所以没有什么用
hidden_dim = int(50)  # hidden_model才是MuPIPR模型真正的维度
# n_epochs = int(150)

n_epochs = int(200)
max_data = -1
seq_array_file = 'embedding/seq.txt'


if len(sys.argv) > 1:
    ds_file, id2seq_file, label_index, rst_file, hidden_dim, n_epochs, model_dim, max_data = sys.argv[1:]
    label_index = int(label_index)
    hidden_dim = int(hidden_dim)
    n_epochs = int(n_epochs)
    model_dim = model_dim
    max_data = int(max_data)

id2index = {}
seqs = []
index = 0
for line in open(id2seq_file):
    line = line.strip().split('\t')
    id2index[line[0]] = index
    seqs.append(line[1])
    index += 1
seq_array = []
id2_aid = {}
sid = 0

# max_data = 30
limit_data = max_data > 0
raw_data = []
raw_ids = []
skip_head = False
x = None
count = 0    


## Serving contextualized embeddings of amino acids ================================

def get_session(gpu_fraction=0.25):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def abs_diff(X):
    assert(len(X) == 2)
    s = X[0] - X[1]
    s = K.abs(s)
    return s

def abs_diff_output_shape(input_shapes):
    return input_shapes[0]

def build_model():

    # 这一块是模型的输入
    seq_input1 = Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = Input(shape=(seq_size, dim), name='seq2')
    seq_input3 = Input(shape=(seq_size, dim), name='seq3')
    seq_input4 = Input(shape=(seq_size, dim), name='seq4')
    l1=Conv1D(hidden_dim, 3)
    # r1=Bidirectional(GRU(hidden_dim, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True))
    r1=Bidirectional(GRU(hidden_dim, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True))

    l2=Conv1D(hidden_dim, 3)
    r2=Bidirectional(GRU(hidden_dim, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True))
    l3=Conv1D(hidden_dim, 3)
    r3=Bidirectional(GRU(hidden_dim, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True))
    r4=Bidirectional(SimpleRNN(hidden_dim, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True)) # DBRNN

    l_end=Conv1D(hidden_dim, 3)
    D1=Dense(100, activation='linear')

    D2=Dense(1, activation='linear')

    D3=Dense(100, activation='linear')

    D4=Dense(1, activation='linear')

    s1=MaxPooling1D(2)(l1(seq_input1))

    s1=concatenate([r1(s1), s1])
    s1=MaxPooling1D(2)(l2(s1))
    s1=concatenate([r2(s1), s1])
    s1=MaxPooling1D(3)(l3(s1))

    s1=concatenate([r3(s1), s1])

    s1=l_end(s1)

    s1=GlobalAveragePooling1D()(s1)
    
    s2=MaxPooling1D(2)(l1(seq_input2))

    s2=concatenate([r1(s2), s2])

    s2=MaxPooling1D(2)(l2(s2))
    s2=concatenate([r2(s2), s2])

    s2=MaxPooling1D(3)(l3(s2))

    s2=concatenate([r3(s2), s2])
    s2=l_end(s2)

    s2=GlobalAveragePooling1D()(s2)
    
    subtract_abs1 = keras.layers.Lambda(abs_diff, abs_diff_output_shape)
    
    merge_text1 = multiply([s1, s2])
    merge_text2 = subtract_abs1([s1,s2])
    
    merge_text_12 = concatenate([merge_text2, merge_text1])

    x12 = D1(merge_text_12)

    x12 = keras.layers.LeakyReLU(alpha=0.3)(x12)

    main_output12 = D2(x12)

    s3=MaxPooling1D(2)(l1(seq_input3))

    s3=concatenate([r1(s3), s3])

    s3=MaxPooling1D(2)(l2(s3))
    s3=concatenate([r2(s3), s3])

    s3=MaxPooling1D(3)(l3(s3))

    s3=concatenate([r3(s3), s3])

    s3=l_end(s3)
    s3=GlobalAveragePooling1D()(s3)

    s4=MaxPooling1D(2)(l1(seq_input4))

    s4=concatenate([r1(s4), s4])

    s4=MaxPooling1D(2)(l2(s4))
    s4=concatenate([r2(s4), s4])

    s4=MaxPooling1D(3)(l3(s4))

    s4=concatenate([r3(s4), s4])

    s4=l_end(s4)
    s4=GlobalAveragePooling1D()(s4)
    
    subtract_abs2 = keras.layers.Lambda(abs_diff, abs_diff_output_shape)
    
    merge_text1 = multiply([s3, s4])

    merge_text2 = subtract_abs2([s3, s4])
    
    merge_text_34 = concatenate([merge_text2, merge_text1])

    x34 = D1(merge_text_34)
    x34 = keras.layers.LeakyReLU(alpha=0.3)(x34)
    main_output34 = D2(x34)
    merge_text_1234 = concatenate([merge_text_12, merge_text_34])
    x1234 = D3(merge_text_1234)
    x1234 = keras.layers.LeakyReLU(alpha=0.3)(x1234)
    main_output = D4(x1234)
    merge_model = Model(inputs=[seq_input1, seq_input2, seq_input3, seq_input4], outputs=[main_output12, main_output34, main_output])

    return merge_model

def scale_back(v):

    return v * (all_max - all_min) + all_min

KTF.set_session(get_session())

'''
接下来的任务是保存seq_array到一个文件中，seq_array中是所有的蛋白质的氨基酸序列
然后使用prot_bert来把所有的氨基酸序列转换为embedding，并保存为另一个文件
'''
# seq.txt和seq2embedding.csv文件都已经制作完成了。就不用再重复的写入fp0了
# fp0 = open(seq_array_file, 'w')
for line in tqdm(open(ds_file)):
    if skip_head:
        skip_head = False
        continue
    line = line.rstrip('\n').rstrip('\r').replace('\t\t','\t').split('\t')
    raw_ids.append((line[sid1_index], line[sid2_index], line[sid3_index], line[sid4_index]))
    if id2_aid.get(line[sid1_index]) is None:
        id2_aid[line[sid1_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid1_index]]])
        # fp0.write(str(seqs[id2index[line[sid1_index]]]) + '\n')
    line[sid1_index] = id2_aid[line[sid1_index]]
    
    if id2_aid.get(line[sid2_index]) is None:
        id2_aid[line[sid2_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid2_index]]])
        # fp0.write(str(seqs[id2index[line[sid2_index]]]) + '\n')

    line[sid2_index] = id2_aid[line[sid2_index]]

    if id2_aid.get(line[sid3_index]) is None:
        id2_aid[line[sid3_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid3_index]]])
        # fp0.write(str(seqs[id2index[line[sid3_index]]]) + '\n')

    line[sid3_index] = id2_aid[line[sid3_index]]

    if id2_aid.get(line[sid4_index]) is None:
        id2_aid[line[sid4_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid4_index]]])
        # fp0.write(str(seqs[id2index[line[sid4_index]]]) + '\n')

    line[sid4_index] = id2_aid[line[sid4_index]]

    raw_data.append(line)
    
    if limit_data:
        count += 1
        if count >= max_data:
            break

len_m_seq = np.array([len(line) for line in seq_array])
avg_m_seq = int(np.average(len_m_seq)) + 1
max_m_seq = max(len_m_seq)

seq_index1 = np.array([line[sid1_index] for line in tqdm(raw_data)])
seq_index2 = np.array([line[sid2_index] for line in tqdm(raw_data)])
seq_index3 = np.array([line[sid3_index] for line in tqdm(raw_data)])
seq_index4 = np.array([line[sid4_index] for line in tqdm(raw_data)])


print("Num of samples", len(raw_data))
print("seq_array", len(seq_array))


batcher_size = 128
seq_tensor = []
max_seq_size = max_m_seq

embedding_tensor = []

with open('embedding/pre_embedding.pkl', 'rb') as f:
    data = pickle.load(f)

for aic_embedding in data:
    aic_embedding = np.array(aic_embedding)

    aic_embedding1 = np.reshape(aic_embedding, (1, aic_embedding.shape[0], aic_embedding.shape[1]))
    if aic_embedding1.shape[1] < max_seq_size:

        npad = ((0, 0), (0, max_seq_size - aic_embedding1.shape[1]), (0, 0))
        aic_embedding1 = np.pad(aic_embedding1, pad_width=npad, mode='constant', constant_values=0)
    if seq_tensor == []:
        seq_tensor = aic_embedding1
    else:
        seq_tensor = np.concatenate((seq_tensor, aic_embedding1), axis=0)

print("seq_tensor shape",seq_tensor.shape)
seq_size, dim = seq_tensor.shape[1], seq_tensor.shape[2]
print("seq_size, dim", seq_size, dim)

num_scores = 3
score_labels = np.zeros((len(raw_data), num_scores))
# use_log = True
for i in range(len(raw_data)):
    score_labels[i] = raw_data[i][label_index:]

all_min, all_max = 99999999, -99999999
for i in range(len(raw_data)):
    score_labels[i] = raw_data[i][label_index:]

for j in range(2):
    min_j = min(score_labels[:,j])
    max_j = max(score_labels[:,j])
    if min_j < all_min:
        all_min = min_j
    if max_j > all_max:
        all_max = max_j

# ddG is normalized differently
for j in range(2):        
    score_labels[:,j] = (score_labels[:,j] - all_min )/(all_max - all_min)

score_labels[:,2] = (score_labels[:,2])/(all_max - all_min)    

print("All max, min",all_min, all_max)    

batch_size1 = 32
adam = Adam(lr=0.005, amsgrad=True, epsilon=1e-5)

kf = KFold(n_splits=10, shuffle = True, random_state=13)


tries = 11
cur = 0
recalls = []
accuracy = []
total = []
total_truth = []
train_test = []
for train, test in kf.split(score_labels):
    train_test.append((train, test))

    cur += 1
    if cur >= tries:
        break

print ("train_test", len(train_test))
num_total = 0.
total_mse = 0.
total_mae = 0.
total_cov = 0.


fp2 = open('records/pred_record_3G_test.'+rst_file[rst_file.rfind('/')+1:], 'w')
n_fold = 0

for train, test in train_test:
    print("In fold: ", n_fold)
    n_fold+=1
    merge_model = build_model()
    adam = Adam(lr=0.001, amsgrad=True, epsilon=1e-6)
    rms = RMSprop(lr=0.001)

    merge_model.compile(optimizer=adam, loss='mse', metrics=['mse', pearson_r])

    merge_model.fit([seq_tensor[seq_index1[train]], seq_tensor[seq_index2[train]], seq_tensor[seq_index3[train]], seq_tensor[seq_index4[train]]], [score_labels[train,0], score_labels[train,1], score_labels[train,2]], batch_size=batch_size1, epochs=n_epochs, verbose = 0)

    pred = merge_model.predict([seq_tensor[seq_index1[test]], seq_tensor[seq_index2[test]], seq_tensor[seq_index3[test]], seq_tensor[seq_index4[test]]])

    this_mae, this_mse, this_cov = 0., 0., 0.
    this_num_total = 0

    dG_pred0 = pred[0]
    dG_pred1 = pred[1]
    ddG_pred = pred[2]

    print("evaluating ......")
    for i in range(len(score_labels[test,num_scores-1])):
        this_num_total += 1

        ddG_label_i = (all_max - all_min)*score_labels[test,2][i]
        ddG_pred_i = (all_max - all_min)*ddG_pred[i]
        
        diff = abs(ddG_label_i - ddG_pred_i)
        this_mae += diff
        this_mse += diff**2

    num_total += this_num_total
    total_mae += this_mae
    total_mse += this_mse
    mse = total_mse / num_total
    mae = total_mae / num_total
    this_cov = scipy.stats.pearsonr(np.ndarray.flatten(ddG_pred), score_labels[test,num_scores-1])[0]

    for i in range(len(test)):
        fp2.write(str(raw_ids[test[i]][sid1_index]) + '\t' + str(raw_ids[test[i]][sid2_index])  + '\t' + str(raw_ids[test[i]][sid3_index])  + '\t' + str(raw_ids[test[i]][sid4_index]) 
            + '\t' + str(scale_back(np.ndarray.flatten(dG_pred0)[i])) + '\t' + str(scale_back(score_labels[test[i], 0]))
            + '\t' + str(scale_back(np.ndarray.flatten(dG_pred1)[i])) + '\t' + str(scale_back(score_labels[test[i], 1]))
            + '\t' + str(np.ndarray.flatten((all_max - all_min)*ddG_pred)[i]) + '\t' + str((all_max - all_min)*score_labels[test[i], 2]) + '\n')

    total_cov += this_cov
    print (mse[0], this_cov, sep = '\t')

mse = total_mse / num_total
mae = total_mae / num_total
total_cov /= len(train_test)
print("Epoch", n_epochs)
print ("Average", mse[0], total_cov)

with open(rst_file, 'w') as fp:
    fp.write('mse=' + str(mse) + '\ncorr=' + str(total_cov))
