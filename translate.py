'''
    author:fengjiaxin
    desc:主要负责生成训练数据集，然后进行训练和decode
'''
import warnings
warnings.filterwarnings('ignore')

from hparams import Hparams
import sys
import os
import numpy as np
import tensorflow as tf
import data_utils
from seq2seq_model import Seq2SeqModel
import time
import math





# define buckets：怎么处理？熟读代码后了解到，如果encode长度为30,decode长度为40，那么首先从训练数据集中找出source长度小于30 and target长度小于40的，
# 然后长度不够的用_PAD填充
_buckets = [(5,10),(10,15),(20,25),(40,50)]


def read_data(source_ids_path,targt_ids_path,max_size=None):
    '''
    读取文件，将数据放到buckets
    :return:
        data_list: a list of lengh len(_buckets) : len(source) < _buckets[n][0] and len(target) < _buckets[n][1]
    '''
    data_set = [[] for _ in range(len(_buckets))]
    with open(source_ids_path,'r') as source_file,open(targt_ids_path,'r') as target_file:
        source,target = source_file.readline(),target_file.readline()
        counter = 0
        while source and target and (not max_size or counter < max_size):
            counter += 1
            if counter % 10000 == 0:
                print('read ids path %d lines'%counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            # 此时将_EOD 加入到target 结尾，因为需要确定预测的结束条件
            target_ids.append(data_utils.EOS_ID)
            # 因为buckets 的 encode_size,decoder_size是递增的，所以需要找到当前的翻译句子填充后可以到达哪个bucket的size
            for bucket_id,(source_size,target_size) in enumerate(_buckets):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append((source_ids,target_ids))
                    break
            source,target = source_file.readline(),target_file.readline()
    return data_set


def create_model(session,forward_only,hp):
    model = Seq2SeqModel(
            hp.en_vocab_size,
            hp.fr_vocab_size,
            _buckets,
            hp.size,
            hp.num_layers,
            hp.max_gradient_norm,
            hp.batch_size,
            hp.learning_rate,
            hp.learning_rate_decay_factor,
            use_lstm=False,
            forward_only=forward_only)
    ckpt = tf.train.get_checkpoint_state(hp.train_checkpoint_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print('reading model parameters form %s'%ckpt.model_checkpoint_path)
        model.saver.restore(session,ckpt.model_checkpoint_path)
    else:
        print('created model with fresh parameters')
        session.run(tf.initialize_all_variables())
    return model




# preprocess file
def process_file(hp):
    # 首先创建词表
    if not os.path.exists(hp.vocab_en):
        data_utils.create_vocabulary(hp.vocab_en,hp.train_en,hp.en_vocab_size)
    if not os.path.exists(hp.vocab_fr):
        data_utils.create_vocabulary(hp.vocab_fr,hp.train_fr,hp.fr_vocab_size)

    # 如果生成的ids_file文件不存在
    if not os.path.exists(hp.train_en_ids):
        data_utils.data_to_token_ids(hp.train_en, hp.train_en_ids, hp.vocab_en)
    if not os.path.exists(hp.train_fr_ids):
        data_utils.data_to_token_ids(hp.train_fr, hp.train_fr_ids, hp.vocab_fr)
    if not os.path.exists(hp.test_en_ids):
        data_utils.data_to_token_ids(hp.test_en, hp.test_en_ids, hp.vocab_en)
    if not os.path.exists(hp.test_fr_ids):
        data_utils.data_to_token_ids(hp.test_fr, hp.test_fr_ids, hp.vocab_fr)



def train(hp):
  
    with tf.Session() as sess:
        # create model
        print('create %d layers of %d units.'%(hp.num_layers,hp.size))
        model = create_model(sess,False,hp)
        print('model create success')

        # 确定每个buckets的数据train的样本数
        train_data = read_data(hp.train_en_ids,hp.train_fr_ids,hp.max_train_data_size)
        test_data = read_data(hp.test_en_ids,hp.test_fr_ids)
        print('generate data success')
        train_buckets_sizes = [len(train_data[b]) for b in range(len(_buckets))]
        # 已知每个桶的样本数量，可以获取buckets的所有样本数
        train_total_size = float(sum(train_buckets_sizes))
        # 接下来确定scale,假设有3个bucket，长度为1，4，5，希望获取一个[0.1,0.5,1]的列表，大小是从0到1递增的列表，最后一个结果一定是1
        # 这个用来做什么，在训练的时候希望随机训练一个bucket的数据，那么怎么确定是哪个bucket_id，如果随机数<0.1,那么bucket_id = 0
        # if random = 0.4,bucket_id = 1;if random = 0.8,bucket_id = 2
        train_buckets_scale = [sum(train_buckets_sizes[:i+1])/train_total_size
                               for i in range(len(train_buckets_sizes))]

        step_time,loss = 0.0,0.0
        current_step = 0
        previous_loss = []
        print('begin train')
        for step_index in range(hp.steps):
            random_number = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number])
            # get a batch and make a step
            start_time = time.time()
            encoder_inputs,decoder_inputs,target_weights = model.get_batch(
                train_data,bucket_id
            )
            _,step_loss,_ = model.step(sess,encoder_inputs,decoder_inputs,target_weights,bucket_id,False)

            step_time += (time.time() - start_time)/hp.steps_per_checkpoint
            loss += step_loss/hp.steps_per_checkpoint
            current_step += 1

            # save model and print log
            if current_step % hp.steps_per_checkpoint == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print('global step %d learning rate %.4f step_time %.2f perplexity %.2f'%
                      (model.global_step.eval(),model.learning_rate.eval(),step_time,perplexity))

                # decreaste learnning rate if no imporvement over last 3 times
                if len(previous_loss) > 2 and loss > max(previous_loss[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_loss.append(loss)

                # save check_point_file
                checkpoint_path = os.path.join(hp.train_checkpoint_dir,'translate.ckpt')
                model.saver.save(sess,checkpoint_path,global_step=model.global_step)

                step_time,loss = 0.0,0.0

                # run evals on test data and print perplexity
                for bucket_id in range(len(_buckets)):
                    if len(test_data[bucket_id]) == 0:
                        print(' eval:empty bucket_id %d'%(bucket_id))
                        continue

                    encoder_inputs,decoder_inputs,target_weights = model.get_batch(
                        test_data,bucket_id)
                    _,eval_loss,_ = model.step(
                        sess,encoder_inputs,decoder_inputs,target_weights,bucket_id,True)
                    eval_perplexity = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print(' eval:bucket %d perplexity %.2f'%(bucket_id,eval_loss))
                sys.stdout.flush()

def decode(hp):
    with tf.Session() as sess:
        model = create_model(sess,True)
        model.batch_size = 1 # decode one sentence at a time

        # load vocab file
        en_vocab, _ = data_utils.initialize_vocabulary(hp.vocab_en)
        _,rev_fr_vocab = data_utils.initialize_vocabulary(hp.vocab_fr)

        sys.stdout.write('> ')
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            token_ids = data_utils.sentence_to_ids(sentence,en_vocab)
            # this sentence belong to whick bucket_id
            bucket_id = min([b for b in range(len(_buckets)) if _buckets[b][0] > len(token_ids)])

            # feed the sentence in the model
            encoder_inputs,decoder_inputs,target_weights = model.get_batch(
                {bucket_id:[(token_ids,[])]},bucket_id)

            # get output logits for the sentence
            _,_,output_logits = model.step(sess,encoder_inputs,decoder_inputs,target_weights,bucket_id,True)

            # greedy decoder
            outputs = [int(np.argmax(logit,axis=1)) for logit in output_logits]

            # if there is a EOS symbol in the end,del it
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]

            # print translate french
            print(" ".join([rev_fr_vocab[output] for output in outputs]))
            print('> ',end='')
            sys.stdout.flush()
            sentence = sys.stdin.readline()

if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    print('process file')
    process_file(hp)
    print('train')
    train(hp)







