'''
    author:fengjiaxin
    desc:define seq2seq model
'''
import random

import numpy as np
import tensorflow as tf
import data_utils
from seq2seq_attention import embedding_attention_seq2seq,model_with_buckets


setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

class Seq2SeqModel(object):
    """
        Sequence-to-sequence model with attention and for multiple buckets.
          This class implements a multi-layer recurrent neural network as encoder,
          and an attention-based decoder. This is the same as the model described in
          this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
          or into the seq2seq library for complete model implementation.
          This class also allows to use GRU cells in addition to LSTM cells, and
          sampled softmax to handle large output vocabulary size. A single-layer
          version of this model, but with bi-directional encoder, was presented in
            http://arxiv.org/abs/1409.0473
          and sampled softmax is described in Section 3 of the following paper.
            http://arxiv.org/abs/1412.2007
      """

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=False,
                 num_samples=512,
                 forward_only=False,
                 dtype=tf.float32):
        """Create the model.
        Args:
          source_vocab_size: size of the source vocabulary.
          target_vocab_size: size of the target vocabulary.
          buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          forward_only: if set, we do not construct the backward pass in the model.
          dtype: the data type to use to store internal variables.
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate),trainable=False,dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0,trainable=False)

        # if use sampled softmax ,need an output projection
        output_projection = None
        softmax_loss_function = None
        # sampled softmax only makes sense if we sample less than vocabulary size
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable('proj_w',[self.target_vocab_size,size],dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable('proj_b',[self.target_vocab_size],dtype=dtype)
            output_projection=(w,b)

            # sampled loss 的提出原因
            # 解码器rnn序列在每一时刻的输出层为softmax分类器，在对上面的目标函数求梯度时，表达式会出现对整个target
            # vocabulary的求和项，计算量非常大，于是工程师们想到了用target vocabulary中的一个子集，来近似对整个
            # 词库的求和，子集中word的选取采用的是均匀采样的策略，从而降低了每次梯度更新步骤的计算复杂度
            def sampled_loss(labels,logits):
                labels = tf.reshape(labels,[-1,1])
                local_w_t = tf.cast(w_t,tf.float32)
                local_b = tf.cast(b,tf.float32)
                local_inputs = tf.cast(logits,tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=local_w_t,
                        biases=local_b,
                        labels=labels,
                        inputs=local_inputs,
                        num_sampled=num_samples,
                        num_classes=self.target_vocab_size),dtype)
            softmax_loss_function = sampled_loss

        # 接下来建立rnn基本的循环单元
        def single_cell():
            if use_lstm:
                return tf.contrib.rnn.BasicLSTMCell(size)
            else:
                return tf.contrib.rnn.GRUCell(size)
        cell = single_cell()
        # 说明是多层lstm叠加
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])

        # seq2seq function :use embedding for the input and attention
        def seq2seq_f(encoder_inputs,decoder_inputs,do_decode):
            return embedding_attention_seq2seq(encoder_inputs,
                                               decoder_inputs,
                                               cell,
                                               num_encoder_symbols=source_vocab_size,
                                               num_decoder_symbols=target_vocab_size,
                                               embedding_size=size,
                                               output_projection=output_projection,
                                               feed_previous=do_decode,
                                               dtype=dtype)


        # feed for inputs
        # shape:[timesteps,batch_size]
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        # 注意bucket元组是(source_len,target_len),但是在训练和预测的时候，decoder_inputs第一步是_GO，
        # 因此len(decoder_inputs) = len(target) + 1,因此预测的结果是target = decoder[1:]
        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name='encoder{0}'.format(i)))

        # 正常的decoder_inputs 是 _go + target_sentence
        # self.deocder_inputs 是全的，最后希望是_go + target_sentence + _eos
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name='decoder{0}'.format(i)))
            self.target_weights.append(tf.placeholder(dtype,shape=[None],name='weight{0}'.format(i)))

        # targets are decoder inputs shifted by one
        targets = [self.decoder_inputs[i+1] for i in range(len(self.decoder_inputs) - 1)]

        # 如果是inference 阶段
        if forward_only:
            # 注意，列表的长度是len(buckets)
            self.outputs,self.losses = model_with_buckets(
                self.encoder_inputs,self.decoder_inputs,targets,self.target_weights,buckets,
                lambda x,y:seq2seq_f(x,y,True),softmax_loss_function=softmax_loss_function)

            if output_projection is not None:
                for b in len(buckets):
                    self.outputs[b] = [
                        tf.matmul(output,output_projection[0]) + output_projection[2]
                        for output in self.outputs[b]
                    ]

        # train phase
        else:
            self.outputs, self.losses = model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets, self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False), softmax_loss_function=softmax_loss_function)


        # 更新参数
        params = tf.trainable_variables()
        if not forward_only:
            # 存储一个张量的所有范数
            self.gradient_norms = []
            # 存储对变量更新参数的操作
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in range(len(buckets)):
                # 获取梯度
                gradients = tf.gradients(self.losses[b],params)
                # 截取梯度，返回截取后的更新梯度 和 所有张量的全局范数
                clipped_gradients,norm = tf.clip_by_global_norm(gradients,max_gradient_norm)

                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients,params),global_step=self.global_step))


        self.saver = tf.train.Saver(tf.global_variables())



    def step(self,session,encoder_inputs,decoder_inputs,target_weights,bucket_id,forward_only):
        '''
        run a step of the model the given inputs
        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.
        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        '''
        encoder_size,decoder_size = self.buckets[bucket_id]

        if len(encoder_inputs) != encoder_size:
            raise ValueError('Encoder length must be equal to the one in bucket,%d != %d'%(len(encoder_size),encoder_size))

        if len(decoder_inputs) != decoder_size:
            raise ValueError('Decoder length must be equal to the one in bucket,%d != %d'%(len(decoder_inputs),decoder_size))

        if len(target_weights) != decoder_size:
            raise ValueError('target_weights length must be equal to the one in bucket,%d != %d'%(len(target_weights),decoder_size))

        # input feed:encoder_inputs,decoder_inputs,target_weights
        input_feed = dict()
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]

        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # since targets are decoder inputs shifted by one,need one more
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size],dtype=np.int32)

        # 计算损失，根据是否反向传播
        if not forward_only:
            output_feed = [self.updates[bucket_id],
                           self.gradient_norms[bucket_id],
                           self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            for l in range(decoder_size):
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed,input_feed)

        if not forward_only:
            return outputs[1],outputs[2],None # gradient norm,loss,no outputs
        else:
            return None,outputs[0],outputs[1:]

    def get_batch(self,data,bucket_id):
        '''
            Get a random batch of data from the specified bucket, prepare for step.
        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.
        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.
        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        '''
        encoder_size,decoder_size = self.buckets[bucket_id]
        encoder_inputs,decoder_inputs = [],[]

        # get a random batch of encoder and decoder inputs from data
        # pad them if needed ,reverse encoder inputs and add GO to the decoder
        for _ in range(self.batch_size):
            encoder_input,decoder_input = random.choice(data[bucket_id])

            # encoder inputs are padded and then reversed
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # decoder inputs get an extra '_GO' symbol ,and are padded then
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input + [data_utils.PAD_ID] * decoder_pad_size)

        # now we create batch_major vectors
        batch_encoder_inputs,batch_decoder_inputs,batch_weights = [],[],[]

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                # train 阶段，encoder_inputs and decoder_inputs 都是已知的，由于target是decoder_inputs shifted right one
                # 因此当前时间段的target就是下一时刻的decoder_inputs
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights




























