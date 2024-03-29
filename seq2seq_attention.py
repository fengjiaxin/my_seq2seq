'''
    author:fengjiaxin
    desc：本文件用于编写带有attention机制的seq2seq实现
'''
import copy
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

Linear = core_rnn_cell._Linear

def sequence_loss_by_example(logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,
                             name=None):
    '''
        Weighted cross-entropy loss for a sequence of logits (per example).
        该函数用于计算所有examples的加权交叉熵损失
    Args:
      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
      targets: List of 1D batch-sized int32 Tensors of the same length as logits.
      weights: List of 1D batch-sized float-Tensors of the same length as logits.
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      softmax_loss_function: Function (labels, logits) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
        **Note that to avoid confusion, it is required for the function to accept
        named arguments.**
      name: Optional name for this operation, default: "sequence_loss_by_example".

    Returns:
      1D batch-sized float Tensor: The log-perplexity for each sequence.

    Raises:
      ValueError: If len(logits) is different from len(targets) or len(weights).
    '''
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError('length of logits,weights and targets must be the same %d %d %d.'%(len(logits),len(weights),len(targets)))

    with ops.name_scope(name,'sequence_loss_by_example',logits + targets + weights):
        # log_perp_list 存储的是加权的损失，长度是预测的时间步长度
        log_perp_list = []
        for logit,target,weight in zip(logits,targets,weights):
            if softmax_loss_function is None:
                target = array_ops.reshape(target,[-1])
                crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                    labels=target,logits=logit)
            else:
                crossent = softmax_loss_function(labels=target,logits=logit)
            log_perp_list.append(crossent * weight)
        # 计算decoder 时间步的所有corssent sum
        log_perps = math_ops.add_n(log_perp_list)
        # 是否计算平均每个时间步的误差
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            total_size += 1e-12
            log_perps /= total_size
    return log_perps

def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None,
                  name=None):
    '''
        Weighted cross-entropy loss for a sequence of logits, batch-collapsed.
        是否计算每个批次batch的平均误差

        Args:
            logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
            targets: List of 1D batch-sized int32 Tensors of the same length as logits.
            weights: List of 1D batch-sized float-Tensors of the same length as logits.
            average_across_timesteps: If set, divide the returned cost by the total
              label weight.
            average_across_batch: If set, divide the returned cost by the batch size.
            softmax_loss_function: Function (labels, logits) -> loss-batch
              to be used instead of the standard softmax (the default if this is None).
              **Note that to avoid confusion, it is required for the function to accept
              named arguments.**
            name: Optional name for this operation, defaults to "sequence_loss".

        Returns:
            A scalar float Tensor: The average log-perplexity per symbol (weighted).

        Raises:
            ValueError: If len(logits) is different from len(targets) or len(weights).
    '''
    with ops.name_scope(name,'sequence_loss',logits + targets + weights):
        cost = math_ops.reduce_sum(
            sequence_loss_by_example(logits,
                                     targets,
                                     weights,
                                     average_across_timesteps=average_across_timesteps,
                                     softmax_loss_function=softmax_loss_function))
        # 计算每个batch 的平均loss
        if average_across_batch:
            batch_size = array_ops.shape(targets[0])[0]
            return cost/math_ops.cast(batch_size,cost.dtype)
        else:
            return cost


# bucketing 策略，就是定义不同长度的graph,但是共享参数。设置若干个buckets，每个bucket指定一个输入和输出的长度，
# 例如buckets=[(5,10),(10,15),(20,25),(40,50)],经过bucketing策略处理后，会把所有的训练样例分成4份，其中每一份的输入序列和输出序列长度分别相同
# 这个之后理解
def model_with_buckets(encoder_inputs,
                       decoder_inputs,
                       targets,
                       weights,
                       buckets,
                       seq2seq,
                       softmax_loss_function=None,
                       per_example_loss=False,
                       name=None):
    '''
        The seq2seq argument is a function that defines a sequence-to-sequence model,
        我的理解:首先decoder_inputs的长度 是预测的时间步长度，weights 是预测的时间步长度
        e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(
            x, y, rnn_cell.GRUCell(24))

        Args:
            encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
            decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
            targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
            weights: List of 1D batch-sized float-Tensors to weight the targets.
            buckets: A list of pairs of (input size, output size) for each bucket.
            seq2seq: A sequence-to-sequence model function; it takes 2 input that
              agree with encoder_inputs and decoder_inputs, and returns a pair
              consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
            softmax_loss_function: Function (labels, logits) -> loss-batch
              to be used instead of the standard softmax (the default if this is None).
              **Note that to avoid confusion, it is required for the function to accept
              named arguments.**
            per_example_loss: Boolean. If set, the returned loss will be a batch-sized
              tensor of losses for each sequence in the batch. If unset, it will be
              a scalar with the averaged loss from all examples.
            name: Optional name for this operation, defaults to "model_with_buckets".

        Returns:
            A tuple of the form (outputs, losses), where:
              outputs: The outputs for each bucket. Its j'th element consists of a list
                of 2D Tensors. The shape of output tensors can be either
                [batch_size x output_size] or [batch_size x num_decoder_symbols]
                depending on the seq2seq model used.
              losses: List of scalar Tensors, representing losses for each bucket, or,
                if per_example_loss is set, a list of 1D batch-sized float Tensors.

        Raises:
            ValueError: If length of encoder_inputs, targets, or weights is smaller
              than the largest (last) bucket.
    '''
    # 首先肯定需要将encoder_inputs，decoder_inputs的时间步长度固定，然后分桶策略就是截取长度，如果桶的长度大于输入的长度，
    # 那么就报错，如果weights的长度小于decoder_inputs的长度，就无法计算加权损失，也抱异常
    if len(encoder_inputs) < buckets[-1][0]:
        raise ValueError('length of encoder inputs(%d) be at least that of last bucket (%d)'%(len(encoder_inputs),buckets[-1][0]))
    if len(decoder_inputs) < buckets[-1][1]:
        raise ValueError('length of decoder inputs(%d) be at least that of last bucket (%d)'%(len(decoder_inputs),buckets[-1][1]))
    if len(weights) < buckets[-1][1]:
        raise ValueError('length of weights(%d) must be at least that of last bucket (%d)' % (len(weights),buckets[-1][1]))

    all_inputs = encoder_inputs + decoder_inputs + targets + weights
    # 列表的长度的桶的个数
    losses = []
    outputs = []
    with ops.name_scope(name,'model_with_buckets',all_inputs):
        for i,bucket in enumerate(buckets):
            with variable_scope.variable_scope(
                variable_scope.get_variable_scope(),reuse=True if i > 0 else None):
                bucket_outputs,_ = seq2seq(encoder_inputs[:bucket[0]],decoder_inputs[:bucket[1]])
                outputs.append(bucket_outputs)
                if per_example_loss:
                    losses.append(sequence_loss_by_example(
                        outputs[-1],
                        targets[:bucket[1]],
                        weights[:bucket[1]],
                        softmax_loss_function=softmax_loss_function))
                else:
                    losses.append(
                        sequence_loss(
                            outputs[-1],
                            targets[:bucket[1]],
                            weights[:bucket[1]],
                            softmax_loss_function=softmax_loss_function))
    return outputs,losses




def embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
    '''
        这个模型分为两个步骤，首先是建立encoder过程，然后建立decoder过程，同时采用attention机制，最终预测输出的过程
    Args:
        encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
        num_encoder_symbols: Integer; number of symbols on the encoder side.
        num_decoder_symbols: Integer; number of symbols on the decoder side.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        num_heads: Number of attention heads that read from attention_states.
        output_projection: None or a pair (W, B) of output projection weights and
          biases; W has shape [output_size x num_decoder_symbols] and B has
          shape [num_decoder_symbols]; if provided and feed_previous=True, each
          fed previous output will first be multiplied by W and added B.
        feed_previous: True or False,if True, only the first
          of decoder_inputs will be used (the "GO" symbol), and all other decoder
          inputs will be taken from previous outputs (as in embedding_rnn_decoder).
          If False, decoder_inputs are used as given (the standard decoder case).
        dtype: The dtype of the initial RNN state (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
          "embedding_attention_seq2seq".
        initial_state_attention: If False (default), initial attentions are zero.
          If True, initialize the attentions from the initial state and attention
          states.
    Returns:
        A tuple of the form (outputs, state), where:
          outputs: A list of the same length as decoder_inputs of 2D Tensors with
            shape [batch_size x num_decoder_symbols] containing the generated
            outputs.
          state: The state of each decoder cell at the final time-step.
            It is a 2D Tensor of shape [batch_size x cell.state_size].
    Raises:
        Exception: When feed_previous has the wrong type.
    '''
    with variable_scope.variable_scope(scope or "embedding_attention_seq2seq",dtype=dtype) as scope:
        dtype = dtype
        # 建立encoder
        encoder_cell = copy.deepcopy(cell)
        encoder_cell = core_rnn_cell.EmbeddingWrapper(
            encoder_cell,embedding_classes=num_encoder_symbols,embedding_size=embedding_size)
        # 获得输出和最终的隐藏层状态
        encoder_outputs,encoder_states = rnn.static_rnn(
            encoder_cell,encoder_inputs,dtype=dtype)
        # encoder_outputs是encoder阶段的隐藏层向量，shape 为[timesteps,batch_size,output_size],需要转换成[batch_size,timesteps,output_size]
        # 1.首先建立列表，长度为T,元素是(batch_size,1,output_size)
        top_states = [array_ops.reshape(e,[-1,1,cell.output_size]) for e in encoder_outputs]
        # 2.将长度为T，元素为(batch_size,1,output_size)合并成[batch_size,T,output_size]的tensor
        attention_states = array_ops.concat(top_states,1)

        # 接下来是decoder阶段
        output_size = None
        if output_projection is None:
            cell = core_rnn_cell.OutputProjectionWrapper(cell,num_decoder_symbols)
            output_size = num_decoder_symbols

        if isinstance(feed_previous,bool):
            return embedding_attention_decoder(
                decoder_inputs,
                encoder_states,
                attention_states,
                cell,
                num_decoder_symbols,
                embedding_size,
                num_heads = num_heads,
                output_size=output_size,
                output_projection=output_projection,
                feed_previous=feed_previous,
                initial_state_attention=initial_state_attention)
        else:
            raise Exception('Invalid input feed_previous,please feed previous type is bool')


def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
    '''
    返回一个loop_function函数，该函数可以 extract the previous symbol and embeds it
    Args:
        embedding: embedding tensor for symbols.
        output_projection: None or a pair (W, B). If provided, each fed previous
          output will first be multiplied by W and added B.
        update_embedding: Boolean; if False, the gradients will not propagate
          through the embeddings.

    Returns:
        A loop function.
    '''
    def loop_function(prev,_):
        # 需要对embedding进行投影
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(prev,output_projection[0],output_projection[1])
        prev_symbol = math_ops.argmax(prev,1)

        # 根据prev_symbol 和 embedding矩阵获取 embedding vector
        emb_prev = embedding_ops.embedding_lookup(embedding,prev_symbol)
        # 不更新embedding 矩阵
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev
    return loop_function

def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads = 1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope = None,
                                initial_state_attention=False):
    '''
    decoder 模型，rnn decoder with embedding and attention
    Args:
        decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].(attn_length就是encoder过程的时间步timesteps,attn_size就是encoder的output_size)
        cell: tf.nn.rnn_cell.RNNCell defining the cell function.
        num_decoder_symbols: Integer, how many symbols come into the embedding.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        num_heads: Number of attention heads that read from attention_states.
        output_size: Size of the output vectors; if None, use output_size.
        output_projection: None or a pair (W, B) of output projection weights and
          biases; W has shape [output_size x num_symbols] and B has shape
          [num_symbols]; if provided and feed_previous=True, each fed previous
          output will first be multiplied by W and added B.
        feed_previous: Boolean; if True, only the first of decoder_inputs will be
          used (the "GO" symbol), and all other decoder inputs will be generated by:
            next = embedding_lookup(embedding, argmax(previous_output)),
          In effect, this implements a greedy decoder. It can also be used
          during training to emulate http://arxiv.org/abs/1506.03099.
          If False, decoder_inputs are used as given (the standard decoder case).
          True:可以理解为inference阶段，False:可以理解为train阶段
        update_embedding_for_previous: Boolean; if False and feed_previous=True,
          only the embedding for the first symbol of decoder_inputs (the "GO"
          symbol) will be updated by back propagation. Embeddings for the symbols
          generated from the decoder itself remain unchanged. This parameter has
          no effect if feed_previous=False.
          就是是否更新embedding_matrix，如果feed_previous=True and update_embedding_for_previous=False
          embedding_matrix只更新'_GO'的向量，其他symbol的向量不更新，如果feed_previous=False，代表是train阶段，
          这个变量就不起作用，很好理解，训练阶段肯定是要更新embedding_matrix的
        dtype: The dtype to use for the RNN initial states (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
          "embedding_attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
          If True, initialize the attentions from the initial state and attention
          states -- useful when we wish to resume decoding from a previously
          stored decoder state and attention states.

    Returns:
        A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
            shape [batch_size x output_size] containing the generated outputs.
        state: The state of each decoder cell at the final time-step.
            It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
        ValueError: When output_projection has the wrong shape.
    '''
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1],dtype= dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_decoder_symbols])

    with variable_scope.variable_scope(scope or 'embedding_attention_decoder',dtype=dtype) as scope:
        # 确定decoder 的 embedding matrix
        embedding = variable_scope.get_variable('embedding',[num_decoder_symbols,embedding_size])
        # 如果feed_previou=True,那么需要根据前一个symbol确定当前的输入embedding,否则不需要
        loop_function = _extract_argmax_and_embed(embedding,output_projection,update_embedding_for_previous) if feed_previous else None
        # 获取decoder 的 inputs 的embedding 矩阵 一个列表长度为decoder_inputs的时间长度 元素为(batch_size,embedding_size)
        emb_inp = [embedding_ops.embedding_lookup(embedding,i) for i in decoder_inputs]
        return attention_decoder(
            emb_inp,
            initial_state,
            attention_states,
            cell,
            output_size=output_size,
            num_heads=num_heads,
            loop_function=loop_function,
            initial_state_attention=initial_state_attention)


# 接下来就是attention_decoder 重中之重
def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
    '''
    rnn decoder with attention for the seq2seq model,attention机制就是可以关注encoder不同时刻的hidden state
    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
        output_size: Size of the output vectors; if None, we use cell.output_size.
        num_heads: Number of attention heads that read from attention_states.
        loop_function: If not None, this function will be applied to i-th output
          in order to generate i+1-th input, and decoder_inputs will be ignored,
          except for the first element ("GO" symbol). This can be used for decoding,
          but also for training to emulate http://arxiv.org/abs/1506.03099.
          Signature -- loop_function(prev, i) = next
            * prev is a 2D Tensor of shape [batch_size x output_size],
            * i is an integer, the step number (when advanced control is needed),
            * next is a 2D Tensor of shape [batch_size x input_size].
        dtype: The dtype to use for the RNN initial state (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
          If True, initialize the attentions from the initial state and attention
          states -- useful when we wish to resume decoding from a previously
          stored decoder state and attention states.

    Returns:
        A tuple of the form (outputs, state), where:
          outputs: A list of the same length as decoder_inputs of 2D Tensors of
            shape [batch_size x output_size]. These represent the generated outputs.
            Output i is computed from input i (which is either the i-th element
            of decoder_inputs or loop_function(output {i-1}, i)) as follows.
            First, we run the cell on a combination of the input and previous
            attention masks:
              cell_output, new_state = cell(linear(input, prev_attn), prev_state).
            Then, we calculate new attention masks:
              new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
            and then we calculate the output:
              output = linear(cell_output, new_attn).
          state: The state of each decoder cell the final time-step.
            It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
        ValueError: when num_heads is not positive, there are no inputs, shapes
          of attention_states are not set, or input size cannot be inferred
          from the input.
    '''
    if not decoder_inputs:
        raise ValueError('Must provide at least 1 input to attention decoder.')
    if num_heads < 1:
        raise ValueError('with less than 1 heads,use non attention decoder.')
    if attention_states.get_shape()[2].value is None:
        raise ValueError('Shape[2] of attention_states must be known.')

    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(scope or 'attention_decoder',dtype = dtype) as scope:

        dtype = scope.dtype
        # 获取批量数据
        batch_size = array_ops.shape(decoder_inputs[0])[0]
        # 编码阶段的时间步
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = array_ops.shape(attention_states)[1]
        # 获取encoder hidden state 的 size
        attn_size = attention_states.get_shape()[2].value

        # 接下来计算e_ij = v*tanh(W1*hj + w2 * si) e_ij 代表decoder i 时刻 对encoder j时刻的attention score
        # 其中j = 1,2,...,T_x,希望计算所有的W_1 * hj,希望可以一下子计算出所有的j的W1_hj，针对某个具体的j，可以从得到的tensor指定index获得
        # 因此，此时引入二维卷积tf.nn.conv2d(input, filter, strides, padding),可以并行的计算不同时刻的结果
        # input:[batch_size,in_height,in_width,in_channels]
        # filter:[filter_height,filter_width,in_channels,out_channels]
        # strides:一个一维向量，长度为4
        # padding:same/valid

        # 首先需要将attention_states reshape -> [batch_size,attn_length,1,attn_size]
        hidden = array_ops.reshape(attention_states,[-1,attn_length,1,attn_size])

        # 由于有num_heads:因此需要存储不同的num_head 对应的hidden_features 和 v
        hidden_features = []
        v = []
        attn_vec_size = attn_size
        for index in range(num_heads):
            k = variable_scope.get_variable("AttnW_%d" % index,[1,1,attn_size,attn_vec_size],dtype=dtype)
            # conv2d计算的张量结果shape [batch_size,attn_length,1,attn_vec_size]
            hidden_features.append(nn_ops.conv2d(hidden,k,[1,1,1,1],"SAME"))
            v.append(variable_scope.get_variable("AttnV_%d" % index,[attn_vec_size],dtype=dtype))

        # state 初始化为 encoder 传给 decoder 的 hidden state vec,最终结果是decoder 的 hidden state
        state = initial_state

        # 定义attention 函数，输入是decoder lstm 的隐藏向量si ,或者是encoder 最后时刻的hidden state
        # 输出是列表，长度为num_heads，其中列表中存储的是 针对query 的 attention context 向量[batch_size,attn_size]
        def attention(query):
            # results of attention ,列表长度为num_heads
            ds = []
            if nest.is_sequence(query):# if query is a tupple ,flatten it
                query_list = nest.flatten(query)
                for q in query_list:
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list,1)

            for index in range(num_heads):
                with variable_scope.variable_scope("Attention_%d"%index):
                    # 首先计算W2 * si,用y表示 其中W1 * h shape [batch_size,attn_length,1,attn_vec_size]
                    y = Linear(query,attn_vec_size,True)(query)
                    # 将y reshape 成 4维张量，方便和W1 * h 相加
                    y = array_ops.reshape(y,[-1,1,1,attn_vec_size])
                    y = math_ops.cast(y,dtype)

                    #  计算 W2 * si 和 W1 * h 的和
                    s = math_ops.reduce_sum(v[index] * math_ops.tanh(hidden_features[index] + y),[2,3])
                    a = nn_ops.softmax(math_ops.cast(s,dtype=dtypes.float32))

                    # 接下来计算context vector d
                    a = math_ops.cast(a,dtype)
                    d = math_ops.reduce_sum(array_ops.reshape(a,[-1,attn_length,1,1]) * hidden,[1,2])
                    ds.append(array_ops.reshape(d,[-1,attn_size]))
            return ds

        outputs = []
        prev = None
        batch_attn_size = array_ops.stack([batch_size,attn_size])
        # 不同的num_head 对应的不同的context vec
        attns = [array_ops.zeros(batch_attn_size,dtype=dtype) for _ in range(num_heads)]

        for a in attns:
            a.set_shape([None,attn_size]) # ensure the second shape of attention vectors is set

        # 如果采用encoder 传递的 hidden state
        if initial_state_attention:
            attns = attention(initial_state)

        # 遍历decoder inputs
        for i,inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
                # 如果loop_function is None ,说明是train 阶段 ，否则 是inference 阶段
                # 如果是inference 阶段 并且 前一时刻的输出 is not None
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function",reuse=True):
                    inp = loop_function(prev,i)
            # 确定i 时刻 decoder inputs 的 input_size
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError('could not inference input size from input')

            # 在decoder 的过程中也使用context vec 向量
            inputs = [inp] + attns
            inputs = [math_ops.cast(e,dtype) for e in inputs]
            x = Linear(inputs,input_size,True)(inputs)
            # run decode rnn
            cell_output,state = cell(x,state)

            # 接下来需要获取context vector
            # inference 阶段
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(variable_scope.get_variable_scope(),reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            # 将输出进行投影
            with variable_scope.variable_scope('AttnOutputProjection'):
                cell_output = math_ops.cast(cell_output,dtype)
                inputs = [cell_output] + attns
                output = Linear(inputs,output_size,True)(inputs)
            if loop_function is not None:
                prev = output
            outputs.append(output)
        return outputs,state










