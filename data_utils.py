'''
    author：fengjiaxin
    desc：该文件主要负责数据处理
'''
import os
import re


# 定义几个特殊的自负
_PAD = '_PAD'
_GO = '_GO' # 开始预测的符号
_EOS = '_EOS' # 结束预测的符号
_UNK = '_UNK'
_START_VOCAB = [_PAD,_GO,_EOS,_UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


# 常用的切分词，也就是说标点符号和数字
_WORD_SPLIT = re.compile(r"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")


def basic_tokenizer(sentence):
    '''
        basie tokenizer,split the sentence into a list of tokens
        将一个句子切分成单词的列表
    '''
    words = []
    for space_seperated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT,space_seperated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path,data_path,max_vocabulary_size,tokenizer=None,normalize_digits=True):
    '''
    根据data_path 文件 创建 vocabulary_path 文件，其中data_path 文件是一行一行的，每行被空格分割
    vocabulary_path文件中每一行代表一个单词，且按照其在data_path中的出现频数从大到小排列
    tokenizer:基本的切词函数
    normalize_digits:if true 将所有的数字都用0代替
    '''
    if not os.path.exists(vocabulary_path):
        with open(data_path,'r') as f:
            vocab_dict = dict()
            for index,line in enumerate(f):
                if index % 100000 == 0:
                    print('process %d lines'%index)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = re.sub(_DIGIT_RE,r"0",w) if normalize_digits else w
                    if word in vocab_dict:
                        vocab_dict[word] += 1
                    else:
                        vocab_dict[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab_dict,key=vocab_dict.get,reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with open(vocabulary_path,'w') as w:
            for word in vocab_list:
                w.write(word + '\n')

def initialize_vocabulary(vocabulary_path):
    '''
    Initialize vocabulary from file.
      We assume the vocabulary is stored one-item-per-line, so a file:
        dog
        cat
      will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
      also return the reversed-vocabulary ["dog", "cat"].
      Args:
        vocabulary_path: path to the file containing the vocabulary.
      Returns:
        a pair: the vocabulary (a dictionary mapping string to integers), and
        the reversed vocabulary (a list, which reverses the vocabulary mapping).
      Raises:
        ValueError: if the provided vocabulary_path does not exist.
    '''
    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path,'r') as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([x,y] for (y,x) in enumerate(rev_vocab))
        return vocab,rev_vocab
    else:
        raise ValueError('vocabulary path not exists')


# 将sentence 句子转换成 id 的列表 我 爱 你 -> [1,2,20]
def sentence_to_ids(sentence,vocab_dict,tokenizer=None,normalize_digits=True):
    words = tokenizer(sentence) if tokenizer else basic_tokenizer(sentence)
    if normalize_digits:
        return [vocab_dict.get(re.sub(_DIGIT_RE,r"0",w),UNK_ID) for w in words]
    else:
        return [vocab_dict.get(w,UNK_ID) for w in words]



# 将字符串为元素的数据文件转换成以id为元素的数据文件，例如我 爱 你 转换成0 2 3
def data_to_token_ids(data_path,target_path,vocabulary_path):
    if not os.path.exists(target_path):
        print('token data in %s'%(data_path))
        vocab_dict,_ = initialize_vocabulary(vocabulary_path)
        with open(data_path,'r') as f,open(target_path,'w') as w:
            for index,line in enumerate(f):
                if index % 10000 == 0:
                    print('tokenizing line %d'%index)
                sentence = line.strip()
                token_ids = sentence_to_ids(sentence,vocab_dict)
                w.write(' '.join([str(tok) for tok in token_ids]) + '\n')

