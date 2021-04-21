import time

from gensim.models import KeyedVectors
import tensorflow.keras as keras
import os
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import *
import jieba
import tensorflow as tf
import numpy as np
from sina.models.newsEmbedding import NewsEmbedding
from mongoengine import Q, connect

MODEL_WEIGHTS_PATH = "D:\\sinaSpider_2\\sina\\resource\\model_weights.h5"
EMBEDDING_PATH = "D:\\sinaSpider_2\\sina\\resource\\embedding_500000.txt"
DICT_PATH = "D:\\sinaSpider_2\\sina\\resource\\dict.txt"
MAX_LEN = 20
_model, _word2idx, _idx2word, _embedding_matrix = None, {}, {}, []


class Encoder(keras.models.Model):
    def __init__(self, vocab_size, embedding_matrix, units=None, **kwargs):
        super().__init__(**kwargs)
        if units is None:
            units = [400, 200]
        self.embedding_layer = Embedding(
            vocab_size,
            output_dim=200,
            mask_zero=True,
            trainable=True,
            weights=[embedding_matrix])
        self.lstm_layer1 = GRU(units=units[0], return_sequences=True, return_state=True, dropout=0.25)
        self.lstm_layer2 = GRU(units=units[1], return_sequences=True, return_state=True, dropout=0.25)
        self.bn_layer = BatchNormalization()

    def call(self, inputs):
        embed = self.embedding_layer(inputs)
        bn = self.bn_layer(embed)
        encoder_output_1, state_h_1 = self.lstm_layer1(bn)

        encoder_output_2, state_h_2 = self.lstm_layer2(encoder_output_1)

        return [encoder_output_2, encoder_output_1], [state_h_2, state_h_1]


class Decoder(keras.models.Model):
    def __init__(self, vocab_size, embedding_matrix, units=None, **kwargs):
        super().__init__(**kwargs)
        if units is None:
            units = [200, 400]
        self.embedding_layer = Embedding(
            vocab_size,
            output_dim=200,
            trainable=True,
            mask_zero=True,
            weights=[embedding_matrix])
        self.attention_1 = Attention()
        self.attention_2 = Attention()
        self.lstm_layer1 = GRU(units=units[0], return_sequences=True, return_state=True, dropout=0.25)
        self.lstm_layer2 = GRU(units=units[1], return_sequences=True, return_state=True, dropout=0.25)
        self.bn_layer = BatchNormalization()

    def call(self, inputs, state, encoder_outputs):
        embed = self.embedding_layer(inputs)
        bn = self.bn_layer(embed)
        decoder_output_1, state_h_1 = self.lstm_layer1(bn, initial_state=state[0])
        decoder_output_2, state_h_2 = self.lstm_layer2(decoder_output_1, initial_state=state[1])
        return decoder_output_2, [state_h_1, state_h_2]


def load_embedding_idx(embedding_path):
    print("加载embedding。。。。")
    assert os.path.exists(embedding_path), "{} is not exist".format(embedding_path)

    wordVec = KeyedVectors.load_word2vec_format(fname=embedding_path, binary=False)
    wordVec.init_sims(replace=True)
    index = 4
    embedding_matrix = np.random.randn(len(wordVec.vocab.keys()) + index, 200)
    word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
    idx2word = {v: k for (k, v) in word2idx.items()}

    for word in wordVec.vocab.keys():
        embedding_matrix[index] = wordVec[word]
        word2idx[word] = index
        idx2word[index] = word
        index += 1
    print("embedding加载完成。。。。")

    return embedding_matrix, word2idx, idx2word


def get_model(model_weights_path, embedding_matrix):
    print("开始加载模型结构。。。")
    encoder_input = Input(shape=(20,), name="encoder_inputs")
    decoder_input = Input(shape=(21,), name="decoder_inputs")
    vocab_size = len(embedding_matrix)
    encoder_outputs, encoder_state = Encoder(vocab_size, embedding_matrix=embedding_matrix, name="encoder")(
        encoder_input)
    decoder_output, _ = Decoder(vocab_size, embedding_matrix=embedding_matrix, name="decoder")(inputs=decoder_input,
                                                                                               state=encoder_state,
                                                                                               encoder_outputs=encoder_outputs)

    output = Dense(vocab_size, activation="softmax", name="output")(decoder_output)

    model = Model([encoder_input, decoder_input], [output])
    print("模型结构加载完成。。。")
    print("开始加载模型参数。。。")
    model.load_weights(model_weights_path)
    print("模型参数加载完成。。。")

    return model


def load_model(model_weights_path, embedding_path):
    embedding_matrix, word2idx, idx2word = load_embedding_idx(embedding_path)
    print("开始加载模型。。。")
    model = get_model(model_weights_path=model_weights_path, embedding_matrix=embedding_matrix)
    return model, word2idx, idx2word, embedding_matrix


def cut_titles(news_titles):
    print("标题分词处理...")
    cut_result = []
    for title in news_titles:
        cut_result.append(jieba.lcut(title.strip()))
    print("标题分词完成...")

    return cut_result


def get_title_seq(cut_news_titles):
    global _word2idx
    title_seq = []
    for news_title in cut_news_titles:
        words_idx = [_word2idx.get(word, 3) for word in news_title]
        title_seq.append(words_idx)

    title_seq = keras.preprocessing.sequence.pad_sequences(sequences=title_seq, maxlen=MAX_LEN, padding="post")

    return title_seq


def get_user_interest_embedding(user_lab_list):
    global _embedding_matrix
    if len(user_lab_list) == 0:
        return np.zeros((200,))

    user_interest_embedding = []
    user_lab_cut_list = cut_titles(user_lab_list)
    for user_lab_line in user_lab_cut_list:
        for user_lab in user_lab_line:
            idx = _word2idx.get(user_lab, -1)
            if idx != -1:
                user_interest_embedding.append(_embedding_matrix[idx])

    return np.sum(user_interest_embedding, axis=-2).tolist()


def get_user_history_embedding(history_seq):
    if len(history_seq) == 0:
        return np.zeros((200,))

    user_history_embedding = []
    # 查找该新闻的embedding
    news_embedding_list = NewsEmbedding.objects.filter(doc_id__in=history_seq)
    for news_embedding in news_embedding_list:
        user_history_embedding.append(list(news_embedding["embedding"]))
    return np.mean(user_history_embedding, axis=-2).tolist()


def cal_news_score(user_interest_embedding, user_history_embedding, user_history):
    user_embedding_matrix = np.array([user_interest_embedding, user_history_embedding]).reshape((200, 2))
    last_time = int(time.time()) - 60 * 60 * 24 * 5
    news_embedding_obj_list = list(NewsEmbedding.objects(create_time__gt=last_time))  # 获取最近一周的新闻
    news_embedding_list = []

    for news_embedding_obj in news_embedding_obj_list:
        # 剔除掉已看过的
        if news_embedding_obj["doc_id"] not in user_history:
            news_embedding_list.append(list(news_embedding_obj["embedding"]))

    news_score = np.sum(np.dot(news_embedding_list, user_embedding_matrix), axis=-1)
    score_index = np.argsort(-news_score)  # 获取索引排序
    news_sorted = []
    for news_index in score_index:
        news_sorted.append(news_embedding_obj_list[news_index]["doc_id"])
    return news_sorted
    # 按照顺序给score排序


def model_init():
    global _model, _word2idx, _idx2word, _embedding_matrix
    print("开始初始化。。。")
    print("加载用户字典数据。。。")
    jieba.load_userdict(DICT_PATH)
    print("用户字典数据加载完成。。。")
    _model, _word2idx, _idx2word, _embedding_matrix = load_model(MODEL_WEIGHTS_PATH, EMBEDDING_PATH)
    print("初始化完成")


def get_title_embedding(news_titles):
    global _model, _word2idx, _idx2word
    assert _model is not None
    print("获取模型编码部分")
    encoder = _model.get_layer("encoder")
    cut_titles_seq = cut_titles(news_titles)
    print("获取标题分词embedding。。。")
    title_seq = get_title_seq(cut_titles_seq)
    print("获取标题分词embedding成功！")
    news_embedding = []
    print("获取标题分词embedding成功！")
    print("开始获取标题embedding。。。")
    index = 0
    for news_title in title_seq:
        print("编码标题：" + news_titles[index])
        index += 1
        _, encoder_state = encoder(tf.constant([news_title]))
        news_embedding.append(np.array(encoder_state[0]).reshape(200, ).tolist())
    print("获取标题embedding成功。。。")
    return news_embedding


def cal_similar_news(doc_id):
    target_news = list(NewsEmbedding.objects(doc_id=doc_id))
    target_embedding = np.array(list(target_news[0]["embedding"])).reshape((200, 1))
    last_time = int(time.time()) - 3600 * 24 * 5
    news_embedding_obj = list(NewsEmbedding.objects(create_time__gt=last_time))
    news_embedding_list = []
    for news_embedding in news_embedding_obj:
        news_embedding_list.append(list(news_embedding["embedding"]))

    news_similar_score = np.dot(news_embedding_list, target_embedding).reshape(len(news_embedding_list), )
    score_index = np.argsort(-news_similar_score)  # 获取索引排序
    news_sorted = []
    for news_index in score_index[:100]:
        similar_doc_id = news_embedding_obj[news_index]["doc_id"]
        if doc_id != similar_doc_id:
            news_sorted.append(similar_doc_id)
    return news_sorted


if __name__ == '__main__':
    doc_id = "comos-kmyaawa9095839"
    import json
    import redis

    rd = redis.StrictRedis(host="127.0.0.1", port=6379, db=0)
    connect("news_recommender")

    similar_news_list = cal_similar_news(doc_id)
    print(similar_news_list)
    # rd.set(doc_id + "_similar", json.dumps(similar_news_list), ex=60 * 20)
