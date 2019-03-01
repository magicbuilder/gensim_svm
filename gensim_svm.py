import os
import json
from gensim import corpora, models
import re
from pprint import pprint
import os
from nlp_common_lib.lib.stopwords import StopWords
import jieba
from scipy.sparse import csr_matrix
from sklearn import svm
from sklearn.datasets.base import Bunch
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import gc

__invalid_reference = re.compile('^[\d\.a-zA-Z]+$')


def is_invalid_word(w):
    """
    判断词是否可用。
    """
    if __invalid_reference.findall(w) or \
            StopWords.is_stop_words(w) or \
            len(w) < 2:
        return True
    return False


def load_json(fp):
    with open(fp, encoding='utf-8') as fi:
        data = json.load(fi)
    return data


def dump_json(data, fp):
    with open(fp, 'w', encoding='utf-8') as fi:
        json.dump(data, fi, ensure_ascii=False)


def segment(news):
    words = [w for w in jieba.cut(news) if not is_invalid_word(w)]
    return words


def read_data(json_fp):
    """
    读入数据，生成内容列表
    """
    data = load_json(json_fp)
    content_list = []
    for record in data['RECORDS']:
        content_list.append(record['Content'])
    return content_list


def generate_vector_model(data, dic_fp, model_fp):
    _dictionary = corpora.Dictionary()
    corpus_segment = []
    print('Segment data..')
    counter = 0
    data_size = len(data)
    for record in data:
        record = segment(record)
        corpus_segment.append(record)
        _dictionary.add_documents([record])
        counter += 1
        print('\r%d/%d..' % (counter, data_size), end='')
    # small_freq_ids = [tokenid for tokenid, docfreq in _dictionary.dfs.items() if docfreq < 2]
    small_freq_ids = sorted(
        [(tokenid, docfreq) for tokenid, docfreq in _dictionary.dfs.items()], key=lambda x: x[1]
    )[:-10000]
    small_freq_ids = [tokenid for tokenid, docfreq in small_freq_ids]
    _dictionary.filter_tokens(small_freq_ids)
    _dictionary.compactify()
    print(_dictionary)
    print('Dumping dictionary which has %d words to %s..' % (len(_dictionary), dic_fp))
    joblib.dump(_dictionary, dic_fp)
    corpus = []
    for words in corpus_segment:
        bow = _dictionary.doc2bow(words)
        corpus.append(bow)
    _tfidf_model = models.TfidfModel(corpus=corpus, dictionary=_dictionary)
    # tfidf_corpus = [_tfidf_model[bow] for bow in corpus]
    print('Dumping model to %s..' % model_fp)
    # lsi_model = models.LsiModel(corpus=tfidf_corpus, num_topics=300)
    joblib.dump(_tfidf_model, model_fp)
    return _dictionary, _tfidf_model


def load_dictionary(fp):
    _dictionary = joblib.load(fp)
    print('Got a dictionary with %d words..' % len(_dictionary))
    return _dictionary


def convert_text_vector(text, _dictionary, vector_model):
    words = segment(text)
    bow = _dictionary.doc2bow(words)
    vectors = [vector_model[bow]]
    _dict_len = len(_dictionary)
    data, rows, cols, line_count = [], [], [], 0
    for vector in vectors:
        for elem in vector:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
        line_count += 1
    if data:
        sparse_matrix = csr_matrix((data, (rows, cols)))
        matrix = sparse_matrix.toarray()
        vector = list(matrix[0])
        vector += [0] * (_dict_len - len(vector))
    else:
        vector = [0] * _dict_len
    return vector


# model = joblib.load("data/model/journey_news_based_on_content-20190228v2.m")
# dictionary = joblib.load("data/model/dic for journey-20190228v2.dic")
# tfidf_model = joblib.load("data/model/tfidf_model for journey-20190228v2.m")
# dict_len = len(dictionary)


def test(text):
    words = segment(text)
    bow = dictionary.doc2bow(words)
    corpus_tfidf = [tfidf_model[bow]]
    data = []
    rows = []
    cols = []
    line_count = 0
    for line in corpus_tfidf:
        for elem in line:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
        line_count += 1
    if data:
        sparse_matrix = csr_matrix((data, (rows, cols)))
        matrix = sparse_matrix.toarray()
        vector = list(matrix[0])
        vector += [0] * (dict_len - len(vector))
    else:
        vector = [0] * dict_len
    res = model.predict([vector])[0]
    # if res == 1:
    #     print('pos; ', text)
    # elif res == 0:
    #     print('neg; ', text)
    # else:
    #     print("error")
    return res


def train():
    # pos_data = load_json('data/fashion_content_list.json')[:3000]
    neg_data = load_json('data/non_journey_news.json')
    data1 = read_data('data/journey_doc.json')
    data2 = read_data('data/journey_test.json')
    data3 = read_data('data/journey_train.json')
    pos_data = []
    pos_data += data1
    pos_data += data2
    pos_data += data3
    limit = 3000
    pos_data, neg_data = pos_data[:limit], neg_data[:limit]
    label = []
    label += [1] * len(pos_data)
    label += [0] * len(neg_data)
    data = []
    data += pos_data
    data += neg_data

    print('Generating dictionary and model..')
    _dictionary, vector_model = generate_vector_model(
        data, 'data/model/dict-for-journey-20190228v3.dic', 'data/model/tfidf-model-for-journey-20190228v3.m'
    )
    print('Converting text to vectors..')
    data_matrix = []
    counter = 0
    data_size = len(data)
    for text in data:
        vector = convert_text_vector(text, _dictionary, vector_model)
        data_matrix.append(vector)
        counter += 1
        print('\r%d/%d..' % (counter, data_size), end='')

    print('Split data to test and train set..')
    x_train, x_test, y_train, y_test = train_test_split(data_matrix, label, test_size=0.2)
    clf = svm.LinearSVC()
    print('Training..')
    clf.fit(x_train, y_train)
    print('Saving model..')
    joblib.dump(clf, "data/model/journey_news_based_on_content-20190228v3.m")
    right_num = 0
    for c1, c2 in zip(clf.predict(x_test), y_test):
        if c1 == c2:
            right_num += 1
    precision = right_num / len(x_test)
    print(precision)


def test2():
    # data = load_json('data/non_journey_news.json')
    data1 = read_data('data/journey_doc.json')
    data2 = read_data('data/journey_test.json')
    data3 = read_data('data/journey_train.json')
    pos_data = []
    pos_data += data1
    pos_data += data2
    pos_data += data3
    data = pos_data
    counter, pos_counter = 0, 0
    for text in data:
        res = test(text)
        if res == 1:
            pos_counter += 1
        counter += 1
        print('\r%d/%d..' % (pos_counter, counter), end='')


if __name__ == '__main__':
    # train()
    # load_dictionary('data/model/dict-for-journey-20190228v3.dic')
    # os._exit(0)
    a=load_json('data/fashion_content.json')
    print(len(a) )
