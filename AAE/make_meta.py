# -*- coding: utf-8 -*-
import pickle as pkl
from collections import defaultdict

import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

from util import load_json, quantization


def main(train_fname, result_fname):
    print('Loading data ... ')
    data = load_json(train_fname)

    song_df = defaultdict(int)
    tag_df = defaultdict(int)
    sentences = []
    print('make meta file ... ')
    for d in tqdm(data):
        songs = d['songs']
        tags = d['tags']
        sentences.append(quantization(d['plylst_title']))

        for song in songs:
            song_df[song] += 1

        for tag in tags:
            tag_df[tag] += 1

    song_over_5_ids = [_ for _, df in song_df.items() if df >= 5]
    tag_over_5_ids = [_ for _, df in tag_df.items() if df >= 5]
    song_length = len(song_over_5_ids)
    tag_length = len(tag_over_5_ids)

    print('make w2v ... ')
    w2v = Word2Vec(sentences=sentences, size=128, window=10)

    zero_vec = np.zeros(w2v.wv.vectors.shape[1]).astype(np.float32)
    w2v.wv.vectors = np.append([zero_vec], w2v.wv.vectors, axis=0)
    word2idx = {_: idx + 1 for idx, _ in enumerate(w2v.wv.index2word)}
    word2idx[''] = 0

    print("song length : ", song_length)
    print("tag length : ", tag_length)
    print("vocab size : {} include padding zero ".format(len(word2idx)))
    song2idx = {song_over_5_ids[i]: i for i in range(song_length)}
    tag2idx = {tag_over_5_ids[i]: i for i in range(tag_length)}

    meta = {"tag_length": tag_length,
            "song_length": song_length,
            "song2idx": song2idx,
            "tag2idx": tag2idx,
            "w2v": w2v.wv.vectors,
            "word2idx": word2idx}

    with open(result_fname, 'wb') as fp:
        pkl.dump(meta, fp)

    print('END')


if __name__ == '__main__':

    main(train_fname='./res/train.json',
         result_fname='./arena_data/meta.pkl')

