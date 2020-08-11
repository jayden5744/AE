# -*- coding:utf-8 -*-
import pickle as pkl
from tqdm import tqdm

import torch
import torch.nn

from arena_util import *
from aae import AAE


class GetAnswer:
    def __init__(self, meta_fname, model_fname, question_fname, result_fname):
        with open(meta_fname, 'rb') as fp:
            meta = pkl.load(fp)
        self.song_length = meta['song_length']
        self.tag_length = meta['tag_length']
        self.song2idx = meta['song2idx']
        self.tag2idx = meta['tag2idx']
        self.word2idx = meta['word2idx']
        self.idx2song = {idx:song for song, idx in self.song2idx.items()}
        self.idx2tag = {idx:tag for tag, idx in self.tag2idx.items()}
        self.model_fname = model_fname
        self.question_fname = question_fname
        self.result_fname = result_fname

    def _make_array(self, data):
        songs = torch.zeros(self.song_length)
        tags = torch.zeros(self.tag_length)
        song_idx = [self.song2idx[song] for song in data['songs'] if song in self.song2idx]
        tag_idx = [self.tag2idx[tag] for tag in data['tags'] if tag in self.tag2idx]
        songs[song_idx] = 1.
        tags[tag_idx] = 1.
        return torch.cat((songs, tags))

    def _generate_answers(self, questions):
        input_size = self.song_length + self.tag_length
        model = AAE(n_input=input_size)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device : {}'.format(device))

        model.to(device)
        print("Load model's parameter ... ")
        model.load_state_dict(torch.load(self.model_fname))

        model.eval()

        answers = []

        for question in tqdm(questions):
            inputs = self._make_array(question).to(device)
            outputs = model.predict(inputs)
            res = outputs.cpu().detach().numpy()

            song_rating = res[:self.song_length]
            tag_rating = res[self.song_length:]
            song_idx = np.argsort(song_rating)[::-1][:200]
            tag_idx = np.argsort(tag_rating)[::-1][:100]
            songs = [self.idx2song[idx] for idx in song_idx]
            tags = [self.idx2tag[idx] for idx in tag_idx]
            answers.append({
                "id": question["id"],
                "songs": remove_seen(question["songs"], songs)[:100],
                "tags": remove_seen(question["tags"], tags)[:10],
            })
        return answers

    def run(self):
        print('Loading question file ... ')
        questions = load_json(self.question_fname)

        print("Writing answers ... ")
        answers = self._generate_answers(questions)
        write_json(answers, self.result_fname)


if __name__ == '__main__':
    get_answer = GetAnswer(meta_fname='./arena_data/meta.pkl',
                           model_fname='./res/model/deepreco_101',
                           question_fname='./arena_data/val.json',
                           result_fname='./results/results.json')
    get_answer.run()
