# -*- coding: utf-8 -*-
import pickle as pkl
import fire
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from util import load_json
from aae import *


class PlayListDataset(Dataset):
    def __init__(self, dataset, meta_fname):
        self.data = dataset  # numpy array
        with open(meta_fname, 'rb') as fp:
            meta = pkl.load(fp)
        self.song_length = meta['song_length']
        self.song2idx = meta['song2idx']
        self.tag_length = meta['tag_length']
        self.tag2idx = meta['tag2idx']
        self.word2idx = meta['word2idx']
        self.len = len(self.data)

    def _make_array(self, data):
        songs = torch.zeros(self.song_length)
        tags = torch.zeros(self.tag_length)
        song_idx = [self.song2idx[song] for song in data['songs'] if song in self.song2idx]
        songs[song_idx] = 1.
        tag_idx = [self.tag2idx[tag] for tag in data['tags'] if tag in self.tag2idx]
        tags[tag_idx] = 1.
        return torch.cat((songs, tags))

    def __getitem__(self, index):
        return self._make_array(self.data[index])

    def __len__(self):
        return self.len


def main():
    data_fname = './arena_data/orig/train.json'
    meta_fname = './arena_data/meta.pkl'
    result_fname = './res/model/deepreco'
    batch_size = 256
    num_epochs = 200
    noise_prob = 0.8
    check_point = 10

    # train-val split
    # train-val split
    raw_data = load_json(data_fname)
    train, val = train_test_split(np.array(raw_data), train_size=0.95, random_state=128)
    train_dataset = PlayListDataset(train, meta_fname)
    val_dataset = PlayListDataset(val, meta_fname)

    # data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)
    song_length = train_dataset.song_length
    tag_length = train_dataset.tag_length

    input_size = song_length + tag_length

    # check the model
    model = AAE(n_input=input_size)
    print('model : ')
    print(model)


    # check available gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device : {}'.format(device))

    model.to(device)

    # de-noise
    if noise_prob > 0:
        dp = nn.Dropout(p=noise_prob)

    # train
    recon_losses = []
    disc_losses = []
    gen_losses = []
    val_loss_array = []
    for epoch in range(num_epochs):
        t_denom = 0.0
        total_recon_t_loss = 0.0
        total_disc_t_loss = 0.0
        total_gen_t_loss = 0.0
        print('Doing epoch {} / {}'.format(epoch + 1, num_epochs))
        model.train()
        for pl in tqdm(train_loader):
            pl = pl.view(pl.size(0), -1)
            inputs = Variable(pl.to(device))
            # forward
            if noise_prob > 0:
                input_noise = inputs.clone()
                # hide and seek
                r = np.random.randint(3)
                if r == 0:
                    input_noise[:, song_length:] = 0.
                elif r == 1:
                    input_noise[:, :song_length] = 0.

                # forward(+denoising)
                inputs_noise = dp(inputs) * (1 - noise_prob)
                output, recon_loss, disc_loss, gen_loss = model(inputs_noise)
            else:
                output, recon_loss, disc_loss, gen_loss = model(inputs)
            t_denom += 1.
            total_recon_t_loss += recon_loss
            total_disc_t_loss += disc_loss
            total_gen_t_loss += gen_loss

        recon_t_loss = total_recon_t_loss / t_denom
        disc_t_loss = total_disc_t_loss / t_denom
        gen_t_loss = total_gen_t_loss / t_denom

        log_losses(recon_t_loss, disc_t_loss, gen_t_loss)

        recon_losses.append(recon_t_loss)
        disc_losses.append(disc_t_loss)
        gen_losses.append(gen_t_loss)

        print("Doing Validation ..")
        model.eval()
        e_denom = 0.0
        total_epoch_loss = 0.0
        for pl in val_loader:
            pl = pl.view(pl.size(0), -1)
            inputs = Variable(pl.to(device))
            outputs = model.predict(inputs)
            loss = BCEloss(outputs, inputs)
            e_denom += 1
            total_epoch_loss += loss.item()
        val_loss = total_epoch_loss / e_denom
        print('val loss : {:.4f}'.format(val_loss))
        val_loss_array.append(val_loss)
        if epoch % check_point == 0:
            torch.save(model.state_dict(), '{}_{}'.format(result_fname, epoch + 1))
            with open('{}_loss'.format(result_fname), 'wb') as fp:
                pkl.dump((recon_losses, disc_losses, gen_losses, val_loss_array), fp)
    torch.save(model.state_dict(), '{}_{}'.format(result_fname, epoch + 1))
    with open('{}_loss'.format(result_fname), 'wb') as fp:
        pkl.dump((recon_losses, disc_losses, gen_losses, val_loss_array), fp)

if __name__ == '__main__':
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)
    torch.manual_seed(128)
    fire.Fire(main)