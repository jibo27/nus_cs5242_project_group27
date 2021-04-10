import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):

#    def get_vocab_size(self):
#        return 118
        # return len(self.get_vocab())
    def get_obj_vocab_size(self):
        return 35

    def get_rel_vocab_size(self):
        return 82

    def __init__(self, opt, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode  # to load train/val/test data

        # load the json file which contains information about the dataset
        self.captions = json.load(open(opt["caption_json"]))
        
        if self.mode == 'train':
            self.len = opt["len"]
            self.feats_dir = opt["feats_dir"]
        else:
            self.len = opt["len_test"]
            self.feats_dir = opt["test_feats_dir"]
        print('load feats from %s' % (self.feats_dir))
        # load in the sequence data
        self.max_len = opt["max_len"]
        print('max sequence length in data is', self.max_len)

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        # which part of data to load
        # if self.mode == 'val':
        #     ix += len(self.splits['train'])
        # elif self.mode == 'test':
        #     ix = ix + len(self.splits['train']) + len(self.splits['val'])
        annix = '{0:06}'.format(ix)
        fc_feat = []
        ### question1 ###
        for dir in self.feats_dir: ## can concatenate features form 
            fc_feat.append(np.load(os.path.join(dir, '{}.npy'.format(annix))))
        fc_feat = np.concatenate(fc_feat, axis=1)

        if self.mode=='test':
            data = {}
            data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
            data['video_ids'] = annix
            return data
        label = np.zeros(self.max_len)
        # mask = np.zeros(self.max_len)
        captions = np.array(self.captions[annix])
        
        # bos: 117; eos: 118; (pad: 119)
        label = np.array([captions[0],captions[1],captions[2]])

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        # data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        #data['gts'] = torch.from_numpy(captions).long()
        data['video_ids'] = annix
        return data

    def __len__(self):
        return self.len
