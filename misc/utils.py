import torch
import torch.nn as nn
from torch.autograd import Variable
from datetime import datetime


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    seq = seq.cpu()
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j].item()
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.contiguous().view(-1)
        reward = reward.contiguous().view(-1)
        mask = (seq > 0).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1).cuda(),
                         mask[:, :-1]], 1).contiguous().view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output


class ObjRelCriterion(nn.Module):

    def __init__(self):
        super(ObjRelCriterion, self).__init__()
        #self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target):
        """
        logits: shape of [(N, obj_size), (N, rel_size), (N, obj_size)]
        target: shape of [(N,), (N,), (N,)]
        """
        # truncate to the same size
        batch_size = logits[0].shape[0]
        #target = target[:, :logits.shape[1]]
        #logits = logits.contiguous().view(-1, logits.shape[2])
        #target = target.contiguous().view(-1)
        loss = 0
        for i in range(len(logits)):
            loss += self.loss_fn(logits[i].squeeze(), target[:, i])
        output = torch.sum(loss) / batch_size
        return output



def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def digit2word(caption_digit, object2idx, relationship2idx):
    caption_word = ''
    idx2object = {idx:obj for obj, idx in object2idx.items()}
    idx2relationship = {idx:rel for rel, idx in relationship2idx.items()}
    caption_word += '%s'%idx2object[caption_digit[0]]
    caption_word += ' %s'%idx2relationship[caption_digit[1]]
    caption_word += ' %s'%idx2object[caption_digit[2]]
    return caption_word


