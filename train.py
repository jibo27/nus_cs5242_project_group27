import json
import os
import csv

import numpy as np

import misc.utils as utils
import opts
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from dataloader import VideoDataset
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel
from torch import nn
from torch.utils.data import DataLoader


def train(loader, model, crit, optimizer, lr_scheduler, opt, testloader):
    object2idx = json.load(open('mydata/object1_object2.json', 'r'))
    relationship2idx = json.load(open('mydata/relationship.json', 'r'))
    idx2object = {idx:obj for obj, idx in object2idx.items()}
    idx2relationship = {idx:rel for rel, idx in relationship2idx.items()}

    model.train()
    for epoch in range(opt["epochs"]):
        lr_scheduler.step()

        iteration = 0
        # If start self crit training

        for data in loader:
            torch.cuda.synchronize()
            fc_feats = data['fc_feats'].cuda()
            labels = data['labels'].cuda()
            #masks = data['masks'].cuda()

            optimizer.zero_grad()
            seq_probs, _ = model(fc_feats, labels, 'train')
            loss = crit(seq_probs, labels)

            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            iteration += 1

            print("iter %d (epoch %d), train_loss = %.6f" %
                  (iteration, epoch, train_loss))

        if epoch % opt["save_checkpoint_every"] == 0:
            csvfile_digit_test = open(os.path.join(opt["checkpoint_path"], 'test_%d_digit.csv'%epoch), 'w', newline='')
            writer_digit_test = csv.DictWriter(csvfile_digit_test, fieldnames=['ID', 'label'], delimiter=',')
            writer_digit_test.writeheader()

            csvfile_word_test = open(os.path.join(opt["checkpoint_path"], 'test_%d_word.csv'%epoch), 'w', newline='')
            writer_word_test = csv.DictWriter(csvfile_word_test, fieldnames=['ID', 'label'], delimiter=',')
            writer_word_test.writeheader()


            model.eval()
            iteration_test = 0
            for data in testloader:
                if iteration_test >= 1:
                    assert False
                torch.cuda.synchronize()
                fc_feats = data['fc_feats'].cuda()

                with torch.no_grad():
                    seq_logprobs, preds5_list = model(fc_feats, None, 'test')
                torch.cuda.synchronize()
                iteration_test += 1
            model.train()

            preds = torch.stack(preds5_list, 1) # (119, 3, 5)

            torch.cuda.synchronize()

            idx = 0
            for vi in range(preds.shape[0]):
                for caption_id in range(preds.shape[1]):
                    indices = preds[vi][caption_id].cpu().numpy()
                    digits = ' '.join([str(elm) for elm in indices])
                    if caption_id == 1: # relationship
                        words = ' '.join([idx2relationship[elm] for elm in indices])
                    else: # object
                        words = ' '.join([idx2object[elm] for elm in indices])

                    writer_digit_test.writerow({'ID': idx, 'label': digits})
                    writer_word_test.writerow({'ID': idx, 'label': words})
                    idx += 1

            csvfile_digit_test.close()
            csvfile_word_test.close()
            print('%d is saved successfully!!'%(epoch))

    model_path = os.path.join(opt["checkpoint_path"], 'model_final.pth' )
    torch.save(model.state_dict(), model_path)
    print("model saved to %s" % (model_path))


def main(opt):
    dataset = VideoDataset(opt, 'train')
    dataset_test = VideoDataset(opt, 'test')
    dataloader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=opt["batch_size"], shuffle=False)
    opt["obj_vocab_size"] = dataset.get_obj_vocab_size()
    opt["rel_vocab_size"] = dataset.get_rel_vocab_size()
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt['dim_vid'],
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            rnn_dropout_p=opt["rnn_dropout_p"])
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["dim_vid"],
            opt["dim_hidden"],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(
            opt["obj_vocab_size"],
            opt["rel_vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=opt["bidirectional"])
        model = S2VTAttModel(encoder, decoder)
    model = model.cuda()
    crit = utils.ObjRelCriterion()
    #rl_crit = utils.RewardCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    train(dataloader, model, crit, optimizer, exp_lr_scheduler, opt, dataloader_test)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    opt["checkpoint_path"] = os.path.join(opt["checkpoint_path"], '%s'%utils.get_timestamp())
    os.makedirs(opt["checkpoint_path"], exist_ok=False)

    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))
    main(opt)
