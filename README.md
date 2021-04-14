# CS5242 Project Codes


### Environments
+ Python: 3.8.5
+ torch: 1.7.1
+ torchvision: 0.8.2
+ pretrainedmodels: 0.7.4

I've tested the codes under the above environment. Feel free to run and check our codes!


### Steps

1. preprocess videos and labels

Before running the following scripts, please set the `frames_path` and `output_dir` correctly.
```bash
# Training Features
python prepro_feats.py --output_dir mydata/feats/senet154 --model senet154 --frames_path mydata/train/train

# Testing Features
python prepro_feats.py --output_dir mydata/test_feats/senet154 --model senet154 --frames_path mydata/test/test
```

2. Training a model

Before running the following scripts, please set the `feats_dir`, `test_feats_dir`, `caption_json`, `object1_object2_json` and `relationship_json` correctly.
```bash

python train.py --epochs 3001 --batch_size 300 --model S2VTAttModel --dim_vid 2048 --rnn_type lstm --dim_hidden 1024 \
                --feats_dir mydata/feats/senet154 \
                --test_feats_dir mydata/test_feats/senet154 \
                --caption_json mydata/training_annotation.json \
                --object1_object2_json mydata/object1_object2.json \
                --relationship_json mydata/relationship.json
```
The output result is saved in `results/<current datetime>`.

3. test

Before running the following scripts, please set the `test_feats_dir`, `caption_json`, `object1_object2_json`, `relationship_json` and `ckpt_path` correctly.

If you want to use the model with 0.75014 score, please download the pre-trained checkpoint model [here](https://drive.google.com/file/d/1vq7DFH_HiHuPkb6H6mpzFzKLNkh4OUzM/view?usp=sharing).
```bash
python eval.py --epochs 3001 --batch_size 300 --model S2VTAttModel --dim_vid 2048 --rnn_type lstm --dim_hidden 1024 \
               --test_feats_dir mydata/test_feats/senet154 \
               --ckpt_path results/75014/model_3000.pth \
               --caption_json mydata/training_annotation.json \
               --object1_object2_json mydata/object1_object2.json \
               --relationship_json mydata/relationship.json
```

The output results are saved as `results/test_digit.csv` and `results/test_word.csv`.

## Acknowledgements
The codes are based on [video-caption.pytorch](https://github.com/xiadingZ/video-caption.pytorch). Thanks for their excellent work!
