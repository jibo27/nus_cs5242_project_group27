# CS5242 Project Codes

### Steps

1. preprocess videos and labels

```bash
python prepro_feats.py --output_dir mydata/feats/senet154 --model senet154 --frames_path mydata/train/train
python prepro_feats.py --output_dir mydata/test_feats/senet154 --model senet154 --frames_path mydata/test/test
```

2. Training a model

```bash

python train.py --epochs 3001 --batch_size 300 --model S2VTAttModel --dim_vid 2048 --rnn_type lstm --feats_dir mydata/feats/senet154 --test_feats_dir mydata/test_feats/senet154 --dim_hidden 1024
```

3. test
```bash
python eval.py --epochs 3001 --batch_size 300 --model S2VTAttModel --dim_vid 2048 --rnn_type lstm --test_feats_dir mydata/test_feats/senet154 --dim_hidden 1024 --ckpt_path results/75014/model_3000.pth
```

## Acknowledgements
The codes are based on [video-caption.pytorch](https://github.com/xiadingZ/video-caption.pytorch). Thanks for their excellent work!
