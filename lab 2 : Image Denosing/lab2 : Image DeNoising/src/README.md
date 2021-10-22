## How to run

### 1. Dependences
```shell
pip install -r requirements.txt
```

### 2. Train DnCNN-S (DnCNN with known noise level)
```shell
python train.py \
  --preprocess True \
  --num_of_layers 17 \
  --noiseL 25 \
  --val_noiseL 25
```
**NOTE**

* If you've already built the training and validation dataset (i.e. train.h5 & val.h5 files), set *preprocess* to be False.
* According to the paper, DnCNN-S has 17 layers.
* *noiseL* is used for training and *val_noiseL* is used for validation. They should be set to the same value for unbiased validation. You can set whatever noise level you need.

### 3. Test
```shell
python test.py \
  --num_of_layers 17 \
  --test_noiseL 15
```
**NOTE**
* Set *num_of_layers* to be 17 when testing DnCNN-S models.
* *test_noiseL* is used for testing. This should be set according to which model your want to test (i.e. *logdir*).
