# MobileNet CIFAR10 PyTorch

MobileNet with CIFAR10 Implementation on PyTorch  
This Code is possible to resume and evaluate model on different GPUs or CPU environment from trained model checkpoint.  
And you can train this model on multi-gpu.

## Requirements

- `python3` (python 3.5+) (Because of using pathlib)
- `tqdm`
- `torch` (PyTorch 0.4.0+)
- `torchvision`
- `numpy`

## Usage

```
usage: main.py [-h] [--batch_size BATCH_SIZE]
               [--test_batch_size TEST_BATCH_SIZE] [--epochs EPOCHS] [--lr LR]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--workers WORKERS] [--cuda] [--gpuids GPUIDS [GPUIDS ...]]
               [--ckpt PATH] [--resume] [--eval]

optional arguments:
      -h, --help            show this help message and exit
      --batch_size BATCH_SIZE
                            training batch size (default: 1024)
      --test_batch_size TEST_BATCH_SIZE
                            testing batch size (default: 256)
      --epochs EPOCHS       number of epochs to train for (Default: 150)
      --lr LR               Learning Rate (Default: 0.1)
      --momentum MOMENTUM   SGD Momentum (Default: 0.9)
      --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                            Weight decay (Default: 5e-4)
      --workers WORKERS     number of data loading workers (default: 16)
      --cuda                use cuda?
      --gpuids GPUIDS [GPUIDS ...]
                            GPU IDs for using (Default: 0)
      --ckpt PATH           path of checkpoint for resuming/testing model
                            (Default: none)
      --resume              resume model?
      --eval                test model?
```

### Training
```
> python3 main.py --cuda --gpuids 0 1 2 3 --epochs 150
```

#### resume training
```
> python3 main.py --cuda --gpuids 0 1 2 3 --ckpt ckpt_epoch_100.pth --resume
```

### Evaluation
```
> python3 main.py --cuda --gpuids 0 1 2 3 --ckpt ckpt_epoch_150.pth --eval
```
or
```
> python3 main.py --cuda --gpuids 0 1 2 3 --ckpt ckpt_best.pth --eval
```

