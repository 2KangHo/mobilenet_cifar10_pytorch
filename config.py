from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='MobileNet PyTorch')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='training batch size (default: 1024)')
parser.add_argument('--test_batch_size', type=int, default=256,
                    help='testing batch size (default: 256)')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train for (Default: 150)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Learning Rate (Default: 0.1)')
parser.add_argument('--momentum', default=0.9,
                    type=float, help='SGD Momentum (Default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, help='Weight decay (Default: 5e-4)')
parser.add_argument('--workers', type=int, default=16,
                    help='number of data loading workers (default: 16)')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--gpuids', default=[0], nargs='+',
                    help='GPU IDs for using (Default: 0)')
parser.add_argument('--ckpt', default='', type=str, metavar='PATH',
                    help='path of checkpoint for resuming/testing model (Default: none)')
parser.add_argument('--resume', action='store_true', help='resume model?')
parser.add_argument('--eval', action='store_true', help='test model?')

cfg = parser.parse_args()
