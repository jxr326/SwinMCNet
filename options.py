import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=48, help='epoch number')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--testsize', type=int, default=384, help='testing dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=5e-4, help='decay rate of learning rate')

parser.add_argument('--load', type=str, default='../res/swin_base_patch4_window12_384_22k.pth')


parser.add_argument('--train_data_root', type=str, default='../VT5000/Train', help='the training datasets root')
parser.add_argument('--val_data_root', type=str, default='../VT5000/Test', help='the value datasets root')
parser.add_argument('--test_data_root', type=str, default='../dataset/')


parser.add_argument('--save_path', type=str, default='../res/', help='the path to save models and logs')
parser.add_argument('--test_model', type=str, default='../res/SwinMCNet_epoch_best.pth', help='saved model path')
parser.add_argument('--maps_path', type=str, default='../maps/', help='saved model path')

opt = parser.parse_args()
