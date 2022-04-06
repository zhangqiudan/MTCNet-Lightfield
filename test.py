import argparse
import os
from dataset import get_loader
from solver import Solver


def main(config):
    if config.mode == 'test':
        test_loader, dataset = get_loader(config.test_batch_size, mode='test', num_thread=config.num_thread,
                                          test_mode=config.test_mode, sal_mode=config.sal_mode)

        test = Solver(None, test_loader, config, dataset.save_folder())
        test.test(test_mode=config.test_mode)
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':

    resnet_path = './weights/resnet50_caffe.pth'
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--resnet', type=str, default=resnet_path)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--load_bone', type=str, default='')
    parser.add_argument('--save_fold', type=str, default='./MTCNet')
    parser.add_argument('--pre_trained', type=str, default=None)
    parser.add_argument('--model', type=str, default='./MTCNet/MTCNet.pth')
    parser.add_argument('--test_mode', type=int, default=1)
    parser.add_argument('--sal_mode', type=str, default='t')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()

    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)
