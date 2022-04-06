import argparse
import os
from dataset_train import get_loader
from solver_train import Solver


def main(config):
    if config.mode == 'train':
        train_loader, dataset = get_loader(config.batch_size, num_thread=config.num_thread)
        run = "mtcnet"
        if not os.path.exists("%s/run-%s" % (config.save_fold, run)):
            os.mkdir("%s/run-%s" % (config.save_fold, run))
            os.mkdir("%s/run-%s/logs_mtcnet" % (config.save_fold, run))
            os.mkdir("%s/run-%s/models_mtcnet" % (config.save_fold, run))
        config.save_fold = "%s/run-%s" % (config.save_fold, run)
        train = Solver(train_loader, None, config)
        train.train()
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':

    resnet_path = './weights/resnet50_caffe.pth'
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--resnet', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=30)  # 12, now x3
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--load_bone', type=str, default='')
    # parser.add_argument('--load_branch', type=str, default='')
    parser.add_argument('--save_fold', type=str, default='./EGNet')
    # parser.add_argument('--epoch_val', type=int, default=20)
    parser.add_argument('--epoch_save', type=int, default=1)  # 2, now x3
    parser.add_argument('--epoch_show', type=int, default=1)
    parser.add_argument('--pre_trained', type=str, default=None)

    # Testing settings
    parser.add_argument('--model', type=str, default='./MTCNet/models.pth')
    parser.add_argument('--sal_mode', type=str, default='t')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--visdom', type=bool, default=False)

    config = parser.parse_args()

    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)
