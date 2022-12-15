# =========================================================
# @ Main File for PolyU Project: Online Contactless Palmprint
#   Identification using Deep Learning
# =========================================================
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import shutil
import scipy.misc
import datetime
from models.model import Model
from models.vgg16_model import Model as VGGModel
from torch.utils.tensorboard import SummaryWriter


def build_parser():
    parser = argparse.ArgumentParser()

    # Checkpoint Options
    parser.add_argument('--checkpoint_dir', type=str,
                        dest='checkpoint_dir', default='./checkpoint/Joint-Finger-RFNet/')
    parser.add_argument('--db_prefix', dest='db_prefix', default='MaskLM')
    parser.add_argument('--checkpoint_interval', type=int, dest='checkpoint_interval', default=20)

    # Dataset Options
    parser.add_argument('--train_path', type=str, dest='train_path',
                        default='/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/mask-seg/03/')
    parser.add_argument('--feature_path', type=str, dest='feature_path',
                        default='/media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnukcleDatabase/Finger-knuckle/feature/03/')
    parser.add_argument('--conf_path', type=str, dest='conf_path',
                        default='media/zhenyuzhou/Data/finger_knuckle_2018/FingerKnuckleDatabase/Finger-knuckle/conf/03/')
    parser.add_argument('--samples_subject', type=int, dest='samples_subject',
                        default=5)
    parser.add_argument('--n_tuple', type=str, dest='n_tuple',
                        default='quadruplet', help="how to select the input tuple, triplet, quadruplet, feature")
    # Model
    parser.add_argument('--model', type=str, dest='model', default="VGG16")
    parser.add_argument('--loss_type', type=str, dest="loss_type", default="ssim")
    parser.add_argument('--if_augment', type=bool, dest="if_augment", default=False)

    # Training StrategyResidualSTNet
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=4)
    parser.add_argument('--epochs', type=int, dest='epochs', default=3000)
    parser.add_argument('--learning_rate1', type=float, dest='learning_rate1', default=1e-3)
    parser.add_argument('--learning_rate2', type=float, dest='learning_rate2', default=1e-5)

    # Training Logging Interval
    parser.add_argument('--log_interval', type=int, dest='log_interval', default=1)
    # Pre-defined Options
    parser.add_argument('--alpha', type=float, dest='alpha', default=0.5)
    parser.add_argument('--alpha2', type=float, dest='alpha2', default=0.3, help="the second margin of quadruplet loss")
    parser.add_argument('--input_size', type=int, dest='input_size', default=(128, 128), help="(w, h)")
    parser.add_argument('--horizontal_size', type=int, dest='horizontal_size', default=0)
    parser.add_argument('--vertical_size', type=int, dest='vertical_size', default=0)
    parser.add_argument('--block_size', type=int, dest="block_size", default=8)
    parser.add_argument('--rotate_angle', type=int, dest="rotate_angle", default=0)
    parser.add_argument('--freeze_stn', type=bool, dest="freeze_stn", default=True)
    parser.add_argument('--freeze_thre', type=float, dest="freeze_thre", default=0)
    parser.add_argument('--sglue_conf', type=dict, dest="sglue_conf", default={
        'GNN_layers': ['self', 'cross'] * 1,
        'weight': ''})
    parser.add_argument('--sinkhorn_it', type=int, dest="sinkhorn_it", default=200)
    parser.add_argument('--matching_type', type=str, dest="matching_type", default='superglue')

    # fine-tuning
    parser.add_argument('--start_ckpt', type=str, dest='start_ckpt', default="")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    this_datetime = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    args.checkpoint_dir = os.path.join(
        args.checkpoint_dir,
        "{}_{}_{}_{}-r{}-a{}-2a{}-hs{}_vs{}_{}".format(
            args.db_prefix,
            args.model,
            args.n_tuple,
            args.loss_type,
            int(args.rotate_angle),
            float(args.alpha),
            float(args.alpha2),
            int(args.horizontal_size),
            int(args.vertical_size),
            this_datetime
        )
    )

    logdir = os.path.join(args.checkpoint_dir, 'runs')

    print("[*] Target Checkpoint Path: {}".format(args.checkpoint_dir))
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    print("[*] Target Logdir Path: {}".format(logdir))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    hyper_parameter = os.path.join(args.checkpoint_dir, 'hyper_parameter.txt')
    with open(hyper_parameter, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))

    writer = SummaryWriter(log_dir=logdir)
    if args.model == "VGG16":
        model_ = VGGModel(args, writer=writer)
        model_.quadruplet_train()
    else:
        model_ = Model(args, writer=writer)
        if args.n_tuple == "feature":
            model_.fusion_triplet_train(args)
            # model_.assistant_triplet_train(args)
        elif args.n_tuple == "quadruplet":
            model_.quadruplet_loss(args)
        else:
            model_.triplet_train(args)


if __name__ == "__main__":
    main()
