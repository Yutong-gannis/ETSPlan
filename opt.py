import time
from argparse import ArgumentParser


def get_opts():
    parser = ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="D:\ETSMotion\ETSMotion_test")
    parser.add_argument('--batch_size', type=int, nargs='+',default=[1, 1, 1, 1], help='batch size for each stage')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, nargs='+',default=[1, 1, 1, 1], help='num workers for each stage')
    parser.add_argument('--epochs', type=int, nargs='+',default=[1, 1, 1, 1], help='num epoch for each stage')
    parser.add_argument('--log_per_n_step', type=int, default=500)
    parser.add_argument('--val_per_n_epoch', type=int, default=1)
    parser.add_argument('--device', type=str, default="cpu")

    parser.add_argument('--resume', type=str, default='D:\ETSPlan\epoch_6.pth')
    parser.add_argument('--load_encoder', type=bool, default=False)

    parser.add_argument('--num_cls', type=int, default=5)
    parser.add_argument('--num_pts', type=int, default=8)
    parser.add_argument('--mtp_alpha', type=float, default=1.0)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--optimize_per_n_step', type=int, default=40)
    
    exp_name = str(time.time())
    parser.add_argument('--exp_name', type=str, default=exp_name)

    args = parser.parse_args()
    return args