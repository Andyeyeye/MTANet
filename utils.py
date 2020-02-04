import logging
import os
from tensorboardX import SummaryWriter
import datetime


def save_log(prefix, output_dir, date_str):
    fmt = '%(asctime)s.%(msecs)03d %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    filename = os.path.join(output_dir, prefix + '_' + date_str + '.log')
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=date_fmt,
                        filename=filename, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def prep_experiment(args):
    """
    Make output directories, setup logging, Tensorboard, snapshot code.
    """
    ckpt_path = args.ckpt_path
    tb_path = args.tb_path
    args.exp_path = os.path.join(ckpt_path, args.exp)
    args.tb_exp_path = os.path.join(tb_path, args.exp)
    args.date_str = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(args.exp_path, exist_ok=True)
    os.makedirs(args.tb_path, exist_ok=True)
    save_log('log', args.exp_path, args.date_str)
    open(os.path.join(args.exp_path, args.date_str + '.txt'), 'w').write(
        str(args) + '\n\n')
    writer = SummaryWriter(logdir=args.tb_exp_path, comment=args.tb_tag)
    return writer


def print_eval(args):
    logging.info('-' * 107)
    fmt_str = 'best record:@[epoch: %d] [val loss %.5f], [best acc1 %.5f], [best va %.5f], [best au f1 %.5f], ' + \
              '[best au strict %.5f], [best expr f1 %.5f], '
    logging.info(fmt_str % (args.best_record['epoch'], args.best_record['val_loss'], args.best_record['best_acc1'],
                            args.best_record['best_va'], args.best_record['best_au_f1'],
                            args.best_record['best_au_strict'], args.best_record['best_expr_f1']))
    logging.info('-' * 107)
