import argparse
import time

arg_lists = []
parser = argparse.ArgumentParser(description='Retinanet Network')

def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--model_date', type=str, default=time.strftime('%d-%m-%Y-%H-%M-%S'),
                      help='Model date used for unique checkpointing')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--data_dir', type=str, default='./data/changed/',
                      help='Directory in which data is stored')
misc_arg.add_argument('--save_model', type=bool, default=True,
                      help='Whether to save the model')                      


train_arg = add_argument_group('Training')
train_arg.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
train_arg.add_argument('--csv_train', help='Path to file containing training annotations (optional, see readme)')
train_arg.add_argument('--csv_val', help='Path to file containing training annotations (optional, see readme)')
train_arg.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=18)
train_arg.add_argument('--epochs', help='Number of epochs', type=int, default=20)

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
