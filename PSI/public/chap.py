import argparse
from main import run_experiment

parser = argparse.ArgumentParser(description='Process inputs for chap project.')

# Necessary inputs unless more advanced uses are being done

parser.add_argument('--dataset', type=str, default = 'fairface', help="name of the dataset to train or use for classification. Ex. Fairface (must follow the guidelines for the organization in order to be used)")
parser.add_argument('--split_layer', type=int, default=6, help="number representing where to split the model to be used for chap.")
parser.add_argument('--dataset_path', type=str, default ='../../dataset',help="path on device where dataset resides. Ex. /u/user/")
parser.add_argument('--prediction_attribute', type=str, default="gender", help="name of label attribute to prediction") # gender animated living
parser.add_argument('--protected_attribute', nargs='+', type=str, default=["race"], help="name of label attributes to protect (can pass multiple)")
parser.add_argument('--model_name', type=str, default="resnet18", help="name of model to use") # resnet34, resnet50
parser.add_argument('--adv_pred_method', type=str, default="ml-leaks", help="method for adversarial prediction") # ml-leaks
parser.add_argument('--adv_gen_method', type=str, default="conv-transpose", help="method for generating adversarial examples") # attention

# check logic for experiment paths (new, etc. for full functionality)
parser.add_argument('--experiment_path', type=str, default=None, help="path to store and access experiements.")

# Update help
parser.add_argument('--pruning_style', type=str, default="k_club", help="style of pruning for data obsufication. Supported styles: 'network', 'nopruning', 'adversarial', 'maxentropy', 'noise'")
parser.add_argument('--grid_crop', default=False, action="store_true", help="whether to use a grid crop")
# More advanced parameters
parser.add_argument('--batch_size', type=int, default=64, help="number of samples per training / testing batch.")
parser.add_argument('--pruning_ratio', type=float, default=0.64, help="number representing the pruning ratio for the chap experiment.")
parser.add_argument('--total_epochs', type=int, default=100, help="number representing the amount of epochs desired for training.")
parser.add_argument('--learning_rate', type=float, default=0.01, help="learning rate used to train model chap models. Ex. 0.01")
parser.add_argument('--dir_of_inputs', type=str, default=None, help="Directory of inputs to be classified.")
parser.add_argument('--is_server', type=int, default=1, help="Is if server")
parser.add_argument('--is_async', type=int, default=0, help="Is if async")
parser.add_argument("--seed", default=42, type=int, help="random seed")
parser.add_argument('--server_ip', type=str, default='127.0.0.1', help="IP address for async socket communication")
parser.add_argument('--server_port', type=int, default=10080, help="Port for async socket communication")
parser.add_argument('--lambda_vib', type=float, default=0.0002, help="lambda for VIB")
parser.add_argument('--lambda_hsic', type=float, default=500, help="lambda for HSIC")
parser.add_argument('--lambda_club', type=float, default=3.0, help="lambda for CLUB")

parser.add_argument('--gpu_devices', type=int, nargs='+', default=[0], help="List of GPU IDs to use. Ex. --gpu_devices 1")

if __name__ == '__main__':
    args = parser.parse_args()
    new_args = args.__dict__
    run_experiment(False, new_args)
