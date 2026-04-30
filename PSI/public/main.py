import random

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import shutil
from config_utils import load_config
import datetime
from draw import visualize
from scheduler_multi_kclub import SchedulerMultiKCLUB
from scheduler import Scheduler
from asy_scheduler import AsyScheduler
from models import (
    ResNet18Client, ResNet18Server,
    ResNet34Client, ResNet34Server,
    ResNet50Client, ResNet50Server,
    VGG16Client, VGG16Server,
    AlexNetClient, AlexNetServer,
    PruningNetwork, loss_fn
)
from adv_models import AdversaryModelGen, AdversaryModelPred, reconstruction_loss
from utils import copy_source_code
import model_utils
import config_utils
import dataset_utils
import model_utils as m
from auxiliary_nets import Decoder, AuxClassifier
from torch.autograd import Variable
from torchsummary import summary


def seed_everything(seed=42):
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def update_args(config, new_args):
    config.update(new_args)
    server_logits, adv_logits = dataset_utils.infer_data_info(new_args.get('dataset_path'), new_args.get("dataset"), config.get("protected_attribute"), config.get("prediction_attribute"))
    config["server_logits"] = server_logits
    config["adversary_logits"] = adv_logits
    config_utils.update_config(config=config,new_variables=new_args)


def run_experiment(auto=None, new_args=None):

    config = load_config()
    if new_args:
        update_args(config, new_args)
        print("updated config:", config)


    seed = config.get("seed", 42)
    experiment_type = config.get("experiment_type", "challenge")
    gpu_devices = config.get("gpu_devices")
    pruning_ratio = config.get("pruning_ratio", 0.64)
    split_layer = config.get("split_layer", 4)
    lr = config.get("learning_rate", 0.01)
    model_name = config.get("model_name", "resnet18")
    pruning_style = config.get("pruning_style", "nopruning")
    grid_crop = config.get("grid_crop", False)
    results_path = config.get("results_path")
    protected_attribute = config.get("protected_attribute")
    if len(protected_attribute) < 2:
        protected_attribute = protected_attribute[0]
        multi_attr = False
    else:
        multi_attr = True
    config["multi_attr"] = multi_attr
    config["protected_attribute"] = protected_attribute
    adversary_logits = config.get("adversary_logits", 10)
    server_logits = config.get("server_logits", 2)
    client_pretrained = config.get("client_pretrained", False)
    adversary_pretrained = config.get("adversary_pretrained", False)
    server_pretrained = config.get("server_pretrained", False)
    pruner_pretrained = config.get("pruner_pretrained", False)
    adv_gen_method = config.get("adv_gen_method", "conv_transpose")
    adv_pred_method = config.get("adv_pred_method", "resnet18")

    gpu_id = gpu_devices[0]
    gpu_available = False

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpu_id))
        gpu_available = True
    else:
        device = torch.device('cpu')

    seed_everything(seed)

    client_hparams = {"pretrained": client_pretrained, "split_layer": split_layer, "grid_crop": grid_crop}
    server_hparams = {"logits": server_logits, "pretrained": server_pretrained, "split_layer": split_layer}
    if multi_attr:
        adversary_hparams_1 = {"logits": adversary_logits[0], "pretrained": adversary_pretrained, "split_layer": split_layer,
                            "model_name": model_name, "adv_gen_method": adv_gen_method, "adv_pred_method": adv_pred_method}
        adversary_hparams_2 = {"logits": adversary_logits[1], "pretrained": adversary_pretrained, "split_layer": split_layer,
                            "model_name": model_name, "adv_gen_method": adv_gen_method, "adv_pred_method": adv_pred_method}
    else:
        adversary_hparams = {"logits": adversary_logits, "pretrained": adversary_pretrained, "split_layer": split_layer,
                            "model_name": model_name, "adv_gen_method": adv_gen_method, "adv_pred_method": adv_pred_method}
    pruner_hparams = {"pretrained": pruner_pretrained, "split_layer": split_layer, "model_name": model_name,
                      "pruning_ratio": pruning_ratio, "pruning_style": pruning_style}

    if experiment_type == "challenge":
        experiment_path = config.get("experiment_path", "./experiments/")
        challenge_path = results_path + "/challenge/"
        if os.path.isdir(challenge_path):
            if auto:  # This it to not duplicate work already done and to continue running experiments
                return None
            print("Challenge for experiment {} already present".format(config.get("experiment_name")))
            inp = input("Press e to exit, r to replace it: ")
            if inp == "e":
                exit()
            elif inp == "r":
                shutil.rmtree(results_path + "/challenge/")
            else:
                print("Input not understood")
                exit()

    else:
        experiment_path = None
        print("exp path:", results_path)
        print("name:", results_path)
        # if os.path.isdir(results_path):
        #     if auto:  # This it to not duplicate work already done and to continue running experiments
        #         print("silently skipping experiment", config.get('experiment_name'))
        #         return None
        #     print("Experiment {} already present".format(config.get("experiment_name")))
        #     inp = input("Press e to exit, r to replace it, c to continue training: ")
        #     if inp == "e":
        #         exit()
        #     elif inp == "r":
        #         print(111)
        #         shutil.rmtree(results_path)
        #         print(222)
        #     elif inp == "c":
        #         experiment_path = results_path
        #     else:
        #         print("Input not understood")
        #         exit()
        if not experiment_path:
            os.makedirs(config.get('model_path'), exist_ok=True)
            os.makedirs(config.get('log_path'), exist_ok=True)


    if experiment_path:
        try:
            server_model, client_model, pruner, adversary_model = model_utils.load_models(experiment_path)
        except:
            server_model, client_model, pruner, adversary_model = model_utils.load_models(experiment_path+ "/")
    else:
        if model_name == "resnet18":
            client_model = ResNet18Client(client_hparams).to(device)
            server_model = ResNet18Server(server_hparams).to(device)
        elif model_name == "resnet34":
            client_model = ResNet34Client(client_hparams).to(device)
            server_model = ResNet34Server(server_hparams).to(device)
        elif model_name == "resnet50":
            client_model = ResNet50Client(client_hparams).to(device)
            server_model = ResNet50Server(server_hparams).to(device)
        elif model_name == "vgg16":
            client_model = VGG16Client(client_hparams).to(device)
            server_model = VGG16Server(server_hparams).to(device)
        elif model_name == "alexnet":
            client_model = AlexNetClient(client_hparams).to(device)
            server_model = AlexNetServer(server_hparams).to(device)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}. "
                             f"Please check your --model_name parameter.")

    print(client_model)
    print(server_model)

    client_channels = client_model(torch.rand(1, 3, config.get('img_size'), config.get('img_size')).to(device)).shape[1]
    if protected_attribute == "data":
        adversary_hparams.update({"channels": client_channels})
        if not experiment_path:
            adversary_model = AdversaryModelGen(adversary_hparams).to(device)
        privacy_loss_fn = reconstruction_loss

    else:
        if not experiment_path:
            if multi_attr:
                adversary_model_1 = AdversaryModelPred(adversary_hparams_1).to(device)
                adversary_model_2 = AdversaryModelPred(adversary_hparams_2).to(device)
            else:
                adversary_model = AdversaryModelPred(adversary_hparams).to(device)
        privacy_loss_fn = loss_fn

    pruner_hparams.update({"logits": client_channels})
    if not experiment_path:
        pruner = PruningNetwork(pruner_hparams).to(device)

    prediction_loss_fn = loss_fn

    decoder = Decoder(64, config["img_size"], widen=1).to(device)

    if config.get("optim_type") == 'adam':
        client_optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
        server_optimizer = optim.Adam(server_model.parameters(), lr=lr)
        if multi_attr:
            adversary_optimizer_1 = optim.Adam(adversary_model_1.parameters(), lr=lr)
            adversary_optimizer_2 = optim.Adam(adversary_model_2.parameters(), lr=lr)
        else:
            adversary_optimizer = optim.Adam(adversary_model.parameters(), lr=lr)
        pruner_optimizer = optim.Adam(pruner.parameters(), lr=lr)
    else:
        client_optimizer = torch.optim.SGD(client_model.parameters(), lr=lr)
        decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=lr)
        server_optimizer = optim.SGD(server_model.parameters(), lr=lr)
        if multi_attr:
            adversary_optimizer_1 = optim.SGD(adversary_model_1.parameters(), lr=lr)
            adversary_optimizer_2 = optim.SGD(adversary_model_2.parameters(), lr=lr)
        else:
            adversary_optimizer = optim.SGD(adversary_model.parameters(), lr=lr)
        pruner_optimizer = optim.SGD(pruner.parameters(), lr=lr)

    if multi_attr:
        objects = {"client_model": client_model,
                   "server_model": server_model,
                   "adversary_model_1": adversary_model_1,
                   "adversary_model_2": adversary_model_2,
                   "pruner": pruner,
                   "decoder": decoder,
                   "client_optim": client_optimizer,
                   "decoder_optim": decoder_optimizer,
                   "server_optim": server_optimizer,
                   "adversary_optim_1": adversary_optimizer_1,
                   "adversary_optim_2": adversary_optimizer_2,
                   "pruner_optim": pruner_optimizer,
                   "prediction_loss_fn": prediction_loss_fn,
                   "privacy_loss_fn": privacy_loss_fn,
                   "device": device}
    else:
        objects = {"client_model": client_model,
                   "server_model": server_model,
                   "adversary_model": adversary_model,
                   "pruner": pruner,
                   "decoder":decoder,
                   "client_optim": client_optimizer,
                   "decoder_optim": decoder_optimizer,
                   "server_optim": server_optimizer,
                   "adversary_optim": adversary_optimizer,
                   "pruner_optim": pruner_optimizer,
                   "prediction_loss_fn": prediction_loss_fn,
                   "privacy_loss_fn": privacy_loss_fn,
                   "device": device}

    last_epoch = None
    if experiment_path:
        last_epoch = m.load_last_epoch(experiment_path)

    print("last epoch:", last_epoch)

    if config['is_async']:
        scheduler = AsyScheduler(config, objects, last_epoch)
    elif isinstance(protected_attribute, list):
        scheduler = SchedulerMultiKCLUB(config, objects, last_epoch)
    else:
        scheduler = Scheduler(config, objects, last_epoch)

    
    if experiment_type == "training":
        print(888)
        # call the training loop
        num_times_to_run = config.get("total_epochs", 0)
        if last_epoch:
            num_times_to_run = num_times_to_run - abs(last_epoch)
            if num_times_to_run < 0:
                num_times_to_run = 0
        print("times to run:", num_times_to_run)
        epoch_time_list = []
        train_acc_list = []
        train_loss_list = []
        test_acc_list = []
        test_loss_list = []
        training_time_list = []
        inference_time_list = []
        for epoch in range(num_times_to_run):
            start = time.time()
            scheduler.train()
            end = time.time()
            epoch_time = end - start
            scheduler.test()
            epoch_time_list.append(epoch_time)
            if config['is_server']:
                train_acc_list.append(scheduler.train_acc)
                train_loss_list.append(scheduler.train_loss)
                test_acc_list.append(scheduler.test_acc)
                test_loss_list.append(scheduler.test_loss.cpu().numpy())
                training_time_list = epoch_time_list
                inference_time_list.append(sum(scheduler.inference_list))
        training_experiment_name = 'PSI'
        training_time = sum(epoch_time_list)
        dataset = config['dataset']
        experiment_style = 'PSI'
        is_async = config['is_async']
        sensitive = config['protected_attribute']
        epoch = config['total_epochs']
        ixx_r, ixy_r, ixz_r = config["ixx_r"], config["ixy_r"], config["ixz_r"]
        print("total training time: ", sum(epoch_time_list))
        print("avg epoch inference time", sum(inference_time_list)/epoch)
        if config['is_server']:
            visualize(train_acc_list,train_loss_list,test_acc_list,test_loss_list,training_time_list,inference_time_list,training_experiment_name,
                  training_time,dataset,experiment_style,is_async,sensitive,epoch,ixx_r,ixy_r,ixz_r)
        scheduler.generate_challenge()
    elif experiment_type == "challenge":
        print("Creating challenge")
        scheduler.generate_challenge()
    else:
        print("Unknown experiment type")


if __name__ == '__main__':
    run_experiment()
