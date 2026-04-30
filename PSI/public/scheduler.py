import random

import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from dataset_utils import Custom

import numpy as np
import os
from utils import LoggerUtils
import torch.nn as nn
from collections import OrderedDict

import time
from losses import HSICLoss, CLUBLoss, EntropyLoss, KernelGuidedCLUBLoss, KLLoss
from thop import profile
import sys
#sys.path.append('./models')
#from dataloader import CIFAR10DataLoader

class Scheduler():
    """docstring for Scheduler"""

    def __init__(self, config, objects, epoch=None):
        super(Scheduler, self).__init__()
        self.config = config
        self.objects = objects
        self.epoch = epoch + 1 if epoch else 0
        self.initialize()


    def initialize(self):
        gpu_devices = self.config.get("gpu_devices")
        gpu_id = gpu_devices[0]
        gpu_available = False

        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(gpu_id))
            gpu_available = True
        else:
            device = torch.device('cpu')
        self.setup_logger()
        self.setup_data_pipeline()
        self.setup_training_params()
        self.mask_train_mean = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
            self.config['train_batch_size'] , 3, self.config["img_size"], self.config["img_size"]
        ).to(device)
        self.mask_train_std = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
            self.config['train_batch_size'], 3, self.config["img_size"], self.config["img_size"]
        ).to(device)

    def _image_restore(self, normalized_image):
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) \
               + self.mask_train_mean[:normalized_image.size(0)]

    def setup_logger(self):
        # if self.config["experiment_type"] == "challenge":
        #    return
        log_config = {"log_path": self.config["log_path"]}
        self.logger = LoggerUtils(log_config)

    def get_split(self, dataset):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.config["train_split"] * dataset_size))
        np.random.shuffle(indices)

        train_indices, test_indices = indices[:split], indices[split:]
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        return train_dataset, test_dataset

    def setup_data_pipeline(self):
        self.IM_SIZE = self.config["img_size"]
        trainTransform = transforms.Compose([
            transforms.Resize((self.IM_SIZE, self.IM_SIZE)),
            transforms.ToTensor()])

        train_config = {"transforms": trainTransform,
                        "train": True,
                        "path": self.config["dataset_path"],
                        "multi_attr": self.config["multi_attr"],
                        "prediction_attribute": self.config["prediction_attribute"],
                        "protected_attribute": self.config["protected_attribute"]}
        test_config = {"transforms": trainTransform,
                        "train": False,
                        "path": self.config["dataset_path"],
                        "multi_attr": self.config["multi_attr"],
                        "prediction_attribute": self.config["prediction_attribute"],
                        "protected_attribute": self.config["protected_attribute"]}

        '''
        embed_length = 128
        data = CIFAR10DataLoader(embed_length=embed_length)
        print('==> Building models..')
        data.load()  # <--This method loads all the parameters for the data including neural-net models and optimizer

        ## Initialize the data loaders
        self.trainloader = torch.utils.data.DataLoader(data.trainset,
                                                  batch_size=data.train_batch_size,
                                                  shuffle=True,
                                                  num_workers=40
                                                  )
        self.testloader = torch.utils.data.DataLoader(data.testset,
                                                 batch_size=data.test_batch_size,
                                                 shuffle=False,
                                                 num_workers=40)

        '''
        train_config["format"] = "jpg"
        test_config["format"] = "jpg"
        train_dataset = Custom(train_config, self.config.get("dataset"))
        test_dataset = Custom(test_config, self.config.get("dataset"))

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(self.config["seed"])

        if self.config["split"] is True:
            train_dataset, test_dataset = self.get_split(dataset)

        self.trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["train_batch_size"],
            shuffle=True, num_workers=5, drop_last=True,
            worker_init_fn=seed_worker, generator=g)

        self.testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config["test_batch_size"],
            shuffle=False, num_workers=5, drop_last=True,
            worker_init_fn=seed_worker, generator=g)

    def log_and_save_model(self):
        self.logger.log_model_stats(self.client_model)
        self.logger.log_model_stats(self.server_model)
        self.logger.log_model_stats(self.adversary_model)
        self.logger.log_model_stats(self.decoder)
        self.logger.log_model_stats(self.pruner)
        self.logger.save_model(self.client_model, self.client_model_path)
        self.logger.save_model(self.server_model, self.server_model_path)
        self.logger.save_model(self.adversary_model, self.adversary_model_path)
        self.logger.save_model(self.decoder, self.decoder_model_path)
        self.logger.save_model(self.pruner, self.pruner_model_path)

    def setup_training_params(self):
        # self.epoch = 0
        self.load_models()
        if self.config["experiment_type"] == "training":
            self.load_loss_fns()
            self.load_optim()
        self.device = self.objects["device"]

        self.client_model_path = self.config["model_path"] + "client_model.pt"
        self.adversary_model_path = self.config["model_path"] + "adversary_model.pt"
        self.server_model_path = self.config["model_path"] + "server_model.pt"
        self.decoder_model_path = self.config["model_path"] + "decoder_model.pt"
        self.pruner_model_path = self.config["model_path"] + "pruner_model.pt"
        self.challenge_dataset_path = self.config["results_path"] + "challenge/"

        self.lambda1 = self.config["lambda1"]
        self.lambda2 = self.config["lambda2"]

    def load_optim(self):
        self.server_optim = self.objects["server_optim"]
        self.adversary_optim = self.objects["adversary_optim"]
        self.client_optim = self.objects["client_optim"]
        self.decoder_optim = self.objects["decoder_optim"]
        self.pruner_optim = self.objects["pruner_optim"]

    def load_loss_fns(self):
        self.prediction_loss_fn = self.objects["prediction_loss_fn"]
        self.privacy_loss_fn = self.objects["privacy_loss_fn"]
        if self.config["pruning_style"] == "maxentropy":
            self.entropy_loss_fn = EntropyLoss()
        if "hsic" in self.config["pruning_style"]:
            self.hsic_loss_fn = HSICLoss()
        if "vib" in self.config["pruning_style"]:
            self.kl_loss_fn = KLLoss()
        if "club" in self.config["pruning_style"] :
            if self.config["pruning_style"] == "k_club":
                is_cont = (self.config["protected_attribute"] == "data")
                self.hsic_loss_fn = HSICLoss()
                self.club_loss_fn = KernelGuidedCLUBLoss(is_continuous=is_cont)
            else:
                self.club_loss_fn = CLUBLoss()

    def make_multi_gpu_if_available(self, model):
        if len(self.config["gpu_devices"]) > 1:
            return nn.DataParallel(model, device_ids = self.config["gpu_devices"])
        return model

    def load_models(self):
        self.client_model = self.make_multi_gpu_if_available(self.objects["client_model"])
        self.server_model = self.make_multi_gpu_if_available(self.objects["server_model"])
        self.adversary_model = self.make_multi_gpu_if_available(self.objects["adversary_model"])
        self.decoder = self.make_multi_gpu_if_available(self.objects["decoder"])
        self.pruner = self.make_multi_gpu_if_available(self.objects["pruner"])
        self.logger.log_console("models have been loaded")

    def clear_optim_grad(self):
        self.server_optim.zero_grad()
        self.adversary_optim.zero_grad()
        self.client_optim.zero_grad()
        self.pruner_optim.zero_grad()

    def eval_mode(self):
        self.client_model.eval()
        self.server_model.eval()
        self.pruner.eval()
        self.adversary_model.eval()
        self.decoder.eval()

    def train_mode(self):
        self.client_model.train()
        self.server_model.train()
        self.pruner.train()
        self.adversary_model.train()
        self.decoder.train()

    def cleanse_state_dict(self, state_dict):
        """
        This is an mismatch of expected keys, etc. Issua comes up is saved via gpu, but running on cpu, etc.
        Ex. mismatch: keys not matching
        Expecting: {"model.0.weight", ...}
        Received: {"module.model.0.weight", ...}
        """
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        return new_state_dict

    def load_saved_models(self):
        # load client model
        try:
            self.objects["client_model"].load_state_dict(torch.load(self.client_model_path, map_location=self.device))
            # load pruner model
            self.objects["pruner"].load_state_dict(torch.load(self.pruner_model_path, map_location=self.device))
            self.objects["adversary_model"].load_state_dict(torch.load(self.adversary_model_path, map_location=self.device))
            self.objects["decoder"].load_state_dict(torch.load(self.decoder_model_path, map_location=self.device))
        except:
            client_state_dict = self.cleanse_state_dict(torch.load(self.client_model_path, map_location=self.device))
            pruner_state_dict = self.cleanse_state_dict(torch.load(self.pruner_model_path, map_location=self.device))
            self.objects["client_model"].load_state_dict(client_state_dict)
            # load pruner model
            self.objects["pruner"].load_state_dict(pruner_state_dict)
            # load adversary model
            adversary_state_dict = self.cleanse_state_dict(torch.load(self.adversary_model_path, map_location=self.device))
            self.objects["adversary_model"].load_state_dict(adversary_state_dict)
            decoder_state_dict = self.cleanse_state_dict(torch.load(self.decoder_model_path, map_location=self.device))
            self.objects["decoder"].load_state_dict(decoder_state_dict)
        self.load_models()

    def save_challenge_dataset(self, z_hat, filename):
        for ele in range(int(z_hat.shape[0])):
            z_hat_path = self.challenge_dataset_path + filename[ele] + '.pt'
            torch.save(z_hat[ele].detach().cpu(), z_hat_path)

    def save_reconstruction_image(self, rec_img, filename):
        for ele in range(int(rec_img.shape[0])):
            img_path = self.challenge_dataset_path + filename[ele] + '_rec.jpg'
            save_image(rec_img[ele], img_path)

    def generate_challenge(self):
        self.load_saved_models()
        os.mkdir(self.challenge_dataset_path)
        self.labels_tensor = []
        self.counter = 0
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.testloader):
                data = Variable(sample["img"]).to(self.device)
                filename = [name.split("/")[-1].split('.')[0] for name in sample["filename"]]

                z = self.client_model(data)
                z_hat, indices = self.pruner(z)
                # z_hat = z

                #print(batch_idx)

                self.save_challenge_dataset(z_hat, filename)
                if self.config['protected_attribute'] == 'data':
                    rec_img = self.adversary_model(z_hat)
                    self.save_reconstruction_image(rec_img, filename)

    def test(self):
        self.eval_mode()
        task_loss, priv_loss, task_pred_correct, privacy_pred_correct = 0, 0, 0, 0
        task_total, priv_total = 0, 0
        total = 0
        # all_predictions, all_labels = np.array([]), np.array([])

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        with torch.no_grad():
            self.inference_list = []
            for batch_idx, sample in enumerate(self.testloader):
                data = Variable(sample["img"]).to(self.device)
                prediction_labels = Variable(sample["prediction_label"]).to(self.device)
                protected_labels = Variable(sample["private_label"]).to(self.device)

                start = time.time()
                z = self.client_model(data)
                #flops, params = profile(self.client_model, inputs=(data,))
                #print("inference client flops:{}, params:{}",flops,params)
                z_hat, indices = self.pruner(z)

                task_attribute_logits = self.server_model(z_hat)
                #flops2, params2 = profile(self.server_model, inputs=(z_hat,))
                #print("inference server flops:{}, params:{}", flops2, params2)
                end = time.time()
                infer_time = end - start
                self.inference_list.append(infer_time)
                protected_attribute_logits = self.adversary_model(z_hat)

                # all_predictions = np.append(all_predictions, [prediction.detach().cpu().numpy()[:, 1]])
                # all_labels = np.append(all_labels, [labels.detach().cpu().numpy()])

                prediction_loss = self.prediction_loss_fn(
                                        task_attribute_logits,
                                        prediction_labels
                                        )
                privacy_loss = self.privacy_loss_fn(
                                    protected_attribute_logits,
                                    protected_labels
                                    )
                task_loss += prediction_loss
                priv_loss += privacy_loss

                task_pred_correct += (task_attribute_logits.argmax(dim=1) ==
                                      prediction_labels).sum().item()
                if self.config["protected_attribute"] != "data":
                   privacy_pred_correct += (protected_attribute_logits.argmax(dim=1) ==
                                        protected_labels).sum().item()
                total += int(data.shape[0])
                task_total += prediction_labels.sum()
                priv_total += protected_labels.sum()

        task_loss /= total
        priv_loss /= total
        task_pred_acc = task_pred_correct / total
        privacy_pred_acc = privacy_pred_correct / total

        # Depracted error

        # print("task ratio: ", task_total / total)
        # print("priv ratio: ", priv_total / total)

        # Metrics logging is used for F1, recall and other score
        # self.logger.log_metrics("test/", all_predictions, all_labels, self.epoch)

        test_latency = sum(self.inference_list)
        peak_mem_mb = 0
        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        self.test_acc = task_pred_acc
        self.test_loss = task_loss
        self.logger.log_scalar("test/task_loss", task_loss.item(), self.epoch)
        self.logger.log_scalar("test/privacy_loss", privacy_loss.item(), self.epoch)
        self.logger.log_scalar("test/task_pred_accuracy", task_pred_acc, self.epoch)
        self.logger.log_scalar("test/privacy_accuracy", privacy_pred_acc, self.epoch)

        self.logger.log_console(
            f"[Test] Epoch {self.epoch} | Latency: {test_latency:.3f} s | "
            f"Peak VRAM: {peak_mem_mb:.2f} MB | Utility(Acc): {task_pred_acc:.4f} | Privacy(Acc): {privacy_pred_acc:.4f}"
        )

    def train(self):
        train_loss = 0
        task_pred_correct_total = 0
        private_pred_correct_total = 0
        privacy_loss_total = 0
        total = 0
        self.train_mode()

        epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        for batch_idx, sample in enumerate(self.trainloader):
            step_num = len(self.trainloader) * self.epoch + batch_idx
            data = Variable(sample["img"]).to(self.device)
            prediction_labels = Variable(sample["prediction_label"]).to(self.device)
            if self.config["protected_attribute"] != "data":
                protected_labels = Variable(sample["private_label"]).to(self.device)
            else:
                protected_labels = data.clone()

            self.clear_optim_grad()
            z = self.client_model(data)

            pruner_input = Variable(z.detach().clone(), requires_grad=True)
            z_hat, indices = self.pruner(pruner_input)

            server_input = z_hat.detach().clone().requires_grad_(True)
            task_attribute_logits = self.server_model(server_input)

            # adversary_input = Variable(z_hat.clone(), requires_grad=True)
            adversary_input = z_hat.detach().clone().requires_grad_(True)
            protected_attribute_logits = self.adversary_model(adversary_input)

            # self.logger.log_metrics("train/", prediction, labels)

            prediction_loss = self.prediction_loss_fn(
                task_attribute_logits,
                prediction_labels
            )
            privacy_loss = self.privacy_loss_fn(
                protected_attribute_logits,
                protected_labels
            )
            final_loss = self.lambda1 * prediction_loss + self.lambda2 * privacy_loss

            task_pred_correct = (task_attribute_logits.argmax(dim=1) == prediction_labels).sum().item()
            if self.config["protected_attribute"] != "data":
                private_pred_correct = (protected_attribute_logits.argmax(dim=1) == protected_labels).sum().item()

            # Backprop begins ...........
            prediction_loss.backward()
            self.server_optim.step()
            privacy_loss.backward()
            self.adversary_optim.step()

            # Backprop over ...............

            if self.config["pruning_style"] == "maxentropy" or self.config["pruning_style"] == "adversarial":
                if self.config["pruning_style"] == "maxentropy":
                    self.adversary_optim.zero_grad()
                    adversary_input.grad.zero_()
                    protected_attribute_prediction = self.adversary_model(adversary_input)
                    entropy_loss = self.entropy_loss_fn(protected_attribute_prediction)
                    entropy_loss.backward()
                z_hat.backward(server_input.grad - adversary_input.grad)
                self.pruner_optim.step()

                z.backward(pruner_input.grad)
                self.client_optim.step()

            elif self.config["pruning_style"] == "network":
                z_hat.backward(server_input.grad, retain_graph=True)
                z.backward(pruner_input.grad)
                self.client_optim.step()

                z_hat.backward(- adversary_input.grad)
                self.pruner_optim.step()

            elif self.config["pruning_style"] == "vib":
                mu, logvar = indices[0], indices[1]
                kl_loss = self.kl_loss_fn(z_hat, mu, logvar)
                beta = self.config["lambda_vib"]
                (beta * kl_loss).backward(retain_graph=True)
                z_hat.backward(server_input.grad)
                z.backward(pruner_input.grad)
                self.client_optim.step()
                self.pruner_optim.step()

            elif self.config["pruning_style"] == "inlp":
                task_grad = server_input.grad.detach()
                g = adversary_input.grad.detach()

                task_grad_flat = task_grad.view(task_grad.size(0), -1)
                g_flat = g.view(g.size(0), -1)

                # Z_proj = Z - ( (Z · g) / ||g||^2 ) * g
                dot_product = (task_grad_flat * g_flat).sum(dim=1, keepdim=True)
                norm_g_sq = (g_flat * g_flat).sum(dim=1, keepdim=True) + 1e-8

                safe_grad_flat = task_grad_flat - (dot_product / norm_g_sq) * g_flat
                safe_grad = safe_grad_flat.view_as(server_input.grad)

                z_hat.backward(safe_grad)
                z.backward(pruner_input.grad)
                self.client_optim.step()
                self.pruner_optim.step()

            elif self.config["pruning_style"] == "hsic":
                z_hat.backward(server_input.grad, retain_graph=True)
                z_pooled = nn.AdaptiveAvgPool2d((1, 1))(z_hat) if len(z_hat.shape) == 4 else z_hat
                hsic_loss = self.hsic_loss_fn(z_pooled, protected_labels)
                lambda_hsic = self.config["lambda_hsic"]
                (lambda_hsic * hsic_loss).backward()
                z.backward(pruner_input.grad)
                self.client_optim.step()
                self.pruner_optim.step()

            elif self.config["pruning_style"] == "club":
                self.adversary_optim.zero_grad()
                if adversary_input.grad is not None:
                    adversary_input.grad.zero_()
                club_logits = self.adversary_model(adversary_input)
                club_loss = self.club_loss_fn(club_logits, protected_labels)
                lambda_club = self.config["lambda_club"]

                (lambda_club * club_loss).backward()
                z_hat.backward(server_input.grad + adversary_input.grad)
                z.backward(pruner_input.grad)
                self.client_optim.step()
                self.pruner_optim.step()

            elif self.config["pruning_style"] == "k_club":
                z_hat.backward(server_input.grad, retain_graph=True)
                z_pooled = nn.AdaptiveAvgPool2d((1, 1))(z_hat) if len(z_hat.shape) == 4 else z_hat

                hsic_loss = self.hsic_loss_fn(z_pooled, protected_labels)
                lambda_hsic = self.config.get("lambda_hsic", 1000.0)

                (lambda_hsic * hsic_loss).backward(retain_graph=True)

                self.adversary_optim.zero_grad()
                if adversary_input.grad is not None:
                    adversary_input.grad.zero_()
                club_logits = self.adversary_model(adversary_input)

                club_loss = self.club_loss_fn(
                    club_logits,
                    protected_labels,
                    prediction_labels,
                    z_pooled.detach()
                )

                lambda_club = self.config.get("lambda_club", 1.0)
                (lambda_club * club_loss).backward()
                z_hat.backward(adversary_input.grad)
                z.backward(pruner_input.grad)

                self.client_optim.step()
                self.pruner_optim.step()


            train_loss += final_loss.item()
            task_pred_correct_total += task_pred_correct
            if self.config["protected_attribute"] != "data":
                private_pred_correct_total += private_pred_correct
            total += int(data.shape[0])
            privacy_loss_total += privacy_loss

            self.logger.log_scalar("train_evaluation/final_loss", final_loss.item(), step_num)
            self.logger.log_scalar("train_evaluation/prediction_loss", prediction_loss.item(), step_num)
            self.logger.log_scalar("train_evaluation/task_accuracy", task_pred_correct_total / total, step_num)

            if self.config["protected_attribute"] != "data":
                self.logger.log_scalar("train_evaluation/privacy_accuracy", private_pred_correct_total / total,
                                       step_num)
            # try:
            #     self.logger.log_histogram("train/histogram_indices", indices.sum(0), step_num)
            # except:
            #     self.logger.log_histogram("train/histogram_indices", 0, step_num)

            if batch_idx % 100 == 0:
                self.logger.log_console(
                    "train epoch {}, iter {}, loss {:.4f}, task_accuracy {:.3f}, privacy_accuracy {:.3f}"
                    .format(self.epoch, batch_idx,
                            train_loss / total,
                            task_pred_correct_total / total,
                            private_pred_correct_total / total
                            ))
                self.log_and_save_model()

        epoch_latency = time.time() - epoch_start_time
        peak_mem_mb = 0
        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        try:
            privacy_acc = private_pred_correct_total / total
        except:
            privacy_acc = 0.0

        self.train_acc = task_pred_correct_total / total
        self.train_loss = train_loss / total

        self.logger.log_console(
            f"[Train] Epoch {self.epoch} | Latency: {epoch_latency:.3f} s | "
            f"Peak VRAM: {peak_mem_mb:.2f} MB | Utility(Acc): {self.train_acc:.4f} | Privacy(Acc): {privacy_acc:.4f}"
        )
        self.epoch += 1
