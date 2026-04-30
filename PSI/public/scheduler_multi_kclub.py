import random
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import os
import time
import torch.nn as nn
from collections import OrderedDict

from dataset_utils import Custom
from utils import LoggerUtils
from losses import HSICLoss, KernelGuidedCLUBLoss


class SchedulerMultiKCLUB():
    def __init__(self, config, objects, epoch=None):
        self.config = config
        self.objects = objects
        self.epoch = epoch + 1 if epoch else 0
        self.initialize()

    def initialize(self):
        gpu_devices = self.config.get("gpu_devices")
        gpu_id = gpu_devices[0]
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

        self.setup_logger()
        self.setup_data_pipeline()
        self.setup_training_params()

    def setup_logger(self):
        log_config = {"log_path": self.config["log_path"]}
        self.logger = LoggerUtils(log_config)

    def setup_data_pipeline(self):
        self.IM_SIZE = self.config["img_size"]
        trainTransform = transforms.Compose([
            transforms.Resize((self.IM_SIZE, self.IM_SIZE)),
            transforms.ToTensor()
        ])

        train_config = {
            "transforms": trainTransform, "train": True,
            "path": self.config["dataset_path"],
            "multi_attr": self.config["multi_attr"],
            "prediction_attribute": self.config["prediction_attribute"],
            "protected_attribute": self.config["protected_attribute"],
            "format": "jpg"
        }
        test_config = train_config.copy()
        test_config["train"] = False

        train_dataset = Custom(train_config, self.config.get("dataset"))
        test_dataset = Custom(test_config, self.config.get("dataset"))

        g = torch.Generator()
        g.manual_seed(self.config["seed"])

        self.trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["train_batch_size"],
            shuffle=True, num_workers=5, drop_last=True, generator=g)

        self.testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config["test_batch_size"],
            shuffle=False, num_workers=5, drop_last=True, generator=g)

    def setup_training_params(self):
        self.client_model = self.objects["client_model"]
        self.server_model = self.objects["server_model"]
        self.pruner = self.objects["pruner"]

        self.adversary_model_1 = self.objects["adversary_model_1"]
        self.adversary_model_2 = self.objects["adversary_model_2"]

        self.client_optim = self.objects["client_optim"]
        self.server_optim = self.objects["server_optim"]
        self.pruner_optim = self.objects["pruner_optim"]
        self.adversary_optim_1 = self.objects["adversary_optim_1"]
        self.adversary_optim_2 = self.objects["adversary_optim_2"]

        self.prediction_loss_fn = self.objects["prediction_loss_fn"]
        self.privacy_loss_fn = self.objects["privacy_loss_fn"]

        # K-CLUB specific losses
        self.hsic_loss_fn = HSICLoss()
        self.club_loss_fn = KernelGuidedCLUBLoss(is_continuous=False)

        self.lambda1 = self.config.get("lambda1", 1.0)
        self.lambda2 = self.config.get("lambda2", 1.0)

    def clear_optim_grad(self):
        self.server_optim.zero_grad()
        self.adversary_optim_1.zero_grad()
        self.adversary_optim_2.zero_grad()
        self.client_optim.zero_grad()
        self.pruner_optim.zero_grad()

    def eval_mode(self):
        self.client_model.eval()
        self.server_model.eval()
        self.pruner.eval()
        self.adversary_model_1.eval()
        self.adversary_model_2.eval()

    def train_mode(self):
        self.client_model.train()
        self.server_model.train()
        self.pruner.train()
        self.adversary_model_1.train()
        self.adversary_model_2.train()

    def train(self):
        train_loss = 0
        task_pred_correct_total = 0
        private_pred_correct_1_total = 0
        private_pred_correct_2_total = 0
        total = 0

        self.train_mode()
        epoch_start_time = time.time()

        for batch_idx, sample in enumerate(self.trainloader):
            step_num = len(self.trainloader) * self.epoch + batch_idx
            data = Variable(sample["img"]).to(self.device)
            prediction_labels = Variable(sample["prediction_label"]).to(self.device)

            protected_labels_1 = Variable(sample["private_label"][:, 0]).to(self.device)
            protected_labels_2 = Variable(sample["private_label"][:, 1]).to(self.device)

            self.clear_optim_grad()
            z = self.client_model(data)

            pruner_input = Variable(z.detach().clone(), requires_grad=True)
            z_hat, indices = self.pruner(pruner_input)

            server_input = z_hat.detach().clone().requires_grad_(True)
            task_attribute_logits = self.server_model(server_input)

            adversary_input_1 = z_hat.detach().clone().requires_grad_(True)
            protected_attribute_logits_1 = self.adversary_model_1(adversary_input_1)

            adversary_input_2 = z_hat.detach().clone().requires_grad_(True)
            protected_attribute_logits_2 = self.adversary_model_2(adversary_input_2)

            prediction_loss = self.prediction_loss_fn(task_attribute_logits, prediction_labels)
            privacy_loss_1 = self.privacy_loss_fn(protected_attribute_logits_1, protected_labels_1)
            privacy_loss_2 = self.privacy_loss_fn(protected_attribute_logits_2, protected_labels_2)

            final_loss = self.lambda1 * prediction_loss + self.lambda2 * (privacy_loss_1 + privacy_loss_2)

            prediction_loss.backward()
            self.server_optim.step()
            privacy_loss_1.backward()
            self.adversary_optim_1.step()
            privacy_loss_2.backward()
            self.adversary_optim_2.step()

            # 1. HSIC Loss
            z_hat.backward(server_input.grad, retain_graph=True)
            z_pooled = nn.AdaptiveAvgPool2d((1, 1))(z_hat) if len(z_hat.shape) == 4 else z_hat

            hsic_loss_1 = self.hsic_loss_fn(z_pooled, protected_labels_1)
            hsic_loss_2 = self.hsic_loss_fn(z_pooled, protected_labels_2)
            lambda_hsic = self.config.get("lambda_hsic", 1000.0)

            (lambda_hsic * (hsic_loss_1 + hsic_loss_2)).backward(retain_graph=True)

            # 2. CLUB Mutual Information Upper Bound
            self.adversary_optim_1.zero_grad()
            self.adversary_optim_2.zero_grad()
            if adversary_input_1.grad is not None: adversary_input_1.grad.zero_()
            if adversary_input_2.grad is not None: adversary_input_2.grad.zero_()

            club_logits_1 = self.adversary_model_1(adversary_input_1)
            club_logits_2 = self.adversary_model_2(adversary_input_2)

            club_loss_1 = self.club_loss_fn(club_logits_1, protected_labels_1, prediction_labels, z_pooled.detach())
            club_loss_2 = self.club_loss_fn(club_logits_2, protected_labels_2, prediction_labels, z_pooled.detach())

            lambda_club = self.config.get("lambda_club", 3.0)
            (lambda_club * (club_loss_1 + club_loss_2)).backward()

            z_hat.backward(adversary_input_1.grad + adversary_input_2.grad)
            z.backward(pruner_input.grad)

            self.client_optim.step()
            self.pruner_optim.step()

            task_pred_correct_total += (task_attribute_logits.argmax(dim=1) == prediction_labels).sum().item()
            private_pred_correct_1_total += (
                        protected_attribute_logits_1.argmax(dim=1) == protected_labels_1).sum().item()
            private_pred_correct_2_total += (
                        protected_attribute_logits_2.argmax(dim=1) == protected_labels_2).sum().item()

            train_loss += final_loss.item()
            total += int(data.shape[0])

            if batch_idx % 100 == 0:
                self.logger.log_console(
                    f"Train Epoch {self.epoch}, Iter {batch_idx}, Loss {train_loss / total:.4f}, "
                    f"TaskAcc {task_pred_correct_total / total:.3f}, "
                    f"PrivAcc1 {private_pred_correct_1_total / total:.3f}, "
                    f"PrivAcc2 {private_pred_correct_2_total / total:.3f}"
                )

        self.train_acc = task_pred_correct_total / total
        self.train_loss = train_loss / total
        self.epoch += 1

    def test(self):
        self.eval_mode()
        task_pred_correct, private_pred_correct_1, private_pred_correct_2, total = 0, 0, 0, 0
        task_loss = 0

        with torch.no_grad():
            self.inference_list = []
            for batch_idx, sample in enumerate(self.testloader):
                data = Variable(sample["img"]).to(self.device)
                prediction_labels = Variable(sample["prediction_label"]).to(self.device)
                protected_labels_1 = Variable(sample["private_label"][:,0]).to(self.device)
                protected_labels_2 = Variable(sample["private_label"][:,1]).to(self.device)

                start = time.time()
                z = self.client_model(data)
                z_hat, _ = self.pruner(z)
                task_attribute_logits = self.server_model(z_hat)
                self.inference_list.append(time.time() - start)

                protected_attribute_logits_1 = self.adversary_model_1(z_hat)
                protected_attribute_logits_2 = self.adversary_model_2(z_hat)

                task_loss += self.prediction_loss_fn(task_attribute_logits, prediction_labels).item()

                task_pred_correct += (task_attribute_logits.argmax(dim=1) == prediction_labels).sum().item()
                private_pred_correct_1 += (
                            protected_attribute_logits_1.argmax(dim=1) == protected_labels_1).sum().item()
                private_pred_correct_2 += (
                            protected_attribute_logits_2.argmax(dim=1) == protected_labels_2).sum().item()

                total += int(data.shape[0])

        self.test_acc = task_pred_correct / total
        self.test_loss = torch.tensor(task_loss / total)

        self.logger.log_console(
            f"[Test] Epoch {self.epoch - 1} | Utility(Acc): {self.test_acc:.4f} | "
            f"Privacy1(Acc): {private_pred_correct_1 / total:.4f} | Privacy2(Acc): {private_pred_correct_2 / total:.4f}"
        )