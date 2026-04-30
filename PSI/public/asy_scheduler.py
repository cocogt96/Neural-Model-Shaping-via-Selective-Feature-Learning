import random
import threading
from queue import Queue

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

from model_utils import recv_msg, send_msg
import socket
import time
from losses import EntropyLoss, HSICLoss, KernelGuidedCLUBLoss
import torch.nn.functional as F


class AsyScheduler():
    """docstring for Scheduler"""

    def __init__(self, config, objects, epoch=None):
        super(AsyScheduler, self).__init__()
        self.config = config
        self.objects = objects
        self.epoch = epoch + 1 if epoch else 0
        self.is_server = config['is_server']
        self.initialize()

        host = self.config.get('server_ip', '127.0.0.1')
        port = self.config.get('server_port', 10080)

        if not self.is_server:
            print(f"Client connecting to {host}:{port}")
            s = socket.socket()
            s.settimeout(10)
            s.connect((host, port))
            s.settimeout(None)
            self.socket = s
        else:
            print(f"Server binding to {host}:{port}")
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen(5)
            conn, addr = s.accept()
            self.socket = conn

    def initialize(self):
        gpu_devices = self.config.get("gpu_devices")
        gpu_id = gpu_devices[0]

        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(gpu_id))
        else:
            device = torch.device('cpu')

        self.setup_logger()
        self.setup_data_pipeline()
        self.setup_training_params()

        self.mask_train_mean = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
            self.config['train_batch_size'], 3, self.config["img_size"], self.config["img_size"]
        ).to(device)
        self.mask_train_std = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
            self.config['train_batch_size'], 3, self.config["img_size"], self.config["img_size"]
        ).to(device)

    def setup_logger(self):
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
                        "protected_attribute": self.config["protected_attribute"],
                        "format": "jpg"}
        test_config = {"transforms": trainTransform,
                       "train": False,
                       "path": self.config["dataset_path"],
                       "multi_attr": self.config["multi_attr"],
                       "prediction_attribute": self.config["prediction_attribute"],
                       "protected_attribute": self.config["protected_attribute"],
                       "format": "jpg"}

        train_dataset = Custom(train_config, self.config.get("dataset"))
        test_dataset = Custom(test_config, self.config.get("dataset"))

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(self.config["seed"])

        if self.config.get("split", False) is True:
            train_dataset, test_dataset = self.get_split(train_dataset)

        self.trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["train_batch_size"],
            shuffle=True, num_workers=5, drop_last=True,
            worker_init_fn=seed_worker, generator=g)

        self.testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config["test_batch_size"],
            shuffle=False, num_workers=5, drop_last=True,
            worker_init_fn=seed_worker, generator=g)

    def log_and_save_model(self):
        if not self.is_server:
            self.logger.log_model_stats(self.client_model)
            self.logger.save_model(self.client_model, self.client_model_path)
        else:
            self.logger.log_model_stats(self.server_model)
            self.logger.log_model_stats(self.adversary_model)
            self.logger.log_model_stats(self.pruner)
            self.logger.save_model(self.server_model, self.server_model_path)
            self.logger.save_model(self.adversary_model, self.adversary_model_path)
            self.logger.save_model(self.pruner, self.pruner_model_path)

    def setup_training_params(self):
        self.device = self.objects["device"]

        self.client_model_path = self.config["model_path"] + "client_model.pt"
        self.adversary_model_path = self.config["model_path"] + "adversary_model.pt"
        self.server_model_path = self.config["model_path"] + "server_model.pt"
        self.decoder_model_path = self.config["model_path"] + "decoder_model.pt"
        self.pruner_model_path = self.config["model_path"] + "pruner_model.pt"
        self.challenge_dataset_path = self.config["results_path"] + "/challenge/"

        self.load_models()
        if self.config["experiment_type"] == "training":
            self.load_loss_fns()
            self.load_optim()

        self.lambda1 = self.config["lambda1"]
        self.lambda2 = self.config["lambda2"]

    def load_optim(self):
        if not self.is_server:
            self.client_optim = self.objects["client_optim"]
        else:
            self.server_optim = self.objects["server_optim"]
            self.adversary_optim = self.objects["adversary_optim"]
            self.pruner_optim = self.objects["pruner_optim"]

    def load_loss_fns(self):
        self.prediction_loss_fn = self.objects["prediction_loss_fn"]
        self.privacy_loss_fn = self.objects["privacy_loss_fn"]
        if self.config["pruning_style"] == "k_club":
            is_cont = (self.config["protected_attribute"] == "data")
            self.hsic_loss_fn = HSICLoss()
            self.club_loss_fn = KernelGuidedCLUBLoss(is_continuous=is_cont)

    def make_multi_gpu_if_available(self, model):
        if len(self.config["gpu_devices"]) > 1:
            return nn.DataParallel(model, device_ids=self.config["gpu_devices"])
        return model

    def load_models(self):
        if not self.is_server:
            self.client_model = self.make_multi_gpu_if_available(self.objects["client_model"])
            self.objects["server_model"].to('cpu')
            self.objects["adversary_model"].to('cpu')
            self.objects["pruner"].to('cpu')
            if "decoder" in self.objects: self.objects["decoder"].to('cpu')
        else:
            self.server_model = self.make_multi_gpu_if_available(self.objects["server_model"])
            self.adversary_model = self.make_multi_gpu_if_available(self.objects["adversary_model"])
            self.pruner = self.make_multi_gpu_if_available(self.objects["pruner"])
            if "decoder" in self.objects:
                self.decoder = self.make_multi_gpu_if_available(self.objects["decoder"])
            self.objects["client_model"].to('cpu')

        torch.cuda.empty_cache()
        self.logger.log_console(f"[{'Server' if self.is_server else 'Client'}] Models loaded and VRAM optimized.")

    def clear_optim_grad(self):
        if not self.is_server:
            self.client_optim.zero_grad()
        else:
            self.server_optim.zero_grad()
            self.adversary_optim.zero_grad()
            if hasattr(self, 'pruner_optim'):
                self.pruner_optim.zero_grad()

    def eval_mode(self):
        if not self.is_server:
            self.client_model.eval()
        else:
            self.server_model.eval()
            self.pruner.eval()
            self.adversary_model.eval()
            if hasattr(self, 'decoder'):
                self.decoder.eval()

    def train_mode(self):
        if not self.is_server:
            self.client_model.train()
        else:
            self.server_model.train()
            self.pruner.train()
            self.adversary_model.train()
            if hasattr(self, 'decoder'):
                self.decoder.train()

    def cleanse_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def load_saved_models(self):
        if not self.is_server:
            try:
                self.objects["client_model"].load_state_dict(
                    torch.load(self.client_model_path, map_location=self.device))
            except:
                self.objects["client_model"].load_state_dict(
                    self.cleanse_state_dict(torch.load(self.client_model_path, map_location=self.device)))
        else:
            try:
                self.objects["pruner"].load_state_dict(torch.load(self.pruner_model_path, map_location=self.device))
                self.objects["adversary_model"].load_state_dict(
                    torch.load(self.adversary_model_path, map_location=self.device))
                self.objects["server_model"].load_state_dict(
                    torch.load(self.server_model_path, map_location=self.device))
            except:
                self.objects["pruner"].load_state_dict(
                    self.cleanse_state_dict(torch.load(self.pruner_model_path, map_location=self.device)))
                self.objects["adversary_model"].load_state_dict(
                    self.cleanse_state_dict(torch.load(self.adversary_model_path, map_location=self.device)))
                self.objects["server_model"].load_state_dict(
                    self.cleanse_state_dict(torch.load(self.server_model_path, map_location=self.device)))
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
        if self.is_server and not os.path.exists(self.challenge_dataset_path):
            os.makedirs(self.challenge_dataset_path)

        self.labels_tensor = []
        self.counter = 0

        with torch.no_grad():
            for batch_idx, sample in enumerate(self.testloader):
                data = Variable(sample["img"]).to(self.device)
                filename = [name.split("/")[-1].split('.')[0] for name in sample["filename"]]

                if not self.is_server:
                    z = self.client_model(data)
                    send_msg(self.socket, z.cpu().numpy())
                else:
                    z_np, _ = recv_msg(self.socket)
                    z = torch.from_numpy(z_np).to(self.device)

                    z_hat, indices = self.pruner(z)
                    self.save_challenge_dataset(z_hat, filename)
                    if self.config['protected_attribute'] == 'data':
                        rec_img = self.adversary_model(z_hat)
                        self.save_reconstruction_image(rec_img, filename)

    def test(self):
        self.eval_mode()
        task_loss, priv_loss, task_pred_correct, privacy_pred_correct = 0, 0, 0, 0
        total = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        with torch.no_grad():
            self.inference_list = []
            for batch_idx, sample in enumerate(self.testloader):
                data = Variable(sample["img"]).to(self.device)
                prediction_labels = Variable(sample["prediction_label"]).to(self.device)

                if self.config["protected_attribute"] != "data":
                    protected_labels = Variable(sample["private_label"]).to(self.device)
                else:
                    protected_labels = data.clone()

                if not self.is_server:
                    if torch.cuda.is_available(): torch.cuda.synchronize(self.device)
                    start = time.perf_counter()

                    z = self.client_model(data)

                    send_msg(self.socket, z.cpu().detach().numpy())

                    if torch.cuda.is_available(): torch.cuda.synchronize(self.device)
                    end = time.perf_counter()
                    self.inference_list.append(end - start)

                else:
                    if torch.cuda.is_available(): torch.cuda.synchronize(self.device)
                    start = time.perf_counter()

                    z_np, _ = recv_msg(self.socket)
                    z = torch.from_numpy(z_np).to(self.device)

                    z_hat, indices = self.pruner(z)

                    task_attribute_logits = self.server_model(z_hat)
                    protected_attribute_logits = self.adversary_model(z_hat)

                    if torch.cuda.is_available(): torch.cuda.synchronize(self.device)
                    end = time.perf_counter()
                    self.inference_list.append(end - start)

                    prediction_loss = self.prediction_loss_fn(task_attribute_logits, prediction_labels)
                    privacy_loss = self.privacy_loss_fn(protected_attribute_logits, protected_labels)

                    task_loss += prediction_loss.item()
                    priv_loss += privacy_loss.item()

                    task_pred_correct += (task_attribute_logits.argmax(dim=1) == prediction_labels).sum().item()
                    if self.config["protected_attribute"] != "data":
                        privacy_pred_correct += (
                                protected_attribute_logits.argmax(dim=1) == protected_labels).sum().item()
                    total += int(data.shape[0])

        peak_mem_mb = 0
        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        test_latency = sum(self.inference_list)

        if self.is_server:
            task_loss /= total
            priv_loss /= total
            task_pred_acc = task_pred_correct / total
            try:
                privacy_pred_acc = privacy_pred_correct / total
            except ZeroDivisionError:
                privacy_pred_acc = 0.0

            self.test_acc = task_pred_acc
            self.test_loss = torch.tensor(task_loss)

            self.logger.log_console(
                f"[Server TEST] Epoch {self.epoch} | Latency: {test_latency:.3f} s | "
                f"Peak VRAM: {peak_mem_mb:.2f} MB | Utility(Acc): {task_pred_acc:.4f} | Privacy(Acc): {privacy_pred_acc:.4f}"
            )
        else:
            self.logger.log_console(
                f"[Client TEST] Epoch {self.epoch} | Latency: {test_latency:.3f} s | Peak VRAM: {peak_mem_mb:.2f} MB"
            )

    def train(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        tem_time, temp_time = [], []
        train_loss = 0
        task_pred_correct_total = 0
        private_pred_correct_total = 0
        privacy_loss_total = 0
        total = 0

        self.train_mode()
        total_batches = len(self.trainloader)

        QUEUE_MAX_SIZE = 16

        if not self.is_server:
            z_queue = Queue(maxsize=QUEUE_MAX_SIZE)

            step_finished_event = threading.Event()
            step_finished_event.set()

            def forward_and_send():
                for batch_idx, sample in enumerate(self.trainloader):
                    if batch_idx % QUEUE_MAX_SIZE == 0:
                        step_finished_event.wait()
                        step_finished_event.clear()

                    data = Variable(sample["img"]).to(self.device)
                    z = self.client_model(data)

                    send_msg(self.socket, z.cpu().detach().numpy())
                    z_queue.put(z)

            def recv_and_backward():
                self.client_optim.zero_grad()

                for batch_idx in range(total_batches):
                    z = z_queue.get()
                    grad_np, _ = recv_msg(self.socket)
                    grad = torch.from_numpy(grad_np).to(self.device)

                    z.backward(grad)

                    is_last_batch = (batch_idx + 1) == total_batches
                    if (batch_idx + 1) % QUEUE_MAX_SIZE == 0 or is_last_batch:
                        self.client_optim.step()
                        self.client_optim.zero_grad()

                        step_finished_event.set()

                    z_queue.task_done()

            start = time.time()
            t_fwd = threading.Thread(target=forward_and_send)
            t_bwd = threading.Thread(target=recv_and_backward)

            t_fwd.start()
            t_bwd.start()

            t_fwd.join()
            t_bwd.join()

            end = time.time()
            temp_time.append(end - start)

            peak_mem_mb = 0
            if torch.cuda.is_available():
                peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

            epoch_latency = sum(temp_time)
            self.logger.log_console(
                f"[Client] Epoch {self.epoch} | Latency: {epoch_latency:.3f} s | Peak VRAM: {peak_mem_mb:.2f} MB"
            )

        else:
            z_queue = Queue(maxsize=QUEUE_MAX_SIZE)

            def receive_z():
                for _ in range(total_batches):
                    z_np, _ = recv_msg(self.socket)
                    z_queue.put(z_np)

            start = time.time()
            t_recv = threading.Thread(target=receive_z)
            t_recv.start()

            for batch_idx, sample in enumerate(self.trainloader):
                step_num = total_batches * self.epoch + batch_idx
                prediction_labels = Variable(sample["prediction_label"]).to(self.device)

                if self.config["protected_attribute"] != "data":
                    protected_labels = Variable(sample["private_label"]).to(self.device)
                else:
                    protected_labels = Variable(sample["img"]).to(self.device)

                z_np = z_queue.get()
                z_received = torch.from_numpy(z_np).to(self.device).requires_grad_(True)

                self.clear_optim_grad()

                pruner_input = Variable(z_received, requires_grad=True)
                z_hat, indices = self.pruner(pruner_input)

                server_input = z_hat.detach().clone().requires_grad_(True)
                task_attribute_logits = self.server_model(server_input)

                adversary_input = z_hat.detach().clone().requires_grad_(True)
                protected_attribute_logits = self.adversary_model(adversary_input)

                prediction_loss = self.prediction_loss_fn(task_attribute_logits, prediction_labels)
                privacy_loss = self.privacy_loss_fn(protected_attribute_logits, protected_labels)
                final_loss = self.lambda1 * prediction_loss + self.lambda2 * privacy_loss

                task_pred_correct = (task_attribute_logits.argmax(dim=1) == prediction_labels).sum().item()
                if self.config["protected_attribute"] != "data":
                    private_pred_correct = (protected_attribute_logits.argmax(dim=1) == protected_labels).sum().item()

                prediction_loss.backward()
                self.server_optim.step()
                privacy_loss.backward()
                self.adversary_optim.step()

                if self.config.get("pruning_style") == "k_club":
                    z_hat.backward(server_input.grad, retain_graph=True)
                    z_pooled = torch.nn.AdaptiveAvgPool2d((1, 1))(z_hat) if len(z_hat.shape) == 4 else z_hat
                    hsic_loss = self.hsic_loss_fn(z_pooled, protected_labels)
                    (self.config["lambda_hsic"] * hsic_loss).backward(retain_graph=True)

                    self.adversary_optim.zero_grad()
                    if adversary_input.grad is not None: adversary_input.grad.zero_()
                    club_logits = self.adversary_model(adversary_input)
                    club_loss = self.club_loss_fn(club_logits, protected_labels, prediction_labels, z_pooled.detach())
                    (self.config["lambda_club"] * club_loss).backward()

                    z_hat.backward(adversary_input.grad)
                    if hasattr(self, 'pruner_optim'): self.pruner_optim.step()
                    z_received.backward(pruner_input.grad)

                send_msg(self.socket, z_received.grad.cpu().detach().numpy())
                z_queue.task_done()

                train_loss += final_loss.item()
                task_pred_correct_total += task_pred_correct
                if self.config["protected_attribute"] != "data":
                    private_pred_correct_total += private_pred_correct
                total += int(prediction_labels.shape[0])
                privacy_loss_total += privacy_loss

                self.logger.log_scalar("train_evaluation/final_loss", final_loss.item(), step_num)
                self.logger.log_scalar("train_evaluation/prediction_loss", prediction_loss.item(), step_num)
                self.logger.log_scalar("train_evaluation/task_accuracy", task_pred_correct_total / total, step_num)

                if self.config["protected_attribute"] != "data":
                    self.logger.log_scalar("train_evaluation/privacy_accuracy", private_pred_correct_total / total,
                                           step_num)

                if batch_idx % 100 == 0:
                    try:
                        privacy_acc = private_pred_correct_total / total
                    except:
                        privacy_acc = 0.0

                    self.logger.log_console(
                        "train epoch {}, iter {}, loss {:.4f}, task_accuracy {:.3f}, privacy_accuracy {:.3f}"
                        .format(self.epoch, batch_idx, train_loss / total, task_pred_correct_total / total, privacy_acc)
                    )
                    self.log_and_save_model()

            t_recv.join()

            end = time.time()
            tem_time.append(end - start)

            peak_mem_mb = 0
            if torch.cuda.is_available():
                peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

            try:
                privacy_acc = private_pred_correct_total / total
            except:
                privacy_acc = 0.0

            self.train_acc = task_pred_correct_total / total
            self.train_loss = train_loss / total

            epoch_latency = sum(tem_time)
            self.logger.log_console(
                f"[Server] Epoch {self.epoch} | Latency: {epoch_latency:.3f} s | "
                f"Peak VRAM: {peak_mem_mb:.2f} MB | Utility(Acc): {self.train_acc:.4f} | Privacy(Acc): {privacy_acc:.4f}"
            )

        self.epoch += 1