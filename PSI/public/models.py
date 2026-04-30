import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
distance = nn.CrossEntropyLoss()

def loss_fn(logits, labels):
    CE = distance(logits, labels)
    return CE


class VGG16Client(nn.Module):
    def __init__(self, config):
        super(VGG16Client, self).__init__()
        self.split_layer = config["split_layer"]
        self.grid_crop = config["grid_crop"]

        self.model = models.vgg16(pretrained=False)

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i > self.split_layer:
                break
            if i == 0 and self.grid_crop:
                B, C, H, W = x.shape
                C_out = l.out_channels
                G = int(C_out ** 0.5)
                assert G * G == C_out, "Grid Crop requires that the number of output channels in the first layer must be a perfect square!"
                w_h, w_w = H // G, W // G

                patches = x.view(B, C, G, w_h, G, w_w)
                patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
                patches = patches.view(B * C_out, C, w_h, w_w)
                patches = F.interpolate(patches, size=(H, W), mode='bilinear', align_corners=False)

                out = l(patches)
                _, _, H_prime, W_prime = out.shape

                out = out.view(B, C_out, C_out, H_prime, W_prime)
                x = out.sum(dim=2)
            else:
                x = l(x)
        return x


class VGG16Server(nn.Module):
    def __init__(self, config):
        super(VGG16Server, self).__init__()
        self.logits = config["logits"]
        self.split_layer = config["split_layer"]

        self.model = models.vgg16(pretrained=False)
        #num_ftrs = self.model.fc.in_features
        num_ftrs = 4096
        num_ftrs = self.model.classifier[0].out_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(num_ftrs, self.logits))

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i <= self.split_layer:
                continue
            x = l(x)
        return nn.functional.softmax(x, dim=1)




class ResNet18Client(nn.Module):
    def __init__(self, config):
        super(ResNet18Client, self).__init__()
        self.split_layer = config["split_layer"]
        self.grid_crop = config["grid_crop"]

        self.model = models.resnet18(pretrained=True)

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i > self.split_layer:
                break
            if i == 0 and self.grid_crop:
                B, C, H, W = x.shape
                C_out = l.out_channels
                G = int(C_out ** 0.5)
                assert G * G == C_out, "Grid Crop requires that the number of output channels in the first layer must be a perfect square!"
                w_h, w_w = H // G, W // G

                patches = x.view(B, C, G, w_h, G, w_w)
                patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
                patches = patches.view(B * C_out, C, w_h, w_w)
                patches = F.interpolate(patches, size=(H, W), mode='bilinear', align_corners=False)

                out = l(patches)
                _, _, H_prime, W_prime = out.shape

                out = out.view(B, C_out, C_out, H_prime, W_prime)
                x = out.sum(dim=2)
            else:
                x = l(x)
        return x


class ResNet18Server(nn.Module):
    def __init__(self, config):
        super(ResNet18Server, self).__init__()
        self.logits = config["logits"]
        self.split_layer = config["split_layer"]

        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(num_ftrs, self.logits))

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i <= self.split_layer:
                continue
            x = l(x)
        # return nn.functional.softmax(x, dim=1)
        return x


class ResNet34Client(nn.Module):
    def __init__(self, config):
        super(ResNet34Client, self).__init__()
        self.split_layer = config["split_layer"]
        self.grid_crop = config["grid_crop"]

        self.model = models.resnet34(pretrained=True)

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i > self.split_layer:
                break
            if i == 0 and self.grid_crop:
                B, C, H, W = x.shape
                C_out = l.out_channels
                G = int(C_out ** 0.5)
                assert G * G == C_out, "Grid Crop requires that the number of output channels in the first layer must be a perfect square!"
                w_h, w_w = H // G, W // G

                patches = x.view(B, C, G, w_h, G, w_w)
                patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
                patches = patches.view(B * C_out, C, w_h, w_w)
                patches = F.interpolate(patches, size=(H, W), mode='bilinear', align_corners=False)

                out = l(patches)
                _, _, H_prime, W_prime = out.shape

                out = out.view(B, C_out, C_out, H_prime, W_prime)
                x = out.sum(dim=2)
            else:
                x = l(x)
        return x


class ResNet34Server(nn.Module):
    def __init__(self, config):
        super(ResNet34Server, self).__init__()
        self.logits = config["logits"]
        self.split_layer = config["split_layer"]

        self.model = models.resnet34(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(num_ftrs, self.logits))

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i <= self.split_layer:
                continue
            x = l(x)
        # return nn.functional.softmax(x, dim=1)
        return x


class ResNet50Client(nn.Module):
    def __init__(self, config):
        super(ResNet50Client, self).__init__()
        self.split_layer = config["split_layer"]
        self.grid_crop = config["grid_crop"]

        self.model = models.resnet50(pretrained=True)

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i > self.split_layer:
                break
            if i == 0 and self.grid_crop:
                B, C, H, W = x.shape
                C_out = l.out_channels
                G = int(C_out ** 0.5)
                assert G * G == C_out, "Grid Crop requires that the number of output channels in the first layer must be a perfect square!"
                w_h, w_w = H // G, W // G

                patches = x.view(B, C, G, w_h, G, w_w)
                patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
                patches = patches.view(B * C_out, C, w_h, w_w)
                patches = F.interpolate(patches, size=(H, W), mode='bilinear', align_corners=False)

                out = l(patches)
                _, _, H_prime, W_prime = out.shape

                out = out.view(B, C_out, C_out, H_prime, W_prime)
                x = out.sum(dim=2)
            else:
                x = l(x)
        return x


class ResNet50Server(nn.Module):
    def __init__(self, config):
        super(ResNet50Server, self).__init__()
        self.logits = config["logits"]
        self.split_layer = config["split_layer"]

        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(num_ftrs, self.logits))

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i <= self.split_layer:
                continue
            x = l(x)
        # return nn.functional.softmax(x, dim=1)
        return x


class AlexNetClient(nn.Module):
    def __init__(self, config):
        super(AlexNetClient, self).__init__()
        self.split_layer = config["split_layer"]
        self.grid_crop = config["grid_crop"]

        self.model = models.alexnet(pretrained=False)

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        #print([x for x in self.model.modules()])
        for i, l in enumerate(self.model):
            #lay = [module for module in l.children()]
            #print(lay)
            #print("self.split_layer", self.split_layer)
            if i > self.split_layer:
                #print("i",i)

                break
            if i == 0 and self.grid_crop:
                B, C, H, W = x.shape
                C_out = l.out_channels
                G = int(C_out ** 0.5)
                assert G * G == C_out, "Grid Crop requires that the number of output channels in the first layer must be a perfect square!"
                w_h, w_w = H // G, W // G

                patches = x.view(B, C, G, w_h, G, w_w)
                patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
                patches = patches.view(B * C_out, C, w_h, w_w)
                patches = F.interpolate(patches, size=(H, W), mode='bilinear', align_corners=False)

                out = l(patches)
                _, _, H_prime, W_prime = out.shape

                out = out.view(B, C_out, C_out, H_prime, W_prime)
                x = out.sum(dim=2)
            else:
                #print(i)

                if i == 2:
                    x = x.view(x.size(0),9216)
                x = l(x)
                #print(l, x.shape)
        # print(x.shape)
        return x


class AlexNetServer(nn.Module):
    def __init__(self, config):
        super(AlexNetServer, self).__init__()
        self.logits = config["logits"]
        self.split_layer = config["split_layer"]

        self.model = models.alexnet(pretrained=False)
        #num_ftrs = self.model.fc.in_features
        #print(self.model.classifier[-1])
        num_ftrs = self.model.classifier[-1].out_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(num_ftrs, self.logits))

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i <= self.split_layer:
                continue
            #print("x.shape",x.shape)
            #print(l)
            if i == 2:
                x = x.view(x.size(0), 9216)
            x = l(x)
        # return nn.functional.softmax(x, dim=1)
        return x


class PruningNetwork(nn.Module):
    """ Nothing special about the pruning model,
    it is a standard resnet predictive model. Might update it later
    """
    def __init__(self, config):
        super(PruningNetwork, self).__init__()
        self.pruning_ratio = config["pruning_ratio"]
        self.pruning_style = config["pruning_style"]
        # decoy layer to allow creation of optimizer
        self.decoy_layer = nn.Linear(10, 10)
        self.temp = 1/30

        if self.pruning_style == "network":
            self.logits = config["logits"]
            self.split_layer = config["split_layer"]
            model_name = config.get("model_name", "resnet18")
            if hasattr(models, model_name):
                self.model = getattr(models, model_name)(pretrained=False)
            else:
                raise ValueError(f"Unsupported model name: {model_name}")

            if model_name in ["resnet18", "resnet34", "resnet50"]:
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Sequential(nn.Flatten(),
                                              nn.Linear(num_ftrs, self.logits))
            elif model_name == "vgg16":
                num_ftrs = self.model.classifier[0].out_features
                self.model.fc = nn.Sequential(nn.Flatten(), nn.Linear(num_ftrs, self.logits))
            elif model_name == "alexnet":
                num_ftrs = self.model.classifier[-1].out_features
                self.model.fc = nn.Sequential(nn.Flatten(), nn.Linear(num_ftrs, self.logits))

            self.model = nn.ModuleList(self.model.children())
            self.model = nn.Sequential(*self.model)
        elif self.pruning_style == "noise":
            self.mean = -1.
            self.stddev = 2.
        elif self.pruning_style == "vib":
            self.split_layer = config["split_layer"]
            model_name = config.get("model_name", "resnet18")
            if hasattr(models, model_name):
                temp_model = getattr(models, model_name)(pretrained=False)
            else:
                raise ValueError(f"Unsupported model name: {model_name}")
            temp_client = nn.Sequential(*list(temp_model.children())[:self.split_layer + 1])
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 112, 112)
                dummy_out = temp_client(dummy_input)
                in_channels = dummy_out.shape[1]
            self.mu_net = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.logvar_net = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            del temp_model
            del temp_client

    def prune_channels(self, z, indices=None):
        # Indexing is an inplace operation which creates problem during backprop hence z is cloned first
        z = z.clone()
        z[:, indices] = 0.
        #print(z)
        return z

    @staticmethod
    def get_random_channels(x, ratio):
        num_channels = x.shape[1]
        num_prunable_channels = int(num_channels * ratio)
        channels_to_prune = torch.randperm(x.shape[1], device=x.device)[:num_prunable_channels]
        return channels_to_prune

    def custom_sigmoid(self, x, offset):
        exponent = (x - offset) / self.temp
        #print(exponent)
        #answer = (1 / (1 + torch.exp( - exponent / self.temp)))
        answer = nn.Sigmoid()(exponent)
        #print(answer)
        return answer

    def get_channels_from_network(self, x, ratio):
        fmap_score = self.network_forward(x)
        num_channels = x.shape[1]
        num_prunable_channels = int(num_channels * ratio)
        threshold_score = torch.sort(fmap_score)[0][:, num_prunable_channels].unsqueeze(1)
        #print("fmap_score shape:",fmap_score.shape)
        #print("threshold_score shape:",threshold_score.shape)

        fmap_score = self.custom_sigmoid(fmap_score, threshold_score)
        #print("fmap_score shape:",fmap_score.shape)
        new_famp_score = fmap_score.clone()
        new_famp_score[fmap_score<0.5]=0
        new_famp_score[fmap_score!=0]=1
        new_famp_score = torch.unsqueeze(new_famp_score,2)
        new_famp_score = torch.broadcast_to(new_famp_score,(x.shape[0],x.shape[1],x.shape[2]))
        new_famp_score = torch.unsqueeze(new_famp_score, 3)
        new_famp_score = torch.broadcast_to(new_famp_score, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        x = x*new_famp_score

        mask = (fmap_score >= 0.5).float()
        mask_ste = mask - fmap_score.detach() + fmap_score
        mask_ste = mask_ste.unsqueeze(-1).unsqueeze(-1)
        x = x * mask_ste
        try:
            index_array = torch.arange(num_channels).repeat(x.shape[0], 1).cuda()  # commented out for cude?
        except:
            index_array = torch.arange(num_channels).repeat(x.shape[0], 1)
        indices = index_array[fmap_score < 0.5]

        return x, indices

    def noise_channels(self, x):
        noise = x.data.new(x.size()).normal_(self.mean, self.stddev)
        return x + noise

    def network_forward(self, x):
        for i, l in enumerate(self.model):
            if i <= self.split_layer:
                continue
            #print(l, x.shape, x.device)
            x = l(x)
        #print(x.shape, x.device)
        return x

    def forward(self, x):
        if self.pruning_style == "random":
            indices = self.get_random_channels(x, self.pruning_ratio)
            x = self.prune_channels(x, indices)
        elif self.pruning_style in ["nopruning", "maxentropy", "adversarial", "hsic", "club", "hsic_club", "k_club", "inlp"]:
            # decoy indices to prevent scheduler from breaking
            indices = torch.randint(1, 100, (x.shape[0], 30), device=x.device)
        elif self.pruning_style == "network":
            x , indices = self.get_channels_from_network(x, self.pruning_ratio)
        elif self.pruning_style == "noise":
            indices = torch.randint(1, 100, (x.shape[0], 30), device=x.device)
            x = self.noise_channels(x)
        elif self.pruning_style == "vib":
            mu = self.mu_net(x)
            logvar = self.logvar_net(x)

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            x = mu + std * eps
            indices = [mu, logvar]
        return x, indices
