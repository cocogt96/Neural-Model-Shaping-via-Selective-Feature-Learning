import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pytorch_msssim import ms_ssim


def reconstruction_loss(img1, img2):
    l1_loss = nn.L1Loss()(img1, img2)
    if ms_ssim is not None:
        try:
            ssim_loss = 1 - ms_ssim(img1, img2, data_range=1.0, size_average=True)
            return l1_loss + 0.1 * ssim_loss
        except Exception as e:
            pass
    return l1_loss

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (CVPR 2018)
    Used to locate and extract residual sensitive features at the channel dimension
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SelfAttention(nn.Module):
    """
    SAGAN Self-Attention Mechanism (ICML 2019)
    For capturing global long-range dependencies within segmented features, aiding in reconstruction attacks
    """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        return self.gamma * out + x

class ConvTransposeAdversaryGen(nn.Module):
    """ 
    """
    def __init__(self, config):
        super(ConvTransposeAdversaryGen, self).__init__()
        input_nc = config["channels"]
        output_nc = 3
        ngf = 32
        use_bias = False
        n_blocks = 2
        use_dropout = False
        padding_type = 'reflect'
        norm_layer = nn.BatchNorm2d

        model = [
                    nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True),
                ]
        n_downsampling = 2

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        n_upsampling_extra = 3 - n_downsampling 
        for i in range(n_upsampling_extra):
            model += [nn.ConvTranspose2d(ngf, ngf,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      norm_layer(ngf), nn.ReLU(True)]
            if False and i == 0:
                model += [nn.Conv2d(ngf, ngf,
                                    kernel_size=2, stride=1, padding=0),
                         norm_layer(ngf), nn.ReLU(True)]
        model += [nn.ConvTranspose2d(ngf, output_nc, kernel_size=3, stride=2,
                                     padding=1, output_padding=1, bias=use_bias)]
        #model += [nn.Conv2d(output_nc, output_nc, kernel_size=3, )
        '''model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, ngf//2, kernel_size=7, padding=0)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf//2, ngf//4, kernel_size=5, padding=0)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf//4, output_nc, kernel_size=5, padding=0)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(output_nc, output_nc, kernel_size=5, padding=0)]'''

        self.m = nn.Sequential(*model)

    def forward(self, x):
        for l in self.m:
            x = l(x)
        return x


class AttentionAdversaryGen(nn.Module):
    """A Reconstruction Attack Model Incorporating Self-Attention Mechanisms and Smooth Upsampling"""

    def __init__(self, config):
        super(AttentionAdversaryGen, self).__init__()
        input_nc = config.get("channels", 64)
        ngf = 64

        # ResnetBlock params
        padding_type = 'reflect'
        norm_layer = nn.BatchNorm2d
        use_dropout = False
        use_bias = False

        self.head = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.body = nn.Sequential(
            ResnetBlock(ngf, padding_type, norm_layer, use_dropout, use_bias),
            SelfAttention(ngf),
            ResnetBlock(ngf, padding_type, norm_layer, use_dropout, use_bias),
            ResnetBlock(ngf, padding_type, norm_layer, use_dropout, use_bias)
        )
        self.tail = nn.Sequential(
            nn.Conv2d(ngf, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        x = self.tail(x)
        return x

class ResNet18AdversaryPred(nn.Module):
    """ Nothing special about the adversary model,
    it is a standard predictive model. Might update it later
    """
    def __init__(self, config):
        super(ResNet18AdversaryPred, self).__init__()
        self.logits = config["logits"]
        self.split_layer = config["split_layer"]

        self.model_name = config["model_name"]
        if self.model_name == "resnet50":
            self.adapter = nn.Conv2d(256, 64, kernel_size=1)
        else:
            self.adapter = nn.Identity()

        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(num_ftrs, self.logits))

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = self.adapter(x)
        for i, l in enumerate(self.model):
            if i <= self.split_layer:
                continue
            x = l(x)
        # return F.softmax(x, dim=1)
        return x


import torch.nn as nn
from torchvision import models


class MLLeaksAdversaryPred(nn.Module):
    """ResNet34 + SEBlock + ML-Leaks (Multi-Layer MLP) Architecture Inference Attack Model"""

    def __init__(self, config):
        super(MLLeaksAdversaryPred, self).__init__()
        self.logits = config["logits"]
        self.split_layer = config["split_layer"]
        self.model_name = config.get("model_name", "resnet18")

        self.base_model = models.resnet34(pretrained=False)
        self.features = nn.ModuleList(self.base_model.children())[:-2]

        in_channels = self._get_client_out_channels()
        out_channels = self._get_adv_in_channels()

        if in_channels != out_channels:
            self.adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.adapter = nn.Identity()

        self.se_block = SEBlock(channel=512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        num_ftrs = self.base_model.fc.in_features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.logits)
        )

    def _get_client_out_channels(self):
        if self.split_layer <= 3:
            return 64

        if self.model_name in ["resnet18", "resnet34"]:
            mapping = {4: 64, 5: 128, 6: 256, 7: 512}
            return mapping.get(self.split_layer, 512)
        elif self.model_name == "resnet50":
            mapping = {4: 256, 5: 512, 6: 1024, 7: 2048}
            return mapping.get(self.split_layer, 2048)
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

    def _get_adv_in_channels(self):
        next_layer_idx = self.split_layer + 1

        if next_layer_idx <= 4:
            return 64

        mapping = {5: 64, 6: 128, 7: 256, 8: 512}
        return mapping.get(next_layer_idx, 512)

    def forward(self, x):
        x = self.adapter(x)

        for i, l in enumerate(self.features):
            if i <= self.split_layer:
                continue
            x = l(x)

        x = self.se_block(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


class AdversaryModelGen(nn.Module):
    """
    Image reconstruction attack model routing.
    config["adv_gen_method"]: default-"conv_transpose", "attention", ...
    """

    def __init__(self, config):
        super(AdversaryModelGen, self).__init__()
        method = config["adv_gen_method"]

        if method == "attention":
            self.model = AttentionAdversaryGen(config)
        elif method == "conv-transpose":
            self.model = ConvTransposeAdversaryGen(config)
        else:
            raise ValueError(f"Unknown adv_gen_method: {method}")

    def forward(self, x):
        return self.model(x)


class AdversaryModelPred(nn.Module):
    """
    Attribute inference attack model routing.
    config["adv_pred_method"]: default-"resnet18", "ml_leaks", ...
    """

    def __init__(self, config):
        super(AdversaryModelPred, self).__init__()
        method = config["adv_pred_method"]

        if method == "ml-leaks":
            self.model = MLLeaksAdversaryPred(config)
        elif method == "resnet18":
            self.model = ResNet18AdversaryPred(config)
        else:
            raise ValueError(f"Unknown adv_pred_method: {method}")

    def forward(self, x):
        return self.model(x)