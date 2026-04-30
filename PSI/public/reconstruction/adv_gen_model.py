import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

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



class AdversaryModelGen(nn.Module):
    """
    """
    def __init__(self, input_nc):
        super(AdversaryModelGen, self).__init__()
        #input_nc = config["channels"]
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


class ResNet18Client(nn.Module):
    def __init__(self):
        super(ResNet18Client, self).__init__()
        self.split_layer = 4
        self.is_grid_crop = False
        #self.split_layer = config["split_layer"]
        #self.is_grid_crop = config["is_grid_crop"]

        self.model = models.resnet18(pretrained=True)

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)
        # if self.is_grid_crop:
        #     self.im_size = 112
        #     self.window_size = 14

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i > self.split_layer:
                break
            if i == 0 and self.is_grid_crop:
                batch_size = x.shape[0]
                out = []
                for n in range(batch_size):
                   for i in range(0, self.im_size, self.window_size):
                      for j in range(0, self.im_size, self.window_size):
                         patch = x[n, :, i:i+self.window_size, j:j+self.window_size]
                         patch = F.interpolate(patch.unsqueeze_(0), scale_factor=(8, 8), mode='bilinear', align_corners=False).squeeze_(0)
                         out.append(patch)
                out = torch.stack(out).float().to(x.device)
                out = l(out)
                x = []
                for i in range(0, out.shape[0], out.shape[1]):
                   x.append(out[i: i+out.shape[1]].sum(1).squeeze(1))
                x = torch.stack(x)
            else:
                #print(l, x.shape)
                x = l(x)
        # print(x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, inplanes, image_size, interpolate_mode='bilinear', widen=1):
        super(Decoder, self).__init__()

        self.image_size = image_size

        assert interpolate_mode in ['bilinear', 'nearest']
        self.interpolate_mode = interpolate_mode

        self.bce_loss = nn.BCELoss()

        self.decoder = nn.Sequential(
            nn.Conv2d(inplanes, int(12 * widen), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(12 * widen)),
            nn.ReLU(),
            nn.Conv2d(int(12 * widen), 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        if self.interpolate_mode == 'bilinear':
            features = F.interpolate(features, size=[self.image_size, self.image_size],
                                     mode='bilinear', align_corners=True)
        elif self.interpolate_mode == 'nearest':   # might be faster
            features = F.interpolate(features, size=[self.image_size, self.image_size],
                                     mode='nearest')
        else:
            raise NotImplementedError

        return self.decoder(features)


# class MinimalDecoder(nn.Module):
#     def __init__(self, input_nc, output_nc=3, input_dim=None, output_dim=None):
#         super(MinimalDecoder, self).__init__()
#         upsampling_num = int(np.log2(output_dim // input_dim))
#         model = [nn.Conv2d(input_nc, 1 * output_nc, kernel_size=1)]
#         for num in range(upsampling_num):
#             # coco delete
# 			# model += [nn.ConvTranspose2d(64 * output_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
# 			model += [
# 				nn.ConvTranspose2d(64 * output_nc, output_nc, kernel_size=3, stride=1, padding=1, output_padding=0)]
#             # model += [nn.Linear(1 * output_nc * input_dim ** 2, output_nc * output_dim ** 2)]
#         # self.m = nn.Sequential(*model)
# 		self.m = torch.nn.DataParallel(*model, device_ids=range(torch.cuda.device_count()))
#
#     def forward(self, x):
#         b = x.shape[0]
#         x = self.m[0](x)
#         x = x.view(b, -1)
#         x = self.m[1](x)
#         x = x.view(b, 3, 224, 224)
#         return self.m(x)