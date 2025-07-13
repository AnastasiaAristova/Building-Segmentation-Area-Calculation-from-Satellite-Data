import torch 
import torch.nn as nn
from PIL import Image
from torchvision.transforms import CenterCrop
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ConvReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EncoderBlockUnet(nn.Module):
    def __init__(self, in_c, out_c, depth=2, kernel_size=3, padding=[1,1,]) -> None:
        super(EncoderBlockUnet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(ConvReLU(in_c if i == 0 else out_c, out_c, kernel_size, padding[i]))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x_copy = x
        x = self.pool(x)
        return x, x_copy

class DecoderBlockUnet(nn.Module):
    def __init__(self, in_c, out_c, depth=2, kernel_size=3, padding=[1,1,]) -> None:
        super(DecoderBlockUnet, self).__init__()
        self.unconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.layers = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.layers.append(ConvReLU(in_c, out_c, kernel_size=kernel_size, padding=padding[i]))
            else:
                self.layers.append(ConvReLU(out_c, out_c, kernel_size=kernel_size, padding=padding[i]))

    def forward(self, x, x_copy):
        x = self.unconv(x)
        (_, _, H, W) = x.shape
        x_crop = CenterCrop(size=[H, W])(x_copy)
        x = torch.cat((x, x_crop), dim=1)
        for layer in self.layers:
            x = layer(x)
        return x

class UNet(nn.Module):
    def __init__(self, input_img_size = 326, n_class=1):
        super().__init__()
        self.input_img_size = input_img_size
        # encoder
        self.encoder1 = EncoderBlockUnet(3, 64, padding=[0, 1])
        self.encoder2 = EncoderBlockUnet(64, 64*2, padding=[0, 1])
        self.encoder3 = EncoderBlockUnet(64*2, 64*4)
        self.encoder4 = EncoderBlockUnet(64*4, 64*8)
        # bottleneck
        self.bottleneck = nn.Sequential(
            ConvReLU(64*8, 64*16, padding=0),
            ConvReLU(64*16, 64*16, padding=0)
        )
        # decoder (upsampling)
        self.decoder1 = DecoderBlockUnet(64*16, 64*8)
        self.decoder2 = DecoderBlockUnet(64*8, 64*4)
        self.decoder3 = DecoderBlockUnet(64*4, 64*2, padding=[1, 0])
        self.decoder4 = DecoderBlockUnet(64*2, 64, padding=[1, 0])
        self.conv_last  = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # encoder
        enc1_in = x
        enc2_in, enc1_out = self.encoder1(enc1_in)
        enc3_in, enc2_out = self.encoder2(enc2_in)
        enc4_in, enc3_out = self.encoder3(enc3_in)
        bottleneck_in, enc4_out = self.encoder4(enc4_in)
        # bottleneck
        bottleneck_out = self.bottleneck(bottleneck_in)
        #decoder
        dec1_out = self.decoder1(bottleneck_out, enc4_out)
        dec2_out = self.decoder2(dec1_out, enc3_out)
        dec3_out = self.decoder3(dec2_out, enc2_out)
        dec4_out = self.decoder4(dec3_out, enc1_out)
        output = self.conv_last(dec4_out)
        return output

# генератор для разбиения изображения image на подизображения размером sub_img_size
def sub_img_generator(image, sub_img_size):
    for row in range(image.shape[1] // sub_img_size[0]):
        for col in range(image.shape[2] // sub_img_size[1]):
            row1 = row * sub_img_size[0]
            row2 = (row + 1) * sub_img_size[0]
            col1 = col * sub_img_size[1]
            col2 = (col + 1) * sub_img_size[1]
            yield image[:, row1:row2, col1:col2], row1, row2, col1, col2


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
test_transform = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def process_image(image):
    (_, H, W) = image.shape
    pad_h = (326 - H) // 2
    pad_w = (326 - W) // 2
    padded_image = F.pad(
        image,
        pad=(pad_w, pad_w, pad_h, pad_h),  # (left, right, top, bottom)
        mode='reflect'
    )
    return padded_image


def predict_one(model, inputs):
    with torch.set_grad_enabled(False):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs).cpu().detach().numpy().squeeze()
        mask = outputs > 0.5
    return mask


def make_prediction(model, image, dimension):
    all_area = 0
    dimension = dimension*dimension
    mask = np.zeros([image.size[1], image.size[0]])
    image = test_transform(image)
    sub_gen = sub_img_generator(image, (250, 250))
    for subimg, row1, row2, col1, col2 in sub_gen:
        img = process_image(subimg)
        submask = predict_one(model, img.unsqueeze(0)).astype(int)
        area = np.count_nonzero(submask)
        all_area += area*dimension
        mask[row1:row2, col1:col2] = submask
    return all_area, mask


def check_img_size(img):
    width, height = img.size  # Get dimensions
    if width % 250 == 0 and height % 250 == 0:
        return img
    new_width = int(width/250)*250
    new_height = int(height / 250) * 250
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return img.crop((left, top, right, bottom))