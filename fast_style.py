import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.datasets as tvdata
import torchvision.transforms as TF
from PIL import Image
from transformer import Transformer
from net_input_norm import NetInputNorm
from total_variation import total_variation
from perceptual_net import PerceptualNet
import os
import sys
import argparse


def show(img, i):
    img.mul_(255)
    torchvision.utils.save_image(img, 'tmp_imgs/' + str(i) + '.jpg')
    img = img.detach().cpu()[0].permute((1, 2, 0))
    save = Image.fromarray(img.numpy().astype('uint8'))
    save.save('tmp_imgs/' + str(i) + '.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--style', required=True)
    parser.add_argument('--styleratio', type=float, default=1e8)
    parser.add_argument('--trainimg', required=True)
    args = parser.parse_args()

    device = torch.device(args.device)

    transformer = Transformer().to(device)

    transforms = TF.Compose([
        TF.ToTensor(),
        NetInputNorm()
    ])

    images = tvdata.ImageFolder(args.trainimg, transforms)

    norm = NetInputNorm()
    style = transforms(Image.open(args.style)).to(device)

    m = PerceptualNet()
    style_loss = StyleLoss(m, style, args.styleratio)
    sr_loss = SRLoss(m, style, 1)

    os.mkdir('tmp_imgs')

    norm.to(device)
    opt = optim.Adam(transformer.parameters(), lr=0.005)
    i = 0
    tloss = 0
    for imgs, _ in images:

        def make_loss():
            imgs = imgs.squeeze(1).to(device)
            opt.zero_grad()
            transformed = transformer(imgs)
            size = imgs[0, 0].shape
            small_result = F.interpolate(norm(transformed), size, mode='bilinear')
            loss = style_loss(small_result, imgs)
            loss += sr_loss(norm(transformed))
            loss +=  100 * total_variation(transformed)
            loss.backward()
            tloss += loss.item()
            return loss

        opt.step(make_loss)
        if i % 20 == 0:
            print(i, tloss / 20)
            tloss = 0
            with torch.no_grad():
                show(transformed, i // 20)
        if i % 100 == 0:
            print("SAVE")
            torch.save(transformer.state_dict(), 'style_model.pth')
        i += 1


