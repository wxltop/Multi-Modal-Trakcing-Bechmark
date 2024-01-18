import os
import sys
import torch
import torchvision
from ltr.models.backbone.resnet_vggm import resnet18_vggmconv1
# import pretrainedmodels

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)


# def main():
#     resnet18 = torchvision.models.__dict__['resnet18'](pretrained=True)
#     vggm = pretrainedmodels.models.vggm()
#
#     resnet18.vggmconv1 = vggm.features[0]
#
#     # Convert to RGB
#     weight = resnet18.vggmconv1.weight.clone()
#     resnet18.vggmconv1.weight[:,0,:,:] = weight[:,2,:,:]
#     resnet18.vggmconv1.weight[:,2,:,:] = weight[:,0,:,:]
#
#     # Normalize
#     vgg_mean = torch.Tensor(vggm.mean[::-1])
#     torch_std = torch.Tensor([0.229, 0.224, 0.225]).view(1,-1,1,1)
#     resnet18.vggmconv1.weight[...] = resnet18.vggmconv1.weight * (torch_std*255)
#
#
#     torch.save(resnet18.state_dict(), '../features/pretrained/resnet18_vggmconv1.pth')


def main():
    resnet50 = torchvision.models.__dict__['resnet50'](pretrained=True)
    resnet18_vggm = resnet18_vggmconv1(path='/data/tracking_networks/resnet18_vggmconv1/resnet18_vggmconv1.pth')

    resnet50.vggmconv1 = resnet18_vggm.vggmconv1

    os.makedirs('/data/tracking_networks/resnet50_vggmconv1', exist_ok=True)

    torch.save(resnet50.state_dict(), '/data/tracking_networks/resnet50_vggmconv1/resnet50_vggmconv1.pth')


if __name__ == '__main__':
    main()