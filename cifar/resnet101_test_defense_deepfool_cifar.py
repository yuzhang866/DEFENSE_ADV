import torch as t
import torchvision
from advertorch.attacks import FGSM
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from Model.DefenseNet import DefenseNet
from Model.DefenseNet import reduce_precision_np
from Model.GoogleNet import GoogleNetCifar
import foolbox as fb
from Model.ResNet import ResNet, BottleNeck


def resnet101(num_classes, channels):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes, channels)


if __name__ == '__main__':
    
    learning_rate = 1e-3  
    batch_size = 128  
    epochs = 20  
    t.manual_seed(0)
    use_cuda = t.cuda.is_available()
    device = t.device("cuda" if use_cuda else "cpu")

    
    test_dataset = torchvision.datasets.CIFAR10(root='../dataset',
                                                train=False,
                                                transform=transforms.Compose([transforms.ToTensor()]),
                                                download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

   
    model1 = resnet101(10, 3)
    model1.load_state_dict(t.load('../SavedNetworkModel/CIFAR/ResNet101/resnet101_cifar_19.pth'))
    model2 = DefenseNet()
    model2.load_state_dict(t.load('../SavedNetworkModel/CIFAR/ResNet101/defense_DeepFool/resnet101_defense_deepfool_cifar_47.pth'))
    model1.to(device)
    model2.to(device)
    count = 0
    epsilons = [.005]
    for eps in epsilons:
        bounds = (0, 1)
        fmodel = fb.PyTorchModel(model1, bounds)
        attack = fb.attacks.LinfDeepFoolAttack()
        
        correct = 0
        for img in test_loader:
            print(" count == ", count)
            count += 1

            data, label = img
            data, label = data.to(device), label.to(device)
            raw, adv_untargeted, adv = attack(fmodel, data, label, epsilons=0.005)
            # clean dataset
            # adv_untargeted = adv_untargeted.cpu().detach().numpy()
            # data1 = reduce_precision_np(adv_untargeted, 8)
            # data1 = t.tensor(data1)
            # data1 = data1.to(device)
            #
            # output = model2(data1)
            output2 = model1(adv_untargeted)
            pred = output2.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()  
        print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset), 100 * correct / len(test_loader.dataset)))
