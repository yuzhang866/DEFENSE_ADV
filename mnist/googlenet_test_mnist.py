import sys
sys.path.append('/home/node/Documents/yxx_code/DEFENSE_ADV2')

import torchvision
import torch as t
from torch import nn
from torch.utils.data import DataLoader
from advertorch.attacks import FGSM
from advertorch.attacks import LinfBasicIterativeAttack
from torchvision import transforms
from model.GoogleNet import GoogleNetMnist


if __name__ == '__main__':
    device = 'cuda' if t.cuda.is_available() else 'cpu'

    test_data = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

    model = GoogleNetMnist
    model.load_state_dict(t.load('./SavedNetworkModel/MNIST/GoogleNet/googlenet_mnist_19.pth', map_location=t.device(device)))
    model.to(device)
    # model.eval()

    epsilons = [.15]
    for eps in epsilons:
       
        #correct = 0
        #for x, y in test_loader:
            #x, y = x.to(device), y.to(device)
            #model.eval()
            #adversary_fgsm = FGSM(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
            #adv_untargeted = adversary_fgsm.perturb(x, y)
            #output = model(adv_untargeted)
            #pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            #correct += pred.eq(y.data.view_as(pred)).cpu().sum()  
        #print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset), 100 * correct / len(test_loader.dataset)))

        # BIM 生成对抗样本
        correct = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            model.eval()
            adversary_BIM = LinfBasicIterativeAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
            adv_untargeted = adversary_BIM.perturb(x, y)
            output = model(adv_untargeted)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()  
        print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset),
                                                             100. * correct / len(test_loader.dataset)))

