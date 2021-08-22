import torch as t
import torchvision
from advertorch.attacks import FGSM
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from Model.DefenseNet import DefenseNet
from Model.DefenseNet import reduce_precision_np
from Model.GoogleNet import GoogleNetCifar


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

    
    model1 = GoogleNetCifar
    model1.load_state_dict(t.load('../SavedNetworkModel/CIFAR/GoogleNet/googlenet_cifar_39.pth'))
    model2 = t.load('../SavedNetworkModel/CIFAR/GoogleNet/Defense_FGSM/googlenet_defense_fgsm_cifar_39.pth')
    model1.to(device)
    model2.to(device)

    epsilons = [.005]
    for eps in epsilons:
        
        correct = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            model1.eval()
            adversary_fgsm = FGSM(model1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
            adv_untargeted = adversary_fgsm.perturb(x, y)

            # adv_untargeted = adv_untargeted.cpu().detach().numpy()
            # data1 = reduce_precision_np(adv_untargeted, 8)
            # data1 = t.tensor(data1)
            # data1 = data1.to(device)
            #
            # adv_untargeted = adversary_fgsm.perturb(x, y)
            #
            # output = model2(data1)
            output=model1(adv_untargeted)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()  
        print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset), 100 * correct / len(test_loader.dataset)))

        # BIM 生成对抗样本
        # correct = 0
        # for x, y in test_loader:
        #     x, y = x.to(device), y.to(device)
        #     model.eval()
        #     adversary_BIM = LinfBasicIterativeAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
        #     adv_untargeted = adversary_BIM.perturb(x, y)
        #     output = model(adv_untargeted)
        #     pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        #     correct += pred.eq(y.data.view_as(pred)).cpu().sum()  
        # print('epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_loader.dataset),
        #                                                      100. * correct / len(test_loader.dataset)))
