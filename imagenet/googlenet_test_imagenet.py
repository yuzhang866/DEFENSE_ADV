import torchvision
import torch as t
from torch import nn
from torch.utils.data import DataLoader
from advertorch.attacks import FGSM, CarliniWagnerL2Attack
from advertorch.attacks import LinfBasicIterativeAttack
from torchvision import transforms, datasets
from torchvision.models import GoogLeNet
import foolbox as fb

'''
The performance of the defense model combined with ResNet101(或50) and GoogLeNet on tiny-imageNet
'''

if __name__ == '__main__':
    dataroot = './dataset/tiny-imagenet-200/'
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    input_size = 224
    batch_size = 64

    test_dataset = datasets.ImageFolder(root=dataroot + 'test', transform=transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = GoogLeNet(num_classes=200)
    model.load_state_dict(
        t.load('./SavedNetworkModel/ImageNet/GoogleNet/googlenet_imagenet_39.pth', map_location=t.device(device)))
    model.to(device)
    model.eval()

    epsilons = [0.0]
    for eps in epsilons:
        
        correct = 0
        for x, y in test_data:
            x, y = x.to(device), y.to(device)
            #model.eval()
            #adversary_fgsm = FGSM(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
            #adv_untargeted = adversary_fgsm.perturb(x, y)
            #output = model(adv_untargeted)
            output = model(x)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            #print(pred, '  ', y)
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()  
        print('FGSM --- epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_data.dataset),
                                                             100. * correct / len(test_data.dataset)))

        '''
        # BIM 生成对抗样本
        correct = 0
        for x, y in test_data:
            x, y = x.to(device), y.to(device)
            model.eval()
            adversary_BIM = LinfBasicIterativeAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps)
            adv_untargeted = adversary_BIM.perturb(x, y)
            output = model(adv_untargeted)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
        print('BIM----epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_data.dataset),
                                                             100. * correct / len(test_data.dataset)))

        # DeepFool 生成对抗样本
        correct = 0
        for x, y in test_data:
            x, y = x.to(device), y.to(device)
            model.eval()
            bounds = (0, 1)
            fmodel = fb.PyTorchModel(model, bounds)
            attack = fb.attacks.LinfDeepFoolAttack()
            raw, adv_untargeted, adv = attack(fmodel, x, y, epsilons=eps)
            output = model(adv_untargeted)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
        print('DeepFool---epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_data.dataset),
                                                             100. * correct / len(test_data.dataset)))

        # C&W 生成对抗样本
        correct = 0
        for x, y in test_data:
            x, y = x.to(device), y.to(device)
            model.eval()
            adversary_cw = CarliniWagnerL2Attack(model, num_classes=200, max_iterations=eps)
            output = model(adversary_cw)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()  
        print('CW---epsilon: {} Accuracy: {}/{} ({:.0f}%)'.format(eps, correct, len(test_data.dataset),
                                                             100. * correct / len(test_data.dataset)))

        '''
