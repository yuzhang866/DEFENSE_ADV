import sys
sys.path.append('/home/node/Documents/yxx_code/DEFENSE_ADV2')

import torch as t
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model.ResNet import ResNet, BottleNeck


def resnet50(num_classes, channels):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes, channels)


# def resnet101(num_classes, channels):
#     return ResNet(BottleNeck, [3, 4, 23, 3], num_classes, channels)

if __name__ == '__main__':
    
    learning_rate = 1e-3  
    batch_size = 128 
    epochs = 60 

    
    train_dataset = torchvision.datasets.CIFAR10(root='./dataset',
                                                      train=True,
                                                      transform=transforms.Compose([transforms.ToTensor()]),
                                                      download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    
    model = resnet50(10, 3)
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

    device = 'cuda' if t.cuda.is_available() else 'cpu'
    model.to(device)

    
    for epoch in range(epochs):
        print('*' * 40)
        running_loss = 0.0
        running_acc = 0.0

        
        for i, data in enumerate(train_loader, 1):
            model.train()
            img, label = data
            img, label = img.to(device), label.to(device)

           
            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.item() * label.size(0)
            _, pred = t.max(out, 1)
            num_correct = (pred == label).sum()
            accuracy = (pred == label).float().mean()
            running_acc += num_correct.item()

           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Finish  {}  Loss: {:.6f}, Acc: {:.6f}'
              .format(epoch + 1, running_loss / (len(train_dataset)),
                      running_acc / (len(train_dataset))))
        
        t.save(model.state_dict(),
               './SavedNetworkModel/CIFAR/ResNet50/resnet50_cifar_{}.pth'.format(i))
