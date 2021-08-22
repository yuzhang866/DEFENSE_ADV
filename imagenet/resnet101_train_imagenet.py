import sys
sys.path.append('/home/node/Documents/yxx_code/DEFENSE_ADV2')

import torch as t
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model.ResNet import ResNet, BottleNeck


# def resnet50(num_classes, channels):
#     return ResNet(BottleNeck, [3, 4, 6, 3], num_classes, channels)


def resnet101(num_classes, channels):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes, channels)

if __name__ == '__main__':
  
    learning_rate = 1e-3  
    batch_size = 10  
    epochs = 40  
    input_size = 224
    dataroot = './dataset/tiny-imagenet-200/'

    train_dataset = datasets.ImageFolder(root=dataroot + 'train', transform=transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.ImageFolder(root=dataroot + 'test', transform=transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # model、loss、optimizer
    model = resnet101(200, 3)
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

    device = 'cuda' if t.cuda.is_available() else 'cpu'
    model.to(device)

    # tainning
    for epoch in range(epochs):
        print('*' * 40)
        running_loss = 0.0
        running_acc = 0.0

        
        for i, data in enumerate(train_data, 1):
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
               './SavedNetworkModel/ImageNet/ResNet101/resnet101_mnist_{}.pth'.format(epoch))
