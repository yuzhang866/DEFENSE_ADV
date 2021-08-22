import torch as t
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model.LeNet import LeNet
from torch.optim import SGD


if __name__ == '__main__':
    
    learning_rate = 1e-3  
    batch_size = 128  
    epochs = 20  

    
    train_dataset = torchvision.datasets.CIFAR10(root='../dataset',
                                                 train=True,
                                                 transform=transforms.Compose([transforms.ToTensor()]),
                                                 download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    model = LeNet(3, 10).to(device)
    criterion = t.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)

    
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
               '../SavedNetworkModel/CIFAR/LeNet/lenet_cifar_{}.pth'.format(epoch))
