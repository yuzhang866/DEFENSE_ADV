import torch as t
import torchvision.datasets as datasets
from torchvision import transforms
import torch.utils.data as data
from torchvision.models.googlenet import GoogLeNet


if __name__ == '__main__':
    input_size = 224
    batch_size = 256
    learning_rate = 0.001
    epochs = 50
    dataroot = './dataset/tiny-imagenet-200/'

    train_dataset = datasets.ImageFolder(root=dataroot + 'train', transform=transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    train_data = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset = datasets.ImageFolder(root=dataroot + 'test', transform=transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    test_data = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 定义 model、loss、optimizer
    model = GoogLeNet(num_classes=200)
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

    device = 'cuda' if t.cuda.is_available() else 'cpu'
    model.to(device)

   
    for epoch in range(epochs):
        print('*' * 40)
        running_loss = 0.0
        running_acc = 0.0

        
        for i, data in enumerate(train_data, 1):
            model.train()
            img, label = data
            img, label = img.to(device), label.to(device)

            
            out, aux2, aux1 = model(img)
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
               './SavedNetworkModel/ImageNet/GoogleNet/googlenet_imagenet_{}.pth'.format(epoch))

