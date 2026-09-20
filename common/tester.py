from task4.data.cifar10 import cifar10_loaders
from task4.models.resnet_cifar import CIFARResNet18

train_loader,val_loader,test_loader= cifar10_loaders(
    data_root="/content/data",
    split_path="/content/atml_PA1/splits/cifar10_seed6304.json",
    method="vanilla"
)

model= CIFARResNet18()

images,labels= next(iter(train_loader))

print("Images:",images.shape)
print("Labels:",labels.shape)

logits= model(images)

print("Logits:",logits.shape)

print("Train size:",len(train_loader.dataset))
print("Val size:",len(val_loader.dataset))
print("Test size:",len(test_loader.dataset))