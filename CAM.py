
        


# %
import cv2
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchsummaryX import summary
from torch.utils.data import DataLoader

from MyDataset import *
from MyModel import *

batch_size = 64
num_epochs = 10
img_size = 224
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(img_size)),
        transforms.CenterCrop(img_size)
    ]
)

dataset = MyDataset(root_dir = '../images', transform = transform)
train_ds, val_ds, test_ds = data_split(dataset, [0.7, 0.2, 0.1])

train_loader = DataLoader(dataset = train_ds, batch_size = batch_size, drop_last = True)
# for images, labels in train_loader:
#     print(images.shape)

#%%
print(Counter(dataset.label_encoded))
print(class_ratio(val_ds))



#%%
# print(Counter(dataset.labels))
print(class_ratio(dataset))
print(class_ratio(val_ds))
# print(Counter(dataset.labels_encoded))


# print(model)


# %
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%%
batch_size = 64
num_epochs = 10
#%%
model = CAM_Net(batch_size)
model = model.to(device)
summary(model,torch.zeros((batch_size,3,224,224)).to(device))

trainable_parameters = []
for name, p in model.named_parameters():
    print(name)
    if "fc" in name:
        trainable_parameters.append(p)
        
optimizer = torch.optim.SGD(params=trainable_parameters, lr=0.1, momentum=1e-5)  
criterion = nn.CrossEntropyLoss()
#%%

total_step = len(train_loader)
loss_list = []
acc_list = []

for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device) 
        # Run the forward pass
        # print('True label shape : ', labels.shape)
        # print(images.shape)
        outputs = model(images)
        # print(outputs.shape)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

    print(f'Epoch: {epoch:>4d}\tLoss: {total_loss / len(labels):.5f}')
        # if (i + 1) % 100 == 0:
        #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
        #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
        #                   (correct / total) * 100))
#%% TEST
PATH = './weights/'

torch.save(model, PATH + 'model.pt')  # 전체 모델 저장
torch.save(model.state_dict(), PATH + 'model_state_dict.pt')  # 모델 객체의 state_dict 저장
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()
}, PATH + 'all.tar') 

# %
test_batch_size = 1
model_test = CAM_Net(test_batch_size)
model_test.load_state_dict(torch.load('.\weights\model_state_dict.pt'))
model_test.eval()
summary(model_test,torch.zeros((test_batch_size,3,224,224)))

#%%
params = list(model_test.named_parameters())
for name, param in params:
    if name == 'fc.weight':
        weight = param.data.numpy()
        
print(weight.shape)
# print(params[0][1].size())
# np.squeeze(params[-1].data.numpy()).shape
# weight = np.squeeze(params[-1].data.numpy())


#%%
def return_CAM(feature_conv, weight, class_idx):
    # generate the class -activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for i, idx in enumerate(class_idx):
        beforeDot =  feature_conv[i].reshape((nc, h*w))
        cam = np.matmul(weight[idx], beforeDot)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam
#%%

test_loader = DataLoader(dataset = test_ds, batch_size = test_batch_size, drop_last = True, shuffle = True)


for images, labels in test_loader:
    logit = model_test(images)
    print(logit.shape)
    
    # h_x = F.softmax(logit, dim =1).data.squeeze()
    # print(logit.shape)
    # print(torch.sum(logit,axis = 1))
    pred =torch.argmax(logit, dim = 1)
    print('target : ', labels, 'predict : ', pred)
    
    features_blobs = model_test.vgg_sub(images).data.numpy()
    print(features_blobs.shape)
    CAMs = return_CAM(features_blobs, weight, pred)
    
    img = np.squeeze(images).permute(1,2,0)
    height, width, _ = img.shape
    heatmap_BGR = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    heatmap_RGB = cv2.cvtColor(heatmap_BGR, cv2.COLOR_BGR2RGB)
    # result = heatmap * 0.5 + img * 0.5
    plt.imshow(img)
    plt.imshow(heatmap_RGB, alpha=0.7)
    break
#%%







# %
# 모델의 state_dict 출력
print("Model's state_dict:")
for param_tensor in vgg16.state_dict():
    print(param_tensor, "\t", vgg16.state_dict()[param_tensor].size())

print()

# 옵티마이저의 state_dict 출력
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])



