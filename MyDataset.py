#%%
import os
import glob
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import itertools
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.sub_dir_name = os.listdir(root_dir)
        self.transform = transform
        self.class_file_num = []
        self.path_li = []
        self.labels_encoded = []
        self.labels = []
        self.file_path_li()
        self.make_labels()
        
        
    def file_path_li(self):
        path_li = []
        self.class_file_num = []
        for path in glob.glob(self.root_dir + '/**'):
            file_path = glob.glob(path+'/**')
            self.class_file_num.append(len(file_path))
            path_li.append(file_path)
        self.path_li = list(itertools.chain(*path_li))
        
    def make_labels(self):
        labels = []
        for i, num in enumerate(self.class_file_num):
            labels.append([self.sub_dir_name[i] for j in range(num)])
        self.labels = list(itertools.chain(*labels))

        encoder = LabelEncoder()
        self.labels_encoded = encoder.fit_transform(self.labels)
    def __len__(self):
        return len(self.path_li)
    
    def __getitem__(self, idx):
        img_path = self.path_li[idx]
        img = Image.open(img_path)
        

            
        label = self.labels_encoded[idx]
        # label = self.labels[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        # if img.size()[0] != 3:
        #     img = img[:3]
            
        return img, label


def train_test_split(dataset, test_size = 0.3): #Dataset 객체 train_test split 하는 함수
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    indices = list(range(len(dataset)))
    y_test = [y for _, y in dataset]
    
    for train_index, test_index in sss.split(indices, y_test):
        # print('test:', test_index, 'train:', train_index)
        print(len(train_index), len(test_index))
    
    train_ds = Subset(dataset, train_index)
    test_ds = Subset(dataset, test_index)
    
    return train_ds, test_ds
    
def data_split(dataset, split_size): # torch random_split 사용해서 split 하는 함수
    dataset_size = len(dataset)
    train_size = int(dataset_size * split_size[0])
    validation_size = int(dataset_size * split_size[1])
    test_size = dataset_size - train_size - validation_size
    
    return random_split(dataset, [train_size, validation_size, test_size] )
    
def show_img(data, num = 1):
    img, label = data[0], data[1]
    # print(img[0].size(), img[1])
    
    ax = plt.subplot(4, num//4 + 1, i+1)
    plt.tight_layout()
    ax.set_title('{}'.format(label))
    ax.axis('off')
    if img.shape != 3:
        img = img.permute(1, 2, 0)
    plt.imshow(img)

from collections import Counter
import traceback
def class_ratio(dataset): # 클래스 별 데이터 개수 세줌
    try:
        print(dict(Counter(dataset.labels)))
        print(dict(Counter(dataset.labels_encoded)))
        
    except:
        print(traceback.format_exc())
        label = [y for _, y in dataset]
        print('Count num per classes : ', dict(Counter(label)))
        
#%%
if __name__=="__main__":

    img_size = 224
    batch_size = 32
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(img_size)),
            transforms.CenterCrop(img_size)
        ]
    )

    dataset = MyDataset(root_dir = './images', transform = transform)
    train_ds, val_ds, test_ds = data_split(dataset, [0.7, 0.2, 0.1])

    train_loader = DataLoader(dataset = train_ds, batch_size = batch_size, drop_last = True)
# for images, labels in train_loader:
#     print(images.shape)
    
#%%

#%% 
# if __name__=="__main__":
#     img_size = 224
#     transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Resize(size=(img_size)),
#             transforms.CenterCrop(img_size)
#         ]
#     )
#     dataset = MyDataset(root_dir = './images', transform = transform)
    
#     print(Counter(dataset.labels_encoded))
#     print(Counter(dataset.labels))
    
#     train_loader = DataLoader(dataset = dataset, batch_size = 32, drop_last = True)

#     for epoch in range(1):
#         for images, labels in train_loader:
#             print(images.shape)
    
# #%%
# if __name__=="__main__":

#     x = [x for x, _ in dataset]
        
#     x_ = x.copy()
#     x = np.array(x)
#     # print(x.reshape((-1,x[0])))
#     c_li = []
#     for i, img in enumerate(x):
#         if img.shape[0] != 3:
#             c_li.append(i)
#             print(img.shape)

# #%%

# # for i in c_li:
# #     # print(dataset.path_li[i])
# #     os.remove(dataset.path_li[i])


# #%%
# if __name__=="__main__":

#     train_loader = DataLoader(dataset = dataset, batch_size = 32, drop_last = True)

#     for epoch in range(1):
#         for images, labels in train_loader:
#             print(images.shape)
            
#     # train_ds, test_ds = train_test_split(dataset)
    
# #%%



# #%% torch random split 이용해서 train test split (위에것 보다 훨씬 빠름)
# if __name__=="__main__":
#     dataset = MyDataset(root_dir = './images', transform = transform)

#     train_ds, val_ds, test_ds = data_split(dataset, [0.7, 0.2, 0.1])
    
    
    
# #%%
# if __name__=="__main__":
#     num = 10
#     for i in range(num):
#         show_img(train_ds[i], num)
        

#     # 클래스별 data 개수 세기
#     print(Counter(train_ds.dataset.labels))
#     print(Counter(dataset.labels))
#     print(class_ratio(val_ds))
    
    # for i in range(num):
    #     sample = dataset[i]
    #     print(sample[0].shape, sample[1])
        
    #     ax = plt.subplot(4, num//3, i+1)
    #     plt.tight_layout()
    #     ax.set_title('{}'.format(sample[1]))
    #     ax.axis('off')
    #     plt.imshow(sample[0].permute(1, 2, 0))

#%%
# translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "ragno" : "spider", "dog": "cane", "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}
# print(translate)