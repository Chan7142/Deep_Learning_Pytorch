import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #GPU 장치 확인

train_dataset = torchvision.datasets.FashionMNIST(root='C:\\Users\\LeeSeungChan\\Desktop\\Chan 제작\\DL\\data\\Ch05', download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = torchvision.datasets.FashionMNIST(root='C:\\Users\\LeeSeungChan\\Desktop\\Chan 제작\\DL\\data\\Ch05', download=True, transform=transforms.Compose([transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100)

labels_map = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

fig = plt.figure(figsize=(10, 10));
columns = 4;
rows = 5;
for i in range(1, columns * rows + 1):
    img_xy = np.random.randint(len(train_dataset))
    imp = train_dataset[img_xy][0][0,:,:]
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[train_dataset[img_xy][1]])
    plt.axis('off')
    plt.imshow(imp, cmap='gray')
plt.show()

#심층 신경망 (합성곱x)

# class FashionDNN(nn.Module):
#     def __init__(self):
#         super(FashionDNN, self).__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.drop = nn.Dropout(0.25)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = x.view(-1, 784)
#         x = F.relu(self.fc1(x))
#         x = self.drop(x)
#         x = F.relu(self.fc2(x))
#         x = self.drop(x)
#         x = self.fc3(x)
#         return x

# #파라미터
# learning_rate = 0.001;
# model = FashionDNN();
# model.to(device)

# criterion = nn.CrossEntropyLoss(); # 분류 문제에서 사용하는 손실 함수
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate); # 최적화 함수
# print(model)

# #학습
# num_epochs = 5; # 학습 반복 횟수
# count = 0; # 학습 횟수 카운트
# loss_list = [] # 손실 값 기록 리스트
# iteration_list = [] # 학습 반복 횟수 기록 리스트
# accuracy_list = [] # 정확도 기록 리스트

# predictions_list = [] # 예측 값 기록 리스트
# labels_list = [] # 실제 값 기록 리스트

# for epoch in range(num_epochs):
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
        
#         # 순전파
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # 역전파 및 최적화
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         count += 1
#         if not (count % 50): #count를 50으로 나누었을때 나머지가 0이 아니라면 실행
#             total = 0
#             correct = 0
#             for images, labels in test_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 labels_list.append(labels)
#                 test = Variable(images.view(100, 1, 28,28))
#                 outputs = model(test)
#                 predictions = torch.max(outputs.data, 1)[1].to(device)
#                 predictions_list.append(predictions)
#                 correct += (predictions == labels).sum().item()
#                 total += len(labels)
#             accuracy = correct / total
#             loss_list.append(loss.item())
#             iteration_list.append(count)
#             accuracy_list.append(accuracy)
#         if not (count % 500):
#             print(f'Epoch [{epoch+1}/{num_epochs}], Step [{count}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

#합성곱

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(64*7*7, 600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 10) # 최종 출력 클래스 수 10

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = FashionCNN().to(device)

criterion = nn.CrossEntropyLoss();
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)

#학습
num_epochs = 5; # 학습 반복 횟수
count = 0; # 학습 횟수 카운트
loss_list = [] # 손실 값 기록 리스트
iteration_list = [] # 학습 반복 횟수 기록 리스트
accuracy_list = [] # 정확도 기록 리스트

predictions_list = [] # 예측 값 기록 리스트
labels_list = [] # 실제 값 기록 리스트

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        train = Variable(images.view(images.size(0), 1, 28, 28))
        labels = Variable(labels)
        
        # 순전파
        outputs = model(train)
        loss = criterion(outputs, labels)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count += 1
        if not (count % 50): #count를 50으로 나누었을때 나머지가 0이 아니라면 실행
            total = 0
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)
                test = Variable(images.view(images.size(0), 1, 28,28))
                outputs = model(test)
                predictions = torch.max(outputs.data, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum().item()
                total += len(labels)
            accuracy = correct / total
            loss_list.append(loss.item())
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if not (count % 500):
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{count}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')