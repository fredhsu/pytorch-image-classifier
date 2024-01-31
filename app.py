import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# setting up cuda

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# creating model


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 36, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(36, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Loading the model
PATH = "./cifar_net.pth"
net = Net()
# net.to(device)
net.load_state_dict(torch.load(PATH))

net.eval()

transform = transforms.Compose(  # Composes several transforms together
    [
        transforms.Resize(
            (32, 32)
        ),  # resize the image to fit the cifar model which was trained on 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)  # The first tuple is a sequence of means, the second std deviations

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

file_up = st.file_uploader("Upload an image", type="jpg")

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    # image = transform(image)
    image = transform(image).unsqueeze(0)

    outputs = net(image)
    _, predicted = torch.max(outputs, 1)
    st.write(f"predicted : {classes[predicted[0]]}")
