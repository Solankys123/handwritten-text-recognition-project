
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import string, os

CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase + " .,!?'-"

class StrLabelConverter:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.char_to_idx = {c: i+1 for i,c in enumerate(alphabet)}
        self.idx_to_char = {i+1: c for i,c in enumerate(alphabet)}
    def decode(self, preds):
        chars, prev = [], 0
        for p in preds:
            if p != prev and p != 0:
                chars.append(self.idx_to_char.get(p, ''))
            prev = p
        return ''.join(chars)

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super().__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden*2, nOut)
    def forward(self, x):
        r, _ = self.rnn(x)
        T, b, h = r.size()
        o = self.embedding(r.view(T*b, h)).view(T, b, -1)
        return o

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super().__init__()
        ks, ps, ss = [3,3,3,3,3,3,2],[1,1,1,1,1,1,0],[1]*7
        nm = [64,128,256,256,512,512,512]
        cnn = nn.Sequential()
        def conv_relu(i, bn=False):
            nIn = nc if i==0 else nm[i-1]
            nOut = nm[i]
            cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if bn: cnn.add_module(f'bn{i}', nn.BatchNorm2d(nOut))
            cnn.add_module(f'relu{i}', nn.ReLU(True))
        conv_relu(0); cnn.add_module('pool0', nn.MaxPool2d(2,2))
        conv_relu(1); cnn.add_module('pool1', nn.MaxPool2d(2,2))
        conv_relu(2, True); conv_relu(3)
        cnn.add_module('pool2', nn.MaxPool2d((2,1),(2,1)))
        conv_relu(4, True); conv_relu(5)
        cnn.add_module('pool3', nn.MaxPool2d((2,1),(2,1)))
        conv_relu(6, True)
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )
    def forward(self, x):
        c = self.cnn(x)
        b, c_, h, w = c.size()
        c = c.squeeze(2).permute(2, 0, 1)
        return self.rnn(c)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CRNN(32,1,len(CHARSET)+1,256).to(device)
model.load_state_dict(torch.load("/kaggle/working/crnn_iam_advanced.pth", map_location=device))
model.eval()
converter = StrLabelConverter(CHARSET)

st.set_page_config(page_title="Handwritten Text Recognition", page_icon="🧠")
st.title("🧠 Handwritten Text Recognition (CRNN)")
st.write("Upload any handwritten image and get recognized text instantly!")

uploaded_file = st.file_uploader("📤 Upload Image", type=["png","jpg","jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    tr = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((32,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    inp = tr(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(inp)
        _, p = preds.max(2)
        p = p.transpose(1,0).contiguous().view(-1)
        text = converter.decode(p.cpu().numpy().tolist()).upper()
    st.success(f"🧾 Predicted Text: {text}")
