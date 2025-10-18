import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import string
import requests
import os
import io
import gdown

# ------------------------------
# CONFIGURATION
# ------------------------------
st.set_page_config(page_title="🧠 Handwritten Text Recognition", page_icon="🧠", layout="wide")

MODEL_PATH = "crnn_iam_advanced.pth"
MODEL_URL = "https://drive.google.com/uc?id=1ectKYXIgzwfvWeVaZvb9Na0Y4AvZiwPH"  # Google Drive link

# ------------------------------
# MODEL DOWNLOAD (Drive)
# ------------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("⬇️ Downloading model weights (please wait)..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# ------------------------------
# CHARACTER SETUP
# ------------------------------
CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase + " .,!?'-"

# ------------------------------
# MODEL CLASSES
# ------------------------------
class StrLabelConverter:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.char_to_idx = {c: i + 1 for i, c in enumerate(alphabet)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(alphabet)}

    def decode(self, preds):
        chars, prev = [], 0
        for p in preds:
            if p != prev and p != 0:
                chars.append(self.idx_to_char.get(p, ""))
            prev = p
        return "".join(chars)


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super().__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        r, _ = self.rnn(x)
        T, b, h = r.size()
        o = self.embedding(r.view(T * b, h)).view(T, b, -1)
        return o


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super().__init__()
        ks, ps, ss = [3, 3, 3, 3, 3, 3, 2], [1, 1, 1, 1, 1, 1, 0], [1] * 7
        nm = [64, 128, 256, 256, 512, 512, 512]
        cnn = nn.Sequential()

        def conv_relu(i, bn=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(f"conv{i}", nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if bn:
                cnn.add_module(f"bn{i}", nn.BatchNorm2d(nOut))
            cnn.add_module(f"relu{i}", nn.ReLU(True))

        conv_relu(0)
        cnn.add_module("pool0", nn.MaxPool2d(2, 2))
        conv_relu(1)
        cnn.add_module("pool1", nn.MaxPool2d(2, 2))
        conv_relu(2, True)
        conv_relu(3)
        cnn.add_module("pool2", nn.MaxPool2d((2, 1), (2, 1)))
        conv_relu(4, True)
        conv_relu(5)
        cnn.add_module("pool3", nn.MaxPool2d((2, 1), (2, 1)))
        conv_relu(6, True)
        self.cnn = cnn

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass),
        )

    def forward(self, x):
        c = self.cnn(x)
        b, c_, h, w = c.size()
        c = c.squeeze(2).permute(2, 0, 1)
        return self.rnn(c)


# ------------------------------
# LOAD MODEL
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CRNN(32, 1, len(CHARSET) + 1, 256).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
converter = StrLabelConverter(CHARSET)

# ------------------------------
# IMAGE TRANSFORM
# ------------------------------
def preprocess_image(img):
    tr = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return tr(img).unsqueeze(0)



# ------------------------------
# DEMO IMAGE LINKS (Hosted on GitHub for guaranteed loading)
# ------------------------------
DEMO_IMAGES = {
    "📜 Sample 1": "https://raw.githubusercontent.com/Solankys123/handwritten-text-recognition-project/main/demo_imagess/sample1.png",
    "🖋️ Sample 2": "https://raw.githubusercontent.com/Solankys123/handwritten-text-recognition-project/main/demo_imagess/sample2.png",
    "✍️ Sample 3": "https://raw.githubusercontent.com/Solankys123/handwritten-text-recognition-project/main/demo_imagess/sample3.png",
    "📖 Sample 4": "https://raw.githubusercontent.com/Solankys123/handwritten-text-recognition-project/main/demo_imagess/sample4.png",
    "📘 Sample 5": "https://raw.githubusercontent.com/Solankys123/handwritten-text-recognition-project/main/demo_imagess/sample5.png",
}


# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("🧠 Handwritten Text Recognition (CRNN)")
st.markdown("### Upload an image or choose a demo sample below 👇")

# Sidebar for demo selection
st.sidebar.header("📚 Demo Images")
demo_choice = st.sidebar.selectbox("Select a sample to test", ["None"] + list(DEMO_IMAGES.keys()))

uploaded_file = st.file_uploader("📤 Upload a Handwritten Image", type=["png", "jpg", "jpeg"])

# Image selection priority
if demo_choice != "None":
    img_url = DEMO_IMAGES[demo_choice]
    try:
        response = requests.get(img_url, timeout=10)
        img_bytes = io.BytesIO(response.content)
        img = Image.open(img_bytes).convert("L")
        st.image(img, caption=f"🖼️ {demo_choice} (Demo)", use_container_width=True)
    except Exception as e:
        st.error("⚠️ Unable to load demo image. Please try another sample.")
        st.stop()
elif uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)
else:
    st.info("📸 Upload your own image or select one from the sidebar.")
    st.stop()

# Prediction
with torch.no_grad():
    inp = preprocess_image(img).to(device)
    preds = model(inp)
    _, p = preds.max(2)
    p = p.transpose(1, 0).contiguous().view(-1)
    text = converter.decode(p.cpu().numpy().tolist()).strip().upper()

st.markdown("---")
st.subheader("🧾 Predicted Text:")
st.success(f"**{text}**")

st.markdown("---")
st.caption("🤖 Model: CRNN (CNN + BiLSTM + CTC) | Dataset: IAM | Built with ❤️ by Sandeep Solanki")
