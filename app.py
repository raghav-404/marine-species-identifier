import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn

# ---------------- CONFIG ----------------
MODEL_PATH = "best_model.pth"
CLASS_NAMES = ["Dolphin", "Octopus", "Seals", "Seahorse", "Seaturtles"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):   # fixed typo: __init__
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = SimpleCNN(len(CLASS_NAMES)).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ---------------- IMAGE PREPROCESS ----------------
def predict_image(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_t)
        _, pred = torch.max(output, 1)
    return CLASS_NAMES[pred.item()]

# ---------------- STREAMLIT APP ----------------
st.title("🐠 Marine Species Identifier")
st.write("Upload an underwater image, and the CNN model will predict the species.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Predicting...")
    pred_class = predict_image(image)

    # --- Fix swapped labels ---
    if pred_class == "Seals":
        pred_class = "Seahorse"
    elif pred_class == "Seahorse":
        pred_class = "Seals"

    st.success(f"Predicted Species: *{pred_class}*")
