import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import os
import io

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)

def tensor_to_image(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    return transforms.ToPILImage()(image)

# Preset style images
PRESET_STYLES = {
    "Starry Night": "styles/starry_night.jpg",
    "The Scream": "styles/the_scream.jpg",
    "Candy": "styles/candy.jpg",
    "Mosaic": "styles/mosaic.jpg",
    "Ghibli Style": "styles/ghibli.jpg",
    "Van Gogh": "styles/van_gogh.jpg",
    "Picasso": "styles/picasso.jpg"
}

# Content and Style Loss classes
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

    @staticmethod
    def gram_matrix(input):
        b, c, h, w = input.size()
        features = input.view(c, h * w)
        G = torch.mm(features, features.t())
        return G.div(c * h * w)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# Style Transfer Function
def run_style_transfer(content_img, style_img, style_weight, content_weight):
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    input_img = content_img.clone()
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            break
    model = model[:j + 1]

    optimizer = optim.LBFGS([input_img.requires_grad_()])
    run = [0]

    while run[0] <= 100:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()
            run[0] += 1
            return loss
        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)
    return tensor_to_image(input_img)

# Streamlit UI
st.set_page_config(page_title="Neural Style Transfer", layout="centered", page_icon="🎨")

st.markdown("<h1 style='text-align: center;'>🎨 Neural Style Transfer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Apply artistic styles to your photos using PyTorch and VGG19!</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    content_image_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if content_image_file:
        content_img = Image.open(content_image_file)
        st.image(content_img, caption="Content Image Preview", use_container_width=True)

with col2:
    style_choice = st.selectbox("Choose Preset Style", list(PRESET_STYLES.keys()))
    style_preview = Image.open(PRESET_STYLES[style_choice])
    st.image(style_preview, caption=f"{style_choice} Style Preview", use_container_width=True)

style_weight = st.slider("Style Weight", 1e5, 1e7, 1e6, step=1e5)
content_weight = st.slider("Content Weight", 0.1, 5.0, 1.0, step=0.1)

if st.button("✨ Apply Style Transfer"):
    if content_image_file is None:
        st.warning("Please upload a content image first.")
    else:
        content_tensor = load_image(content_image_file)
        style_tensor = load_image(PRESET_STYLES[style_choice])
        with st.spinner("Processing..."):
            result = run_style_transfer(content_tensor, style_tensor, style_weight, content_weight)
        st.success("Style transfer complete!")
        st.image(result, caption="Stylized Output", use_container_width=True)

        # Save history and display
        if "styled_images" not in st.session_state:
            st.session_state.styled_images = []

        st.session_state.styled_images.append(result)

        # Download Button
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("Download Image", byte_im, "styled_image.png", mime="image/png")

        # History Tracking
        st.subheader("Styled Image History")
        for idx, img in enumerate(st.session_state.styled_images):
            st.image(img, caption=f"Styled Image {idx + 1}", use_container_width=True)

