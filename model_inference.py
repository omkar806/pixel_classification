from train import TreeClassifierCNN
import torch
from PIL import Image
import torchvision.transforms as transforms


def load_model(model_path, device='cpu'):
    """Load the trained model"""
    model = TreeClassifierCNN(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    return model

def predict_image(model, image_path, transform, device='cpu'):
    """Predict the class of a single image"""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    class_labels = ['GROUND', 'LIVE', 'DEAD']
    predicted_class = class_labels[predicted.item()]

    return predicted_class

def setup_device():
    """
    Set up the appropriate device for training on Mac.
    Returns the device and prints its properties.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders)")
        return 'mps'
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
        return 'cpu'
    return None

# Load the trained model
device = setup_device()
model = load_model('tree_classifier.pth', device)

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Perform inference
#This take 5x5 grid as input of the 512x512 image.
image_path = "/path/to/your/test_image.jpg"
prediction = predict_image(model, image_path, transform, device)
print(f"Predicted class: {prediction}")
