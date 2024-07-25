import argparse
import torch
from PIL import Image
from torchvision import transforms
from models.cnn import CNN
from models.custom_dataset import DogsAndWolvesDataset
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

def predict_images(path, weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4633, 0.4484, 0.3953], std=[0.4633, 0.4484, 0.3953])
    ])

    for file in os.listdir(path):
        image = Image.open(path + file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        model_for_eval = CNN().to(device)
        model_for_eval.load_state_dict(torch.load(weights_path, map_location=device))
        model_for_eval.eval()

        with torch.no_grad():
            outputs = model_for_eval(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidance, predicted_class = torch.max(probs, 1) # Returns the maximum value and index of the maximum value
        
        classes = ['dog', 'wolf']
        predicted_label = classes[predicted_class.item()]
        confidance_score = confidance.item()

        # Plot the images and predicted labels
        plt.imshow(image)
        plt.title(f'The image is a {predicted_label} with a confidance of {confidance_score:.2f}')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the class of a images using the trained CNN.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the custom image')
    parser.add_argument('--weights_path', type=str, default='final_model_weights.pth', help='Path to the model weights')
    args = parser.parse_args()

    predict_images(image_path=args.image_path, weights_path=args.weights_path)