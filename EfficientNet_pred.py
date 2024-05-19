import argparse
import torch
import timm
from functions import distribute_images_pred_efficientnet
from torchvision import transforms


def load_best_model(model_path, model_name='efficientnet_b1', num_classes=3, device='cpu'):
    preprocess = transforms.Compose([
        transforms.Resize(240),  # Измените размер на 240x240 для EfficientNet-B1
        transforms.CenterCrop(240),
        transforms.ToTensor()
    ])

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, preprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify images and distribute them into folders.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder with images.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder to save sorted images.")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to the saved model.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run the model on.")

    args = parser.parse_args()

    # Class names corresponding to the indices
    class_names = ["Кабарга", "Косуля", "Олень", "не найденно"]
    model, preprocess = load_best_model(args.model_path)

    distribute_images_pred_efficientnet(args.input_folder, args.output_folder, model, preprocess, class_names, args.device)