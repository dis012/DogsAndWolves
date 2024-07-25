import argparse
from scripts.train import train_model
from scripts.predict import predict_images

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate or predict using DogsAndWolvesClassifier.')
    parser.add_argument('mode', choices=['train', 'predict'], help='Mode to run the script in.')
    parser.add_argument('--data_path', type=str, default='data/', help='Path to the dataset')
    parser.add_argument('--image_path', type=str, default='newImages/', help='Path to the directory containing images for prediction')
    parser.add_argument('--weights_path', type=str, default='weights/old/final_model_weights.pth', help='Path to the model weights')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.mode == 'train':
        train_model(data_path=args.data_path)
    elif args.mode == 'predict':
        if not args.image_path:
            raise ValueError('Please provide --image_path for prediction')
        predict_images(path=args.image_path, weights_path=args.weights_path)
    else:
        raise ValueError('Invalid mode. Please choose either "train" or "predict".')

if __name__ == '__main__':
    main()