import torchvision
import torch
from PIL import Image
from torchvision import transforms
from utils.helper import show


class DeepLabV3Segmentation:
    def __init__(self, weights):
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.to(self.device)
        else:
            self.device = torch.device('cpu')

    def segment(self, image_path):
        input_image = Image.open(image_path)
        input_image = input_image.convert("RGB")
        input_tensor = self.preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_batch.to(self.device))['out'][0]

        return (output.argmax(0) == 15).cpu().numpy().astype('uint8')*255


def main():
    # example usage
    segmenter = DeepLabV3Segmentation(torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
    img_path = 'Data/test.jpg'
    output_predictions = segmenter.segment(img_path)
    # img = cv2.imread(img_path)
    show(output_predictions)

if __name__ == '__main__':
    main()