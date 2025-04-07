import torchvision
from torchvision.transforms import functional as TF
import torch
import torch.nn as nn

class MaskRCNNDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes)

    def forward(self, image):
        # expects image to be normalized and batched
        self.model.eval()
        with torch.no_grad():
            preds = self.model(image)[0]  # single image
        return preds
