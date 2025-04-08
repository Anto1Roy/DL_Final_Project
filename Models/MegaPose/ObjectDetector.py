class ClassAgnosticDetector(nn.Module):
    """A generic object detector that outputs object proposals (bounding boxes)
    and corresponding feature crops.
    """
    def __init__(self):
        super().__init__()
        # TODO: initialize backbone, region proposal network, etc.
    
    def forward(self, image):
        """
        Args:
            image: Input image tensor of shape (B, C, H, W)
        Returns:
            proposals: List of detections per image. Each detection could be a dict with:
                - bbox: bounding box coordinates
                - score: confidence score
                - features: extracted feature crop for that object
        """
        # TODO: implement detection logic.
        # For now, return a dummy proposal for each image in the batch.
        B = image.shape[0]
        proposals = []
        for b in range(B):
            proposals.append([
                {"bbox": torch.tensor([10, 10, 100, 100]), 
                 "score": 0.9,
                 "features": torch.rand(1, 64, 32, 32)}  # dummy feature map
            ])
        return proposals