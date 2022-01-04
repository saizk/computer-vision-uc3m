import numpy as np
from torchvision import transforms


class CropByMask(object):
    """Crop the image using the lesion mask.

    Args:
        border (tuple or int): Border surrounding the mask. We dilate the mask as the skin surrounding
        the lesion is important for dermatologists.
        If it is a tuple, then it is (bordery,borderx)
    """

    def __init__(self, border):
        assert isinstance(border, (int, tuple))
        if isinstance(border, int):
            self.border = (border, border)
        else:
            self.border = border

    def __call__(self, image, mask):

        h, w = image.size[:2]

        # Calculamos los índices del bounding box para hacer el cropping
        sidx = np.nonzero(mask)
        minx = np.maximum(sidx[1].min() - self.border[1], 0)
        maxx = np.minimum(sidx[1].max() + 1 + self.border[1], w)
        miny = np.maximum(sidx[0].min() - self.border[0], 0)
        maxy = np.minimum(sidx[0].max() + 1 + self.border[1], h)

        image = image.crop([minx, maxy, maxx, miny])

        return image


def get_train_transform():
    return transforms.Compose([
        CropByMask((25, 25)),
        transforms.CenterCrop(224),
        transforms.Resize((256, 256)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
