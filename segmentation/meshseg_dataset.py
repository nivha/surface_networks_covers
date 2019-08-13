import os
from scipy.io import loadmat
import numpy as np
# from imageio import imread
import torch.utils.data as data
import torch


IMG_EXTENSIONS = ['.mat']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.startswith('.'): continue
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def img_loader(imgfile):
    arr = loadmat(imgfile)['data']

    if np.any(np.isnan(arr)):
        print('Warning: img', imgfile, 'contains nan')

    # implicitly assume this is an image (and not target....)
    if len(arr.shape) == 3:
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr).type(torch.get_default_dtype())
    else:
        # matlab 1-base indexing --> python 0-base indexing (!!!)
        arr = arr - 1
        return torch.from_numpy(arr).long()


class MeshSeg(data.Dataset):
    def __init__(self, images_path, split='train', joint_transform=None, transform=None, loader=img_loader):

        self.images_path = images_path
        assert split in ('train', 'val', 'test')
        self.split = split

        self.transform = transform
        self.joint_transform = joint_transform  # e.g. flipping..
        self.loader = loader

        self.imgs = _make_dataset(images_path)

    def __getitem__(self, index):
        path = self.imgs[index]
        try:
            img = self.loader(path)
            target = self.loader(path.replace('images', 'labels'))
        except ValueError:
            print('Warning: problem reading image file', path)
            return None

        if self.joint_transform is not None:
            img, target = self.joint_transform([img, target])

        if self.transform is not None:
            img = self.transform(img)

        # target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

