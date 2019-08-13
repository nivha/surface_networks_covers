import os
from scipy.io import loadmat
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
    try:
        matfile = loadmat(imgfile)
        data = matfile['data']
        target = matfile['target']
    except:
        Exception('Error reading matfile:', imgfile)

    # implicitly assume this is an image (and not target....)
    data = data.transpose(2, 0, 1)
    data = torch.from_numpy(data).type(torch.get_default_dtype())
    target = target[0]
    target = torch.from_numpy(target).long()
    target = target.item()

    return data, target


def img_loader_test(imgfile):

    matfile = loadmat(imgfile)
    data = matfile['data']
    file_name = matfile['file_name'][0]

    # implicitly assume this is an image (and not target....)
    data = data.transpose(0, 3, 1, 2)
    data = torch.from_numpy(data).type(torch.get_default_dtype())

    return data, file_name


class SphereSet(data.Dataset):

    def __init__(self, images_path, split='train', loader=img_loader):

        self.images_path = images_path
        assert split in ('train', 'val', 'test')
        self.split = split

        if split == 'train':
            self.loader = loader
        else:
            self.loader = img_loader_test

        self.imgs = _make_dataset(images_path)

    def __getitem__(self, index):
        path = self.imgs[index]
        # try:
        img, target = self.loader(path)
        # except ValueError:
        #     print('Warning: problem reading image file', path)
        #     return None

        return img, target

    def __len__(self):
        return len(self.imgs)
