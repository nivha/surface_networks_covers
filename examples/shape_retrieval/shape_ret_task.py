"""
This example file shows the process used in the shape retrievl task, to generate flattened images 
from the SHREC17 dataset, train a model (neural network) on the flattened images, 
and make evaluate the test set.

The example here uses only small batch of meshes (the full pipeline used for the paper was done by distributing
jobs on multiple cores).
 The evaluation below is made by using the weights of the trained model used in our paper. 

In order to make this pipeline run on real data, one has to download the whole dataset,
then change the paths and parameters accordingly
"""

import os
import sys
from scipy.io import savemat
import subprocess
import torch
import torchvision
import torch.nn as nn
import numpy as np
import shutil
import requests
import zipfile

from spherical.data_generation.dataset import Shrec17, CacheNPY, ToMesh, ProjectOnSphere
from spherical.dataloader import SphereSet
import spherical.training_spherical as train_utils
from spherical.inception import Inception3


###############################################################################
# General Parameters                                                          #
###############################################################################
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
gptoolbox_path = '../../gptoolbox-master'
matlab_path = 'matlab2017a'


# set cuda GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

###############################################################################
# General Functions                                                           #
###############################################################################


def flatten_images(data_dir, split, n_augment=1, n_features=6):
    # prepare matlab aggregate_predictions script's parameters
    d = {
        'matlab_path': matlab_path,
        'script_path': '../../matlab/api',
        'gptoolbox_path': gptoolbox_path,
        'batches_dir': os.path.join(data_dir, split, 'sphere_imgs'),
        'n_augment': n_augment,
        'n_features': n_features,
        'dst_dir': os.path.join(data_dir, split, 'images'),
    }

    if split == 'train':
        matlab_cmd = "cd(\'{script_path}\');" \
                     "shape_ret_flatten_train(\'{gptoolbox_path}\', \'{batches_dir}\', \'{dst_dir}\');" \
                     "exit;".format(**d)
    elif split == 'test':
        matlab_cmd = "cd(\'{script_path}\');" \
                     "shape_ret_flatten_test(\'{gptoolbox_path}\', \'{batches_dir}\', \'{dst_dir}\', " \
                     "{n_augment}, {n_features});" \
                     "exit;".format(**d)
    else:
        raise Exception('unknown split', split)

    # call matlab aggregate_predictions script
    cmd = [matlab_path, '-nodisplay', '-nosplash', '-nodesktop', '-r', matlab_cmd]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in iter(process.stdout.readline, b''):
        sys.stdout.write(line.decode(sys.stdout.encoding))


###############################################################################
# Generate Data                                                               #
###############################################################################

def generate_train_data(meshes_dir, dataset, epoch, batch_size, num_workers, augmentation=1,
                        perturbed=True, include_loc=False, loc_only=False):
    sphere_img_dir = os.path.join(DATA_DIR, 'train', 'sphere_imgs')
    os.makedirs(sphere_img_dir, exist_ok=True)

    sphere_grid_path = '../../spherical/data_generation/ours_sphere_grid.mat'

    bw = None
    random_rotations = True if perturbed else False
    random_translation = 0.1 if perturbed else 0

    transform = CacheNPY(prefix="b{}_".format(bw), repeat=augmentation, transform=torchvision.transforms.Compose(
        [
            ToMesh(random_rotations=random_rotations, random_translation=random_translation),
            ProjectOnSphere(bandwidth=bw, include_loc=include_loc, loc_only=loc_only, sphere_grid_path=sphere_grid_path)
        ]
    ))

    def target_transform(x):
        classes = ['02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884', '02843684', '02871439', '02876657',
                   '02880940', '02924116', '02933112', '02942699', '02946921', '02954340', '02958343', '02992529', '03001627', '03046257',
                   '03085013', '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137', '03593526', '03624134',
                   '03636649', '03642806', '03691459', '03710193', '03759954', '03761084', '03790512', '03797390', '03928116', '03938244',
                   '03948459', '03991062', '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267', '04379243',
                   '04401088', '04460130', '04468005', '04530566', '04554684']
        return classes.index(x[0])

    train_set = Shrec17(meshes_dir, dataset, perturbed=perturbed, download=True, transform=transform, target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    for batch_idx, (data, target) in enumerate(train_loader):
        fpath = os.path.join(sphere_img_dir, '%s_%s' % (str(epoch), str(batch_idx)))
        savemat(fpath, mdict={'data': data.numpy(), 'target': target.numpy()})


###############################################################################
# Train                                                                       #
###############################################################################

def train(data_dir):
    # hyperparameters
    lr = 0.5
    lr_decay = 0.995
    batch_size = 2
    n_epochs = 10
    model = Inception3(in_channels=6, num_classes=55, aux_logits=True).cuda()

    # set train and validation sets
    trn_dir = os.path.join(data_dir, 'train', 'images')
    print('loading train set from:', trn_dir)
    train_dset = SphereSet(trn_dir, 'train')
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Kaiming init
    model.apply(train_utils.weights_init)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(1, n_epochs + 1):
        # train
        trn_loss, trn_err = train_utils.train(model, train_loader, optimizer, criterion, epoch)
        print('Epoch {:d}: Train - Loss: {:.7f}, Acc: {:.7f}'.format(epoch, trn_loss, 1 - trn_err))
        # adjust lr
        train_utils.adjust_learning_rate(lr, lr_decay, optimizer, epoch, every_n_epochs=1)


###############################################################################
# Predict                                                                     #
###############################################################################

class KeepName:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, file_name):
        return file_name, self.transform(file_name)


def generate_test_data(meshes_dir, epoch, augmentation, split, batch_size, num_workers,
                       perturbed=True, include_loc=False, loc_only=False):
    sphere_grid_path = '../../spherical/data_generation/ours_sphere_grid.mat'
    sphere_img_dir = os.path.join(DATA_DIR, 'test', 'sphere_imgs')
    os.makedirs(sphere_img_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    random_rotations = True if perturbed else False
    random_translation = 0.1 if perturbed else 0

    # Increasing `repeat` will generate more cached files
    transform = torchvision.transforms.Compose([
        CacheNPY(prefix="b64_", repeat=augmentation, pick_randomly=False, transform=torchvision.transforms.Compose(
            [
                ToMesh(random_rotations=random_rotations, random_translation=random_translation),
                ProjectOnSphere(bandwidth=64, include_loc=include_loc, loc_only=loc_only, sphere_grid_path=sphere_grid_path)
            ]
        )),
        lambda xs: torch.stack([torch.FloatTensor(x) for x in xs])
    ])
    transform = KeepName(transform)

    test_set = Shrec17(meshes_dir, split, perturbed=perturbed, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    for batch_idx, data in enumerate(loader):
        file_names, data = data
        data = data.view(-1, *data.size()[2:])

        # save to image
        fpath = os.path.join(sphere_img_dir, '%s_%s' % (str(epoch), str(batch_idx)))
        savemat(fpath, mdict={'data': data.numpy(), 'file_names': file_names})


def predict_net(test_images_path, results_dir, batch_size, n_features=6, num_workers=2):
    os.makedirs(results_dir, exist_ok=True)

    # load model
    model = Inception3(in_channels=n_features, num_classes=55, aux_logits=True).cuda()
    weights_path = 'model_weights.pth'
    print('loading weights:', weights_path)
    pretrained_dict = torch.load(weights_path)['state_dict']
    model.load_state_dict(pretrained_dict)

    predictions = []
    ids = []
    test_set = SphereSet(test_images_path, 'test')
    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    for batch_idx, data in enumerate(loader):
        model.eval()

        data, file_names = data
        bs, augs, f, w, h = data.shape
        n_classes = 55

        data = data.cuda()
        with torch.no_grad():
            preds = torch.zeros(bs, n_classes).cuda()
            for i in range(augs):
                batch = data[:, i, ...]
                pred = model(batch).data
                preds.add_(pred)

        predictions.append(preds.cpu().numpy())
        ids.extend([x.split("/")[-1].split(".")[0] for x in file_names])

        print("[{}/{}]      ".format(batch_idx, len(loader)))

    predictions = np.concatenate(predictions)
    predictions_class = np.argmax(predictions, axis=1)
    for i in range(len(ids)):
        if i % 100 == 0:
            print("{}/{}    ".format(i, len(ids)), end="\r")
        idfile = os.path.join(results_dir, ids[i])

        retrieved = [(predictions[j, predictions_class[j]], ids[j]) for j in range(len(ids)) if predictions_class[j] == predictions_class[i]]
        retrieved = sorted(retrieved, reverse=True)
        retrieved = [i for _, i in retrieved]

        with open(idfile, "w") as f:
            f.write("\n".join(retrieved))

    print('saved results in:', results_dir)

    ###########################################################################
    # evaluation using original shrec17 script                                #
    ###########################################################################
    url = "https://shapenet.cs.stanford.edu/shrec17/code/evaluator.zip"
    file_path = "evaluator.zip"

    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    zip_ref = zipfile.ZipFile(file_path, 'r')
    zip_ref.extractall(".")
    zip_ref.close()

    log_dir = os.path.join(results_dir, '..', '')
    print(subprocess.check_output(['node', 'evaluate.js', log_dir], cwd='evaluator').decode('utf-8'))
    dst_path = os.path.join(log_dir, 'summary.csv')
    prefix = os.path.split(os.path.split(log_dir)[0])[1]
    shutil.copy2(os.path.join('evaluator', prefix + '.summary.csv'), dst_path)
    print('copied summary results to:', dst_path)


###############################################################################
# stacked                                                                     #
###############################################################################

def generate_data(data_dir, meshes_dir):
    generate_train_data(meshes_dir, dataset='train', epoch=1, batch_size=2, num_workers=2, augmentation=1, perturbed=True)
    flatten_images(data_dir, 'train')


def test_and_evaluate(data_dir, meshes_dir):
    generate_test_data(meshes_dir, epoch=1, augmentation=1, split='test', batch_size=2, num_workers=2)
    flatten_images(data_dir, 'test', n_augment=1, n_features=6)

    test_images_path = os.path.join(data_dir, 'test', 'images')
    results_dir = os.path.join(data_dir, 'res', 'test_perturbed')
    predict_net(test_images_path, results_dir, batch_size=32, n_features=6, num_workers=2)


if __name__ == "__main__":

    generate_data(DATA_DIR, meshes_dir=MESH_DIR)

    train(DATA_DIR)

    test_and_evaluate(DATA_DIR, meshes_dir=MESH_DIR)
