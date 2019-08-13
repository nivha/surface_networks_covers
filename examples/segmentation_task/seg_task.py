"""
This example file shows the process used to generate flattened images from given models (meshes),
train a model (neural network) on the flattened images, and make predictions on such flattened images,
then use these predictions to make predictions on the original test models (meshes).

The example here uses only one mesh (SHREC07 2nd mesh). The predictions are made using the weights
of the trained model used in our paper. 

In order to make this pipeline run on real data, one has to download the whole dataset,
then change the paths and parameters accordingly
"""

import os
import sys
import subprocess
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

import segmentation.unet_model as unet
import segmentation.training as train_utils
from segmentation.meshseg_dataset import MeshSeg, img_loader

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
# General                                                                     #
###############################################################################
def get_meshes_paths(meshes_dir):
    meshes_paths = []
    for root, _, fnames in sorted(os.walk(meshes_dir)):
        for fname in fnames:
            if fname.startswith('.'): continue
            if fname.endswith('.off'):
                path = os.path.join(root, fname)
                meshes_paths.append(path)
    return meshes_paths


###############################################################################
# Generate Data                                                               #
###############################################################################
def generate_data_mesh(data_dir, mesh_path, segmentation_path, n_augment=1):
    # number of augmentations per mesh (augmentation = rotation, scale and cones-permutation)

    # create directories
    imgs_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    flat_dir = os.path.join(data_dir, 'flat_info')
    for dirpath in [data_dir, imgs_dir, labels_dir, flat_dir]:
        os.makedirs(dirpath, exist_ok=True)

    # prepare matlab flatten_mesh script's parameters
    d = {
        'matlab_path': matlab_path,
        'script_path': '../../matlab/api',
        'gptoolbox_path': gptoolbox_path,
        'mesh_path': mesh_path,
        'segmentation_path': segmentation_path,
        'data_dir': data_dir,
        'split': 'test',
        'n_augment': n_augment,
    }
    matlab_cmd = "cd(\'{script_path}\');" \
                 "flatten_mesh(\'{gptoolbox_path}\', \'{mesh_path}\', \'{segmentation_path}\'," \
                 "\'{data_dir}\', \'{split}\', {n_augment});" \
                 "exit;".format(**d)
    cmd = [matlab_path, '-nodisplay', '-nosplash', '-nodesktop', '-r', matlab_cmd]

    # call matlab flatten_mesh script
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in iter(process.stdout.readline, b''):
        sys.stdout.write(line.decode(sys.stdout.encoding))


def generate_data(data_dir, meshes_dir, n_augment=1):
    # number of augmentations per mesh (augmentation = rotation, scale and cones-permutation)
    meshes_paths = get_meshes_paths(meshes_dir)
    for mesh_path in meshes_paths:
        print('processing:', mesh_path)
        name = os.path.split(mesh_path)[-1]
        segm_path = os.path.join(meshes_dir, name.replace('.off', '.txt'))
        generate_data_mesh(data_dir, mesh_path, segm_path, n_augment)


###############################################################################
# Train                                                                       #
###############################################################################
def train_net(imgs_dir):
    # learning parameters
    batch_size = 2
    n_epochs = 4
    lr = 0.2
    model = unet.UNetDeep(n_channels=3, n_classes=8).cuda()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set train and validation set
    train_dset = MeshSeg(imgs_dir, 'train')
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dset = MeshSeg(imgs_dir, 'val')
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=4)

    # set learning parameters
    model.apply(train_utils.weights_init)
    class_counts = np.array([28207994, 1142531, 12597950, 52804502, 219003769, 58886386, 13121845, 1421711])
    dist = class_counts / class_counts.sum()
    dist_fixed = 1 / dist / dist.shape[0]
    weights = torch.from_numpy(dist_fixed).type(torch.get_default_dtype()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # actual training
    for epoch in range(1, n_epochs + 1):
        trn_loss, trn_err = train_utils.train(model, train_loader, optimizer, criterion, epoch)
        print('Epoch {:d}: Train - Loss: {:.7f}, Acc: {:.7f}'.format(epoch, trn_loss, 1 - trn_err))
        val_loss, val_err = train_utils.test(model, val_loader, criterion, epoch)
        print('Epoch {:d}: Val - Loss: {:.7f} | Acc: {:.7f}'.format(epoch, val_loss, 1 - val_err))
        # adjust lr
        train_utils.adjust_learning_rate(lr=lr, decay=0.995, optimizer=optimizer, cur_epoch=epoch, every_n_epochs=1)


###############################################################################
# Predict                                                                     #
###############################################################################
def make_predictions(img_path, res_dir, model):
    # imgpaths = sorted(os.listdir(imgs_dir))
    # for img_name in imgpaths:

    # read test image
    # img_path = os.path.join(imgs_dir, img_name)
    img = img_loader(img_path).cuda().unsqueeze_(0)
    target = img_loader(img_path.replace('images', 'labels')).cuda().unsqueeze_(0)

    # predict with model
    output = model(img)
    pred = train_utils.get_predictions(output, max_index=False)
    # transform python 0-base to matlab 1-base
    pimg = pred[0].cpu().numpy() + 1
    targ = target[0].cpu().numpy() + 1

    # save some visualizations
    fname = os.path.split(img_path)[1][:-4]
    cmap = plt.cm.jet
    plt.imsave(os.path.join(res_dir, '%s_targ.png' % fname), targ, cmap=cmap)
    bs, c, h, w = pred.size()
    values, indices = pred.max(1)
    indices = indices.view(bs, h, w)
    pimg_indexes = indices[0]
    plt.imsave(os.path.join(res_dir, '%s_pred.png' % fname), pimg_indexes.cpu(), cmap=cmap)

    # save mat file prediction
    mat_path = os.path.join(res_dir, '%s_pred.mat' % fname)
    savemat(mat_path, {'data': pimg})
    print('saved:', mat_path)


def aggregate_predictions(data_dir, segmentation_path, preds_dir):
    # prepare matlab aggregate_predictions script's parameters
    d = {
        'matlab_path': matlab_path,
        'script_path': '../../matlab/api',
        'gptoolbox_path': gptoolbox_path,
        'segmentation_path': segmentation_path,
        'preds_dir': preds_dir,
        'flat_dir': os.path.join(data_dir, 'flat_info'),
        'ploteach': 'true',
    }
    matlab_cmd = "cd(\'{script_path}\');" \
                 "aggregate_predictions(\'{gptoolbox_path}\', \'{preds_dir}\', \'{segmentation_path}\', " \
                 "\'{flat_dir}\', {ploteach});" \
                 "exit;".format(**d)
    cmd = [matlab_path, '-nodisplay', '-nosplash', '-nodesktop', '-r', matlab_cmd]

    # call matlab aggregate_predictions script
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in iter(process.stdout.readline, b''):
        sys.stdout.write(line.decode(sys.stdout.encoding))


def predict(data_dir, imgs_dir, meshes_dir, model):
    results_dir = os.path.join(data_dir, 'results')
    # predict all images
    imgpaths = sorted(os.listdir(imgs_dir))
    for img_name in imgpaths:
        print('processing:', img_name)
        shrec_num = img_name[:img_name.find('_')]
        # create results dir
        mesh_results_dir = os.path.join(results_dir, shrec_num)
        os.makedirs(mesh_results_dir, exist_ok=True)
        # call trained model on test set to create predictions on the flattened model
        img_path = os.path.join(imgs_dir, img_name)
        make_predictions(img_path, mesh_results_dir, model)

    # aggregate predictions if needed, and make predictions on the original model
    for shrec_num in os.listdir(results_dir):
        segm_path = os.path.join(meshes_dir, '{}.txt'.format(shrec_num))
        mesh_results_dir = os.path.join(results_dir, shrec_num)
        aggregate_predictions(data_dir=data_dir, segmentation_path=segm_path, preds_dir=mesh_results_dir)


if __name__ == "__main__":
    # folder where to save the generated flattened images of the train models (meshes)
    IMGS_DIR = os.path.join(DATA_DIR, 'images')

    # generate some train flattened-images from the train set of meshes
    generate_data(DATA_DIR, MESH_DIR, n_augment=2)

    # train a model using the flattened images
    train_net(imgs_dir=IMGS_DIR)

    # predict results using the trained model
    # let's load our trained model
    model = unet.UNetDeep(n_channels=3, n_classes=8).cuda()
    weights_path = 'model_weights.pth'
    pretrained_dict = torch.load(weights_path)['state_dict']
    model.load_state_dict(pretrained_dict)
    print('loaded weights:', weights_path)

    predict(data_dir=DATA_DIR, imgs_dir=IMGS_DIR, meshes_dir=MESH_DIR, model=model)

