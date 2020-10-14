import subprocess
import argparse
import torch
from torch import optim, nn
import os
import time
import gc
from collections import Mapping, Container
from sys import getsizeof
import h5py
from torch.utils.data import DataLoader, Dataset
from model_Feature_AAN_only_video_Pooling import *

def deep_getsizeof(o, ids):
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, np.unicode):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r

# Memory check
def memoryCheck():
    ps = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv'],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    print(ps.communicate(), '\n')
    os.system("free -m")

# Free memory
def freeCacheMemory():
    torch.cuda.empty_cache()
    gc.collect()

# Build dataloaders
def myDataloader(resnet_videoFeatures, flownet_videoFeatures, i3d_videoFeatures, args, shuffleBool=False):
    class my_dataset(Dataset):
        def __init__(self, resnet50_videoData, flownet_videoData, i3d_videoData):
            self.resnet50_videoData = resnet50_videoData
            self.flownet_videoData = flownet_videoData
            self.i3d_videoData = i3d_videoData

        def __getitem__(self, index):
            return self.resnet50_videoData[index], self.flownet_videoData[index], self.i3d_videoData[index]

        def __len__(self):
            return len(self.i3d_videoData)

    # Build dataloaders
    my_dataloader = DataLoader(dataset=my_dataset(resnet_videoFeatures, flownet_videoFeatures, i3d_videoFeatures), batch_size=args.batch_size, shuffle=shuffleBool)
    return my_dataloader

#-----------------------------------------------------------------------------------------------------------------------
def get_position(ntimes):
    # get_pos = []
    row = np.arange(args.seq_length)
    positions = np.repeat(row[:, np.newaxis], ntimes, axis=1)
    return positions

# VALIDATE
# LOAD TEST DATA TO GPU IN BATCHES
def validate_func(validate_loader, the_model, device):
    the_model.eval()
    all_cont_output = []
    for (v_resnet50_feature, v_flownet_feature, v_i3d_feature) in validate_loader:
        v_resnet50_feature, v_flownet_feature, v_i3d_feature = \
            v_resnet50_feature.to(device), v_flownet_feature.to(device), v_i3d_feature.to(device)
        v_resnet50_feature, v_flownet_feature, v_i3d_feature = \
            v_resnet50_feature.unsqueeze(1), v_flownet_feature.unsqueeze(1), v_i3d_feature.unsqueeze(1)

        vout = the_model(v_resnet50_feature, v_flownet_feature, v_i3d_feature)

        vout = vout.squeeze(1).cpu().detach().numpy()
        all_cont_output.append(vout)

        del v_resnet50_feature, v_flownet_feature, v_i3d_feature
        freeCacheMemory()

    all_cont_output = np.concatenate(all_cont_output, axis=0)

    return all_cont_output

# Decay the learning rate
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    newlr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = newlr

# Load extracted features and arousal files
def loadingfiles():
    # Load extracted features and arousal .h5 files
    print('\n')
    print('Loading h5 files containing extracted features')
    loading_time = time.time()

    # resnet50 features
    h5file = h5py.File(resnet50_feature_file, mode='r')
    get_resnet50 = h5file.get('default')
    resnet50_features = get_resnet50.value
    h5file.close()

    # flownet features
    h5file = h5py.File(flownet_feature_file, mode='r')
    get_flownet = h5file.get('default')
    flownet_features = get_flownet.value
    h5file.close()

    # i3d features
    h5file = h5py.File(i3d_feature_file, mode='r')
    get_i3d = h5file.get('default')
    i3d_features = get_i3d.value
    h5file.close()

    return resnet50_features, flownet_features, i3d_features

# Main
def main(args):
    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Manual seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device: ', device)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Get the input dim
    resnet50, flownet, i3d = loadingfiles()

    # Data loader
    all_dataset = myDataloader(resnet50, flownet, i3d, args, False)

    model = torch.load(os.path.join(model_path, model_name))  # load the entire model
    model.to(device)

    output_cont = validate_func(all_dataset, model, device)
    print(" ")

if __name__ == "__main__":

    emo_dim = 'valence' # 'arousal' #
    model_name = 'MDB_'+ emo_dim +'_COGNIMUSE_attention_model_Feature_AAN_only_video.pth'
    sample_path = 'path_to_sample_inputs'  # path to inputs
    model_path = 'path_to_trained_models'  # path to trained models
    pred_path = 'path_to_save_predicted_arousal_and_valence_values'  # path to save predicted arousal/valence values
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=model_path, help='path for saving trained models')
    # -------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--batch_size', type=int, default=30, help='number of feature vectors loaded per batch')

    #parser.add_argument('-d_model', type=int, default=8)
    #parser.add_argument('-n_head', type=int, default=2)
    #parser.add_argument('-n_layers', type=int, default=6)
    #parser.add_argument('-dropout', type=float, default=0.1)
    #parser.add_argument('-embs_share_weight', action='store_true')
    #parser.add_argument('-proj_share_weight', action='store_true')
    #parser.add_argument('-label_smoothing', action='store_true')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 123)')

    args = parser.parse_args()
    print(args)

    # ------------------------------------------------------------------------------------------------------------------
    resnet50_feature_file = os.path.join(sample_path, 'resnet50_sample.h5')
    i3d_feature_file = os.path.join(sample_path, 'i3d_sample.h5')
    flownet_feature_file = os.path.join(sample_path, 'flownet_sample.h5')

    main_start_time = time.time()
    main(args)
    print('Total running time: {:.5f} seconds'.format(time.time() - main_start_time))

