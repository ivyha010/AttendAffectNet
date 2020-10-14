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
from model_Feature_AAN_only_audio_Pooling import *

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
def myDataloader(audioFeatures_1, audioFeatures_2, args, shuffleBool=False):
    class my_dataset(Dataset):
        def __init__(self, audioData_1, audioData_2):
            self.audioData_1 = audioData_1
            self.audioData_2 = audioData_2

        def __getitem__(self, index):
            return self.audioData_1[index], self.audioData_2[index]

        def __len__(self):
            return len(self.audioData_1)

    # Build dataloaders
    my_dataloader = DataLoader(dataset=my_dataset(audioFeatures_1, audioFeatures_2), batch_size=args.batch_size, shuffle=shuffleBool)
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
    for (v_audio_feature_1, v_audio_feature_2) in validate_loader:
        v_audio_feature_1, v_audio_feature_2 = v_audio_feature_1.to(device), v_audio_feature_2.to(device)
        v_audio_feature_1, v_audio_feature_2 = v_audio_feature_1.unsqueeze(1), v_audio_feature_2.unsqueeze(1)

        vout = the_model(v_audio_feature_1, v_audio_feature_2)

        vout = vout.squeeze(1).cpu().detach().numpy()
        all_cont_output.append(vout)

        del v_audio_feature_1, v_audio_feature_2
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

    # vggish features
    h5file = h5py.File(vggish_feature_file, mode='r')
    get_vggish = h5file.get('default')
    vggish_features = get_vggish.value
    h5file.close()

    # emobase2010 features
    h5file = h5py.File(emobase_feature_file, mode='r')
    get_emobase = h5file.get('default')
    emobase_features = get_emobase.value
    h5file.close()

    return vggish_features, emobase_features

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
    vggish, emobase = loadingfiles()

    # Data loader
    all_dataset = myDataloader(vggish, emobase, args, False)

    model = torch.load(os.path.join(model_path, model_name)) # load the entire model
    model.to(device)

    output_cont = validate_func(all_dataset, model, device)
    print(" ")

if __name__ == "__main__":

    emo_dim = 'arousal' # 'valence' #
    model_name = 'MDB_'+ emo_dim +'_COGNIMUSE_attention_model_Feature_AAN_only_audio.pth'
    sample_path = 'path_to_sample_inputs'  # path to inputs
    model_path = 'path_to_trained_models'  # path to trained models
    pred_path = 'path_to_save_predicted_arousal_and_valence_values'  # path to save predicted arousal/valence values
    # ------------------------------------------------------------------------------------------------------------------
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
    vggish_feature_file = os.path.join(sample_path, 'vggish_sample.h5')
    emobase_feature_file = os.path.join(sample_path, 'emobase_sample.h5')

    main_start_time = time.time()
    main(args)
    print('Total running time: {:.5f} seconds'.format(time.time() - main_start_time))

