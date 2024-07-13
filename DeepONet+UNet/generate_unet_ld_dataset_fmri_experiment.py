import numpy as np

from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# from ae import ae

import sys, os, argparse, pickle
sys.path.append('./NPJCOMPUMATS-03747/code')

from unet_fmri import UNet

# Set the XLA_FLAGS environment variable
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/vast/palmer/apps/avx.grace/software/CUDA/11.3.1'

class Train_val_split3:
    '''
    In this class, each frame is a new curve
    '''
    def __init__(self, IDs,val_size_fraction, segment_len,segment_window_factor):
        verbose=True
        bins = np.arange(len(IDs)-segment_len,dtype=np.dtype(np.int16))[:-1]
        if verbose: print("bins: ",bins)

        IDs = np.random.permutation(bins) 
        if verbose: print('IDs:', IDs)

        val_size = int(val_size_fraction*len(IDs))
        
        self.IDs = IDs
        self.val_size = int(np.ceil(val_size_fraction*len(IDs)))
    
    def train_IDs(self):
        train = sorted(self.IDs[:len(self.IDs)-self.val_size])
        return train
    
    def val_IDs(self):
        val = sorted(self.IDs[len(self.IDs)-self.val_size:])
        return val
        
parser = argparse.ArgumentParser(description='DeepONet')
args = parser.parse_args("")

def encode_decode(model, x, fname, make_fig=False):
    e1, ld = model.encode(x)
    # print('low dimensional data: ', ld.shape)
    if make_fig:
        x_hat = model.decode(e1,ld)
        print('reconstructed data: ', x_hat.shape)
        x_hat = x_hat[:100]
        x = x[:100]

        x_hat = np.reshape(x_hat, (x_hat.shape[0], x_hat.shape[1], x_hat.shape[2]))
        x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))

        res=128
        X = np.linspace(0,1,res)
        Y = np.linspace(0,1,res)

        for t in range(100):
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))

            ax1.set_aspect('equal')
            ax1.set_title("True Phase Field (t=%i)"%t, fontsize = 22)
            cont1 = ax1.contourf(X,Y,x[t], 100, cmap='viridis', vmin=0, vmax=1)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(cont1, ax=ax1, cax=cax)
            cbar.ax.tick_params(labelsize=15)

            ax2.set_aspect('equal')
            ax2.set_title("Reconstructed Phase Field (t=%i)"%t, fontsize = 22)
            cont2 = ax2.contourf(X,Y,x_hat[t], 100, cmap='viridis', vmin=0, vmax=1)
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(cont1, ax=ax2, cax=cax)
            cbar.ax.tick_params(labelsize=15)

            plt.savefig(fname + "/t_" + str(t) + ".png")

    return e1, ld

def main():
    args.experiment_name = 'GeneratedFMRI' #'Data_RandProj_20pcs_150frames', Data_20pcs_150frames
    args.data_dim = 'orig' #'Data_2D', 'Data_10D', 'Data_50D', 'Data_orig'
    args.downsample_orig_data=10 # Factor by which we will downsampled the original data 
    args.use_first_n_frames = 5000
    args.perturbation_to_obs = False
    args.segment_len=20
    args.validation_split=0.5
    args.segment_window_factor = 0
    args.randomly_drop_n_last_frames = None
    args.drop_n_last_frames=None
    args.perturbation_to_obs = False
    args.perturbation_to_t = False
    args.random_sample_n_points=None
    args.one_curve_per_frame=True
    args.integral_c='cte_2nd_half' #to pass c as a function fitted on few real points defined by 'num_points_for_c' or None 
    args.num_points_for_c=1
    args.c_scaling_factor=1
    np.random.seed(1)

    address='saved_models/unet_models'

    print('Loading ',os.path.join("./datasets",args.experiment_name + ".p"))
    Data_dict = pickle.load(open(os.path.join("./datasets",args.experiment_name + ".p"), "rb" )) #This data was saved in GPU. So transform it to CPU first
    print(Data_dict.keys())
    Data = Data_dict['Data_'+args.data_dim]
    print('[imported] Data.shape: ',Data.shape)
    
    if args.data_dim=='orig':
        import matplotlib.pyplot as plt
        Data = np.log(Data.values)
        args.scaling_factor = np.quantile(np.abs(Data),0.90)
        Data = (Data-np.mean(Data))/args.scaling_factor
        Data = Data[::args.downsample_orig_data,:]
    else: 
        args.scaling_factor = np.quantile(np.abs(Data),0.90)
    
    print('Data.shape: ',Data.shape)
    args.std_noise = np.mean(np.std(Data,axis=1))/args.perturbation_to_obs_factor if args.perturbation_to_obs else 0
    print('scaling_factor: ',args.scaling_factor)
    
    train_val = 20000 # Number of frames for train and validation. The remaining will be for test
    n_steps = 3000 # number of iterations for training. default=3k epochs
    
    Data = Data[:args.use_first_n_frames,:] 
    
    n_points = Data.shape[0]
    extrapolation_points = Data.shape[0]
    
    t_max = 1 
    t_min = 0
    
    index_np = np.arange(0, n_points, 1, dtype=int)
    index_np = np.hstack(index_np[:, None])
    times_np = np.linspace(t_min, t_max, num=args.segment_len)
    times_np = np.hstack([times_np[:, None]])
    
    ###########################################################
    times = times_np[:, :, None]
    times = times.flatten()
    time_seq = times/t_max
    
    print('Data.shape: ',Data.shape)
    print('times.shape: ',times.shape)
    
    # Data = torch.Tensor(Data).double()
    
    # Original Dataset setup 
    if args.one_curve_per_frame: 
        Data_splitting_indices = Train_val_split3(np.copy(index_np),args.validation_split, args.segment_len,args.segment_window_factor) #Just the first 100 are used for training and validation
    else:
        Data_splitting_indices = Train_val_split2(np.copy(index_np),args.validation_split, args.segment_len,args.segment_window_factor) #Just the first 100 are used for training and validation
    
    Train_Data_indices = np.arange(len(Data_splitting_indices.train_IDs()))
    Val_Data_indices = np.arange(len(Data_splitting_indices.val_IDs()))+len(Data_splitting_indices.train_IDs())
    
    if args.randomly_drop_n_last_frames is not None:
        frames_to_drop = np.random.randint(args.randomly_drop_n_last_frames, size=len(Val_Data_indices)+len(Train_Data_indices))
    elif args.drop_n_last_frames is not None:
        frames_to_drop = np.ones(len(Val_Data_indices)+len(Train_Data_indices),dtype=np.int8) * args.drop_n_last_frames
    elif args.num_points_for_c is not None:
        args.drop_n_last_frames = args.segment_len-args.num_points_for_c
        frames_to_drop = np.ones(len(Val_Data_indices)+len(Train_Data_indices),dtype=np.int8) * args.drop_n_last_frames
        
    print('\nlen(Train_Data_indices): ',len(Train_Data_indices))
    print('Train_Data_indices: ',Train_Data_indices)
    print('\nlen(Val_Data_indices): ',len(Val_Data_indices))
    print('Val_Data_indices: ',Val_Data_indices)
    print('frames_to_drop [for train]: ',frames_to_drop[Train_Data_indices])
    print('frames_to_drop [for val]: ',frames_to_drop[Val_Data_indices])

    x_train = Data[Train_Data_indices,:]
    x_train = tf.cast(x_train, dtype=tf.float32)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))
    x_test = Data[Val_Data_indices,:]
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))

    #Loading model
    model = UNet()
    model_number = np.load(address+'/best_unet_fmri_model_number.npy')
    model_address = address + "/model_"+str(model_number)
    model.load_weights(model_address)

    batch_size=1
    train_ld_ls=[]
    test_ld_ls=[]
    test_e1_ls=[]
    train_e1_ls=[]

    for end in np.arange(batch_size, x_train.shape[0]+1, batch_size):
        start=end-batch_size
        e1, ld = encode_decode(model, x_train[start:end], 'train')
        train_ld_ls.append(ld)
        train_e1_ls.append(e1)

    for end in np.arange(batch_size, x_test.shape[0]+1, batch_size):
        start=end-batch_size
        e1, ld = encode_decode(model, x_test[start:end], 'test', make_fig=True)
        test_e1_ls.append(e1) 
        test_ld_ls.append(ld)

    train_ld = np.concatenate(train_ld_ls,axis=0)
    print('train_ld: ', train_ld.shape)
    np.savez('data/train_ld_unet', X_func=train_ld)

    test_ld = np.concatenate(test_ld_ls, axis=0)
    print('test_ld: ', test_ld.shape)
    np.savez('data/test_ld_unet', X_func=test_ld)
    
    test_e1 = np.concatenate(test_e1_ls, axis=0)
    print('test_e1: ', test_e1.shape)
    np.savez('data/test_e1_unet', X_func=test_e1)
    
    train_e1 = np.concatenate(train_e1_ls,axis=0)
    print('train_e1: ', train_e1.shape)
    np.savez('data/train_e1_unet', X_func=train_e1)

    print('Complete')

if __name__ == '__main__':
    main()
