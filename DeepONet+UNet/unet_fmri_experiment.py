import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
import time
import matplotlib.pyplot as plt

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

@tf.function(jit_compile=True)
def train(model, x, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss   = model.loss(tf.reshape(y_pred, [-1]), tf.reshape(x, [-1]))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return(loss)

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

    # Load data 
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
    num_samples = x_train.shape[0]

    np.save('data/train_data.npy', x_train)
    np.save('data/test_data.npy', x_test)

    Par={}

    address = 'saved_models/unet_models'
    Par['address'] = address

    #Create an object
    model = UNet()
    print('Model created')

    n_epochs = 20
    batch_size = 1
    n_batches = int(num_samples/batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 10**-4)

    begin_time = time.time()
    print('Training Begins')
    for i in range(n_epochs+1):
        for j in np.arange(0, num_samples-batch_size, batch_size):
            loss = train(model, x_train[j:(j+batch_size)], optimizer)

        if i%1 == 0:

            model.save_weights(address + "/model_"+str(i))

            train_loss = loss.numpy()

            y_pred = model(x_test)
            val_loss = np.mean( (y_pred - x_test)**2 )

            print("epoch:" + str(i) + ", Train Loss:" + "{:.3e}".format(train_loss) + ", Val Loss:" + "{:.3e}".format(val_loss) +  ", elapsed time: " +  str(int(time.time()-begin_time)) + "s"  )

            model.index_list.append(i)
            model.train_loss_list.append(train_loss)
            model.val_loss_list.append(val_loss)

    print('Training complete')

    #Convergence plot
    index_list = model.index_list
    train_loss_list = model.train_loss_list
    val_loss_list = model.val_loss_list
    np.savez(address+'/convergence_data', index_list=index_list, train_loss_list=train_loss_list, val_loss_list=val_loss_list)

    plt.close()
    fig = plt.figure(figsize=(10,7))
    plt.plot(index_list, train_loss_list, label="train", linewidth=2)
    plt.plot(index_list, val_loss_list, label="val", linewidth=2)
    plt.legend(fontsize=16)
    plt.yscale('log')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.savefig( address + "/convergence.png", dpi=800)
    plt.close()

    best_model_number = index_list[np.argmin(val_loss_list)]
    print('Best autencoder model: ', best_model_number)

    np.save(address+'/best_unet_fmri_model_number', best_model_number)

    print('--------Complete--------')

main()
