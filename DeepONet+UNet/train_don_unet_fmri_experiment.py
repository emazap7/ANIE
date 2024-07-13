import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
# from ae import ae

import sys, os, argparse, pickle
sys.path.append('./NPJCOMPUMATS-03747/code')

# from ae_fmri import ae
from unet_fmri import UNet


import matplotlib
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)

from don_fmri import DeepONet_Model ###############################

# Set the XLA_FLAGS environment variable
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/vast/palmer/apps/avx.grace/software/CUDA/11.3.1'


def count_parameters(model: tf.keras.Model) -> int:
    """
    Count the total number of parameters in a TensorFlow model.

    Args:
        model (tf.keras.Model): The TensorFlow model.

    Returns:
        int: The total number of parameters.
    """
    return model.count_params()


def preprocess2(x):
    X_fun = np.transpose(x.squeeze(), axes = [1, 0]) 

    # Parameters
    num_time_points = 20  # Length of each sequence
    num_sequences = X_fun.shape[1] - num_time_points + 1
    
    # Pre-allocate the new 3D tensor
    X_3D = np.zeros((num_sequences, X_fun.shape[0], num_time_points)) # [batch, space, time]
    
    # Fill the 3D tensor with sequences
    for i in range(num_sequences):
        X_3D[i] = X_fun.squeeze()[:, i:i+num_time_points]

    X_func = np.reshape(X_3D, (-1,X_3D.shape[2], X_3D.shape[1])) # [batch, time, space]
    return X_func
    
def preprocess(x, e1_vec):

    # Creating a dataset from the tensor
    X_fun = np.transpose(x, axes = [1, 0]) #axes=[0, 2, 3, 1]) np.random.rand(80, 2459)  # Example data    
    e1_vec = np.transpose(e1_vec, axes = [1, 2, 0])

    num_time_points = 20  # Length of each sequence
    num_sequences = X_fun.shape[1] - num_time_points + 1
    
    # Pre-allocate the new 3D tensor
    X_3D = np.zeros((num_sequences, X_fun.shape[0], num_time_points)) # [batch, space, time]
    E1_3D_vec = np.zeros((num_sequences, e1_vec.shape[0],  e1_vec.shape[1], num_time_points)) 
    
    # Fill the 3D tensor with sequences
    for i in range(num_sequences):
        X_3D[i] = X_fun.squeeze()[:, i:i+num_time_points]
        E1_3D_vec[i,:,:,:] = e1_vec.squeeze()[:, :, i:i+num_time_points]

    X_func = np.reshape(X_3D, (-1,X_3D.shape[2], X_3D.shape[1])) # [batch, time, space]
    E1_3D_vec = np.reshape(E1_3D_vec, (X_func.shape[0],E1_3D_vec.shape[3], E1_3D_vec.shape[1], E1_3D_vec.shape[2])) # [batch, time, space]

    index = list(range(0, 10)) # Train on the first 10 time points (total length is 20) 
    X_func = X_func[:, index] # 
    X_func = np.transpose(X_func, axes = [0, 2, 1]) #axes=[0, 2, 3, 1])

    X_loc = np.array(index)/num_time_points #100 (or divide by time length)
    X_loc = X_loc[:,None]

    y = np.transpose(X_func, axes = [0, 2, 1])

    return X_func, X_loc, y, E1_3D_vec

def tensor(x):
    return tf.convert_to_tensor(x, dtype=tf.float32)


@tf.function(jit_compile=True)
def train(don_model, X_func, X_loc, y):
    with tf.GradientTape() as tape:
        y_hat  = don_model(X_func, X_loc)
        loss   = don_model.loss(y_hat, y)[0]

    gradients = tape.gradient(loss, don_model.trainable_variables)
    don_model.optimizer.apply_gradients(zip(gradients, don_model.trainable_variables))
    return(loss)

def error_metric(true, pred):
    true = true[:pred.shape[0],:]
    pred = np.reshape(pred, (-1,20, 80)) # covert back to [samples, time, space]
    num = np.abs(true - pred)**2 #
    num = np.sum(num) #[samples, time steps]
    den = np.abs(true)**2
    den = np.sum(den)

    return num/den

def show_error(don_model, unet_model, X_func, X_loc, E1_3D_vec, pf_true, save_vars=False):
    y_pred = don_model(X_func, X_loc)

    y_pred = np.reshape(y_pred, (-1,unet_model.latent_dim)) # reshape to [samples * time, latent_space]
    E1_3D_vec = np.reshape(E1_3D_vec, (-1,E1_3D_vec.shape[2], E1_3D_vec.shape[3])) # [batch, time, space]
    pf_pred = unet_model.decode(E1_3D_vec,y_pred) # takes  e1 and latent as input
    error = error_metric(pf_true, pf_pred)


    if save_vars: 
        np.save('data/200_curves_DeepONet_UNET_GT.npy', pf_true)
        np.save('data/200_curves_DeepONet_UNET_pred.npy', pf_pred)

def main():
    Par = {}

    # These are the variables are encoded to latent space
    # train_dataset = np.load('data/train_ld.npz')['X_func']
    # test_dataset = np.load('data/test_ld.npz')['X_func']
    train_dataset = np.load('data/train_ld_unet.npz')['X_func']
    test_dataset = np.load('data/test_ld_unet.npz')['X_func']
    
    train_dataset_e1 = np.load('data/train_e1_unet.npz')['X_func']
    test_dataset_e1 = np.load('data/test_e1_unet.npz')['X_func']


    Par['address'] = 'saved_models/don_models_unet' #####################################################

    print(Par['address'])
    print('------\n')

    X_func_train, X_loc_train, y_train, E1_3D_vec_train = preprocess(train_dataset, train_dataset_e1)
    X_func_test, X_loc_test, y_test, E1_3D_vec_test = preprocess(test_dataset, test_dataset_e1)
    Par['n_channels'] = X_func_train.shape[-1]

    Par['mean'] = np.mean(X_func_train)
    Par['std'] =  np.std(X_func_train)

    
    don_model = DeepONet_Model(Par)
    
    n_epochs = 100
    batch_size = 1

    print("\n\nDeepONet Training Begins")
    begin_time = time.time()


    for i in range(n_epochs+1):
        for end in np.arange(batch_size, X_func_train.shape[0]+1, batch_size):
            start = end - batch_size
            loss = train(don_model, tensor(X_func_train[start:end]), tensor(X_loc_train), tensor(y_train[start:end]))

        if i%1 == 0:

            don_model.save_weights(Par['address'] + "/model_"+str(i))

            train_loss = loss.numpy()

            y_hat = don_model(X_func_test, X_loc_test)

            val_loss = np.mean( (y_hat - y_test)**2 )

            print("epoch:" + str(i) + ", Train Loss:" + "{:.3e}".format(train_loss) + ", Val Loss:" + "{:.3e}".format(val_loss) +  ", elapsed time: " +  str(int(time.time()-begin_time)) + "s"  )

            don_model.index_list.append(i)
            don_model.train_loss_list.append(train_loss)
            don_model.val_loss_list.append(val_loss)



    #Convergence plot
    index_list = don_model.index_list
    train_loss_list = don_model.train_loss_list
    val_loss_list = don_model.val_loss_list
    np.savez(Par['address']+'/convergence_data', index_list=index_list, train_loss_list=train_loss_list, val_loss_list=val_loss_list)


    plt.close()
    fig = plt.figure(figsize=(10,7))
    plt.plot(index_list, train_loss_list, label="train", linewidth=2)
    plt.plot(index_list, val_loss_list, label="val", linewidth=2)
    plt.legend(fontsize=16)
    plt.yscale('log')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.savefig(Par["address"] + "/convergence.png", dpi=800)
    plt.close()

    if True:
        unet_model = UNet()
        print('Number of parameters for UNET: ',count_parameters(unet_model))
        unet_model_number = np.load('saved_models/unet_models/best_unet_fmri_model_number.npy')
        unet_model_address = "saved_models/unet_models/model_"+str(unet_model_number)
        unet_model.load_weights(unet_model_address)


        don_model = DeepONet_Model(Par)
        print('Number of parameters for DON: ',count_parameters(don_model))
        don_model_number = index_list[np.argmin(val_loss_list)]
        np.save('data/best_don_unet_model_number', don_model_number)
        don_model_address = Par['address'] + "/model_"+str(don_model_number)
        don_model.load_weights(don_model_address)

        print('best DeepONet model: ', don_model_number)

        n_samples = 200
        pf_true = np.load('data/train_data.npy').astype(np.float32)
        pf_true = np.reshape(pf_true, (-1,2489,80))
        pf_true = preprocess2(pf_true)
        pf_true_train = pf_true[:n_samples, :]

        # reshape
        print('pf_true_train.shape: ',pf_true_train.shape)
        
        pf_true = np.load('data/test_data.npy').astype(np.float32)
        pf_true = np.reshape(pf_true, (-1,2490,80))
        pf_true = preprocess2(pf_true)
        pf_true_test = pf_true[:n_samples, :]

        X_loc = np.linspace(0,1,20)[:][:,None]

        print('Train Dataset')
        show_error(don_model, unet_model, X_func_train[:n_samples],X_loc, E1_3D_vec_train[:n_samples], pf_true_train)

        print('Test Dataset')
        show_error(don_model, unet_model, X_func_test[:n_samples], X_loc, E1_3D_vec_test[:n_samples], pf_true_test, save_vars=True)

        print('--------Complete--------')


main()
