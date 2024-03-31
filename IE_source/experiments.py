#General libraries
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
import scipy
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

import torchcubicspline

from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

#Custom libraries
from IE_source.utils import Select_times_function, EarlyStopping, SaveBestModel, to_np, Integral_part, LRScheduler, load_checkpoint, Train_val_split, Dynamics_Dataset, Test_Dynamics_Dataset, fun_interpolation, plot_dim_vs_time
from torch.utils.data import SubsetRandomSampler
from IE_source.solver import IESolver_monoidal
from IE_source.Attentional_IE_solver import Integral_attention_solver, Integral_attention_solver_multbatch, Integral_spatial_attention_solver_multbatch #, Integral_spatial_attention_solver
from IE_source.kernels import RunningAverageMeter, log_normal_pdf, normal_kl
from IE_source.utils import plot_reconstruction

#Torch libraries
import torch
from torch.nn import functional as F

if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"

def Full_IE_experiment(G_NN, kernel, F_func, f_func, Data, dataloaders, time_seq, args, extrapolation_points):
    # scaling_factor=1
        
    # -- metadata for saving checkpoints
    if args.model=='nie': 
        str_model_name = "nie"
    elif args.model=='node': 
        str_model_name = "node"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)

    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else: 
            num_experiments = [int(i[3:]) for i in txt]
            num_experiments = np.array(num_experiments).max()
         # -- logger location
        writer = SummaryWriter(os.path.join(path_to_experiment,'run'+str(num_experiments+1)))
        print('writer.log_dir: ',writer.log_dir)
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)
            
        with open(os.path.join(writer.log_dir,'commandline_args.txt'), 'w') as f:
            for key, value in args.__dict__.items(): 
                f.write('%s:%s\n' % (key, value))



    obs = Data
    times = time_seq
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    loader_test = dataloaders['test']
    # Train Neural IDE
    get_times = Select_times_function(times,extrapolation_points)

    if args.kernel_split is True:
        if args.kernel_type_nn is True and args.free_func_nn is True:
            All_parameters = list(F_func.parameters()) + list(kernel.parameters()) + list(f_func.parameters())
        elif args.kernel_type_nn is True:
            All_parameters = list(F_func.parameters()) + list(kernel.parameters())
    else:
        All_parameters = G_NN.parameters()
    
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)
    
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)
    
    
    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        if args.free_func_nn is True:
            G_NN, optimizer, scheduler, kernel, F_func, f_func = load_checkpoint(path, G_NN, optimizer, scheduler, kernel, F_func, f_func)
        else:
            G_NN, optimizer, scheduler, kernel, F_func, f_ = load_checkpoint(path, G_NN, optimizer, scheduler, kernel, F_func, None)
            f_func = f_func


    
    if args.mode=='train':
        
        early_stopping = EarlyStopping(patience=1000,min_delta=0)

        
        all_train_loss=[]
        all_val_loss=[]

        save_best_model = SaveBestModel()
        start = time.time()
        for i in range(args.epochs):
            if args.kernel_split is True:
                kernel.train()
            else: 
                G_NN.train()
            start_i = time.time()
            print('Epoch:',i)
            
            counter=0
            train_loss = 0.0
            for obs_, ts_, ids_ in tqdm(train_loader): 
                obs_ = obs_.to(args.device)
                ts_ = ts_.to(args.device)
                ids_ = ids_.to(args.device)
                

                ids_, indices = torch.sort(ids_)
                
                obs_ = obs_[indices,:]
                ts_ = ts_[indices]


                z_ = IESolver_monoidal(
                        x = ts_.to(device),
                        dim = args.dim, 
                        c = lambda x: obs_[0].flatten().to(device), 
                        d = lambda x,y: torch.Tensor([1]).to(device), 
                        k = kernel, 
                        f = F_func,
                        lower_bound = lambda x: torch.Tensor([ts_[0]]).to(device),
                        upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                        max_iterations = args.max_iterations,
                        kernel_nn=True,
                        kernel_split = args.kernel_split,
                        G_nn = args.G_NN,
                        integration_dim = 0,
                        mc_samplings=1000,
                        num_internal_points = args.num_internal_points
                        ).solve()


                loss = F.mse_loss(z_[:,:], obs_.detach()[:,:])  



                optimizer.zero_grad()
                loss.backward()#(retain_graph=True)
                optimizer.step()

                counter += 1
                train_loss += loss.item()
                
                del loss
                del obs_, ts_, z_
                if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                    scheduler.step()
                 

            train_loss /= counter
            all_train_loss.append(train_loss)
            
            if args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
            
            del train_loss


            ## Validating
            if args.kernel_split is True:
                kernel.eval()
            else:
                G_NN.eval()
            with torch.no_grad():

                
                val_loss = 0.0
                counter = 0
                if len(val_loader)>0:
                    
                    for obs_val, ts_val, ids_val in tqdm(val_loader):
                        obs_val = obs_val.to(args.device)
                        ts_val = ts_val.to(args.device)
                        ids_val = ids_val.to(args.device)

                        ids_val, indices = torch.sort(ids_val)
                        
                        obs_val = obs_val[indices,:]
                        ts_val = ts_val[indices]

                        
                        obs_val = torch.cat((obs_[0][None,:],obs_val))
                        ts_val = torch.hstack((ts_[0],ts_val))
                        ids_val = torch.hstack((ids_[0],ids_val))

                        
                        z_val = IESolver_monoidal(
                                    x = ts_val.to(device),
                                    y_0 = obs_val[0].flatten().to(device), 
                                    c = lambda x: obs_val[0].flatten().to(device), 
                                    d = lambda x,y: torch.Tensor([1]).to(device), 
                                    k = kernel, 
                                    f = F_func,
                                    lower_bound = lambda x: torch.Tensor([ts_val[0]]).to(device),
                                    upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                                    max_iterations = args.max_iterations,
                                    kernel_nn=True,
                                    kernel_split = args.kernel_split,
                                    G_nn = args.G_NN,
                                    integration_dim = 0,
                                    mc_samplings=1000,
                                    num_internal_points = args.num_internal_points
                                    ).solve()

                        
                        loss_validation = F.mse_loss(z_val[:,:], obs_val.detach()[:,:])
                        
                        
                        del obs_val, ts_val, z_val

                        counter += 1
                        val_loss += loss_validation.item()
                        
                        del loss_validation

                        #LRScheduler(loss_validation)
                        if args.lr_scheduler == 'ReduceLROnPlateau':
                            scheduler.step(val_loss)
                else: counter += 1

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

                writer.add_scalar('train_loss', all_train_loss[-1], global_step=i)
                if len(all_val_loss)>0:
                    writer.add_scalar('val_loss', all_val_loss[-1], global_step=i)
                if args.lr_scheduler == 'ReduceLROnPlateau':
                    writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], global_step=i)
                elif args.lr_scheduler == 'CosineAnnealingLR':
                    writer.add_scalar('Epoch/learning_rate', scheduler.get_last_lr()[0], global_step=i)


                if i % args.plot_freq == 0:
                    obs_test, ts_test, ids_test = next(iter(loader_test))

                    ids_test, indices = torch.sort(ids_test)
                    
                    obs_test = obs_test[indices,:]
                    ts_test = ts_test[indices]
                    


                    obs_test = obs_test.to(args.device)
                    ts_test = ts_test.to(args.device)
                    ids_test = ids_test.to(args.device)
                    
                    z_test = IESolver_monoidal(
                                    x = ts_test.to(device), 
                                    c = lambda x: obs_test[0].flatten().to(device), 
                                    d = lambda x,y: torch.Tensor([1]).to(device), 
                                    k = kernel, 
                                    f = F_func,
                                    lower_bound = lambda x: torch.Tensor([ts_test[0]]).to(device),
                                    upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                                    max_iterations = args.max_iterations,
                                    kernel_nn=True,
                                    kernel_split = args.kernel_split,
                                    G_nn = args.G_NN,
                                    integration_dim = 0,
                                    mc_samplings=1000,
                                    num_internal_points = args.num_internal_points
                                    ).solve()
                    
                    plt.figure(0, facecolor='w')
                    
                    plt.plot(np.log10(all_train_loss))
                    plt.plot(np.log10(all_val_loss))
                    plt.xlabel("Epoch")
                    plt.ylabel("MSE Loss")
                    
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))

                    new_times = to_np(ts_test)

                    plt.figure(figsize=(8,8),facecolor='w')
                    z_p = z_test
                    z_p = to_np(z_p)

                    plt.figure(1, facecolor='w')
                    plt.plot(z_p[:,0],z_p[:,1],c='r', label='model')
                    obs_print = to_np(obs)
                    
                    plt.scatter(obs_print[:,0],obs_print[:,1],label='Data',c='blue', alpha=0.5)
                    plt.xlabel("dim 0")
                    plt.ylabel("dim 1")
                    
                    plt.legend()
                    
                    plt.savefig(os.path.join(path_to_save_plots,'plot_dim0vsdim1_epoch'+str(i)))

                    
                    if obs_print.shape[1]<args.num_dim_plot: 
                        args.num_dim_plot=obs_print.shape[1]
                        n_plots_x = int(np.ceil(np.sqrt(args.num_dim_plot)))
                        n_plots_y = int(np.floor(np.sqrt(args.num_dim_plot)))
                        fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=(10, 10), sharex=True, dpi=100, facecolor='w', edgecolor='k')
                        ax=ax.ravel()
                        for idx in range(args.num_dim_plot):
                            ax[idx].plot(new_times,z_p[:,idx],c='r', label='model')

                            ax[idx].scatter(to_np(times)[:],obs_print[:,idx],label='Data',c='blue', alpha=0.5)
                            ax[idx].set_xlabel("Time")
                            ax[idx].set_ylabel("dim"+str(idx))

                            ax[idx].legend()

                        fig.tight_layout()
                        plt.savefig(os.path.join(path_to_save_plots, 'plot_ndim_epoch'+str(i)))

                    if 'calcium_imaging' in args.experiment_name:
                        
                        data_to_plot = obs_print[:20,:]*args.scaling_factor #Get the first 10 samples for a test 
                        predicted_to_plot = z_p[:20,:]*args.scaling_factor
                        data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                        predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                        predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                        data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                        fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                        c=0
                        for idx_row in range (2): 
                            for idx_col in range(10):
                                ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                ax[2*idx_row,idx_col].axis('off')
                                _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                ax[2*idx_row+1,idx_col].axis('off')
                                c+=1
                        fig.tight_layout()
                        plt.savefig(os.path.join(path_to_save_plots, 'plot_first20frame_rec'+str(i)))


                        # Plot the last 20 frames  
                        data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                        predicted_to_plot = z_p[-20:,:]*args.scaling_factor
                        data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                        predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                        predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                        data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                        fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                        c=0
                        for idx_row in range (2): 
                            for idx_col in range(10):
                                ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                ax[2*idx_row,idx_col].axis('off')
                                _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                ax[2*idx_row+1,idx_col].axis('off')
                                c+=1
                        fig.tight_layout()
                        plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_rec'+str(i)))


                        #Plot the R2 and MSE loss between the original data and the predicted overtime. 
                        data_to_plot = obs_print[:,:]*args.scaling_factor #Get the first 10 samples for a test 
                        predicted_to_plot = z_p[:,:]*args.scaling_factor
                        data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                        predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                        all_r2_scores = []
                        all_mse_scores = []

                        for idx_frames in range(len(data_to_plot)):
                            _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                            all_r2_scores.append(r_value)
                            
                            tmp_mse_loss = mean_squared_error(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                            all_mse_scores.append(tmp_mse_loss)

                        fig,ax = plt.subplots(2,1, figsize=(15,5), sharex=True, facecolor='w')
                        ax[0].plot(np.arange(len(all_r2_scores)),all_r2_scores)
                        ax[1].plot(np.arange(len(all_mse_scores)),all_mse_scores)
                        ax[1].set_xlabel("Frames")
                        ax[0].set_ylabel("R2")
                        ax[1].set_ylabel("MSE")
                        fig.tight_layout()
                        plt.savefig(os.path.join(path_to_save_plots, 'plot_performance_rec'+str(i)))

                        #Plot integral and ode part separated
                        if ode_func is not None and F_func is not None:
                            Trained_Data_ode = odeint(ode_func,torch.Tensor(obs_print[0,:]).flatten().to(args.device),times.to(args.device),rtol=1e-4,atol=1e-4)
                            Trained_Data_ode_print = to_np(Trained_Data_ode)
                            Trained_Data_integral_print  = z_p - Trained_Data_ode_print
                            

                            data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                            predicted_to_plot_ode = Trained_Data_ode_print[-20:,:]*args.scaling_factor
                            predicted_to_plot_ide = Trained_Data_integral_print[-20:,:]*args.scaling_factor
                            data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                            predicted_to_plot_ode = args.fitted_pca.inverse_transform(predicted_to_plot_ode)
                            predicted_to_plot_ide = args.fitted_pca.inverse_transform(predicted_to_plot_ide)

                            predicted_to_plot_ode = predicted_to_plot_ode.reshape(predicted_to_plot_ode.shape[0],184, 208) # Add the original frame dimesion as input
                            predicted_to_plot_ide = predicted_to_plot_ide.reshape(predicted_to_plot_ide.shape[0],184, 208)
                            data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                            fig,ax = plt.subplots(6,10, figsize=(15,8), facecolor='w')
                            c=0
                            step = 0
                            for idx_row in range (2): 
                                for idx_col in range(10):
                                    ax[2*idx_row+step,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row+step,idx_col].axis('off')

                                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ode[c,:].flatten())
                                    ax[2*idx_row+1+step,idx_col].set_title('ODE R2: {:.3f}'.format(r_value**2))
                                    ax[2*idx_row+1+step,idx_col].imshow(predicted_to_plot_ode[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row+1+step,idx_col].axis('off')

                                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ide[c,:].flatten())
                                    ax[2*idx_row+2+step,idx_col].set_title('IDE R2: {:.3f}'.format(r_value**2))
                                    ax[2*idx_row+2+step,idx_col].imshow(predicted_to_plot_ide[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row+2+step,idx_col].axis('off')
                                    c+=1
                                step += 1
                            fig.tight_layout()
                            plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_ode_vs_ide_rec'+str(i)))




                    if args.plot_F_func is True:
                        
                        F_out = to_np(F_func.forward(z_test,ts_test))
                        
                        n_plots_x = int(np.ceil(np.sqrt(F_out.shape[1])))
                        n_plots_y = int(np.floor(np.sqrt(F_out.shape[1])))
                        fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=(10, 10), sharex=True, dpi=100, facecolor='w', edgecolor='k')
                        ax=ax.ravel()
                        for idx in range(args.num_dim_plot):
                            
                            ax[idx].scatter(to_np(times)[:],F_out[:,idx],label='F_out',c='blue', alpha=0.5)
                            ax[idx].set_xlabel("Time")
                            ax[idx].set_ylabel("F"+str(idx))
                            
                            ax[idx].legend()
                            
                        fig.tight_layout()

                        plt.savefig(os.path.join(path_to_save_plots, 'plot_F_func'+str(i)))

                    del obs_test, ts_test, z_test, z_p
                    
                    plt.close('all')

            end_i = time.time()
            
            if args.kernel_split is True:
                model_state = {
                        'epoch': i + 1,
                        'state_dict': kernel.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }
            else:
                model_state = {
                        'epoch': i + 1,
                        'state_dict': G_NN.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            if len(val_loader)>0:
                if args.free_func_nn is False:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, G_NN, kernel, F_func, None)
                else:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, G_NN, kernel, F_func, f_func)
            else: 
                if args.free_func_nn is False:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, G_NN, kernel, F_func, None)
                else:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, G_NN, kernel, F_func, f_func)

            #lr_scheduler(loss_validation)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break


        end = time.time()
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')
        
        model.eval()
        with torch.no_grad():
            obs_test, ts_test, ids_test = next(iter(loader_test))
            ids_test, indices = torch.sort(ids_test)
            
            obs_test = obs_test[indices,:]
            ts_test = ts_test[indices]
            

            obs_test = obs_test.to(args.device)
            ts_test = ts_test.to(args.device)
            ids_test = ids_test.to(args.device)
            
            z_test = model(obs_test[0],ts_test, return_whole_sequence=True).squeeze()
            
            
            z_p = to_np(z_test)
            obs_print = to_np(obs_test)
            
             
            data_to_plot = obs_print[:,:]  
            predicted_to_plot = z_p[:,:]
            
            
            plt.figure(figsize=(10,10),dpi=200,facecolor='w')
            plt.scatter(data_to_plot[:,0],data_to_plot[:,1],label='Data')
            plt.plot(predicted_to_plot[:,0],predicted_to_plot[:,1],label='Model',c='red',linewidth=3)
            plt.xlabel("dim 0",fontsize=20)
            plt.ylabel("dim 1",fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20) 
            
            

            all_r2_scores = []
            all_mse_scores = []

            for idx_frames in range(len(data_to_plot)):
                _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                all_r2_scores.append(r_value)
                
                tmp_mse_loss = mean_squared_error(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                all_mse_scores.append(tmp_mse_loss)

            fig,ax = plt.subplots(2,1, figsize=(15,5), sharex=True, facecolor='w')
            ax[0].plot(np.arange(len(all_r2_scores)),all_r2_scores)
            ax[1].plot(np.arange(len(all_mse_scores)),all_mse_scores)
            ax[1].set_xlabel("Frames")
            ax[0].set_ylabel("R2")
            ax[1].set_ylabel("MSE")
            fig.tight_layout()
            
            print('R2: ',all_r2_scores)
            print('MSE: ',all_mse_scores)
            
            print('R2: ',all_r2_scores)
            print('MSE: ',all_mse_scores)
            
            _, _, r_value_seq, _, _ = scipy.stats.linregress(data_to_plot[:,:].flatten(), predicted_to_plot[:,:].flatten())
            mse_loss = mean_squared_error(data_to_plot[:,:].flatten(), predicted_to_plot[:,:].flatten())
            
            print('R2:',r_value_seq)
            print('MSE:',mse_loss)
        

def Full_IE_experiment_multiple_init_cond(G_NN, kernel, F_func, f_func, Data, time_seq, index_np, args, extrapolation_points): # experiment_name, plot_freq=1):
    # scaling_factor=1
    
    #metadata for saving checkpoints
    if args.model=='nie': 
        str_model_name = "nie"
    elif args.model=='node': 
        str_model_name = "node"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)

    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else: 
            num_experiments = [int(i[3:]) for i in txt]
            num_experiments = np.array(num_experiments).max()
         # -- logger location
        writer = SummaryWriter(os.path.join(path_to_experiment,'run'+str(num_experiments+1)))
        print('writer.log_dir: ',writer.log_dir)
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)
            
        with open(os.path.join(writer.log_dir,'commandline_args.txt'), 'w') as f:
            for key, value in args.__dict__.items(): 
                f.write('%s:%s\n' % (key, value))



    obs = Data
    times = time_seq
    

    
    if args.kernel_split is True:
        if args.kernel_type_nn is True and args.free_func_nn is True:
            All_parameters = list(F_func.parameters()) + list(kernel.parameters()) + list(f_func.parameters())
        elif args.kernel_type_nn is True:
            All_parameters = list(F_func.parameters()) + list(kernel.parameters())
    else:
        All_parameters = G_NN.parameters()
    
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0,last_epoch=-1)# Emanuele's version
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,last_epoch=-1) #My first version
    #scheduler = LRScheduler(optimizer,patience = 20,min_lr=1e-12,factor=0.1)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0,last_epoch=-1)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        if args.free_func_nn is True:
            G_NN, optimizer, scheduler, kernel, F_func, f_func = load_checkpoint(path, G_NN, optimizer, scheduler, kernel, F_func, f_func)
        else:
            G_NN, optimizer, scheduler, kernel, F_func, f_ = load_checkpoint(path, G_NN, optimizer, scheduler, kernel, F_func, None)
            f_func = f_func


    
    if args.mode=='train':
        #lr_scheduler = LRScheduler(optimizer,patience = 50,min_lr=1e-5,factor=0.1)
        early_stopping = EarlyStopping(patience=1000,min_delta=0)

        # Loss_print = []
        # Val_Loss = []
        all_train_loss=[]
        all_val_loss=[]
        
        
        Data_splitting_indices = Train_val_split(np.copy(index_np)[1:],args.training_split)
        Train_Data_indices = Data_splitting_indices.train_IDs()
        Val_Data_indices = Data_splitting_indices.val_IDs()
        print('\nlen(Train_Data_indices): ',len(Train_Data_indices))
        print('Train_Data_indices: ',Train_Data_indices)
        print('\nlen(Val_Data_indices): ',len(Val_Data_indices))
        print('Val_Data_indices: ',Val_Data_indices)
        
        # Train Neural IDE
        get_times = Select_times_function(times,extrapolation_points)

        save_best_model = SaveBestModel()
        start = time.time()
        
        split_size = int(args.training_split*obs.size(0))
        
        for i in range(args.epochs):
            
            
            if args.kernel_split is True:
                kernel.train()
            else: 
                G_NN.train()
            start_i = time.time()
            print('Epoch:',i)
            # GPUtil.showUtilization()
            counter=0
            train_loss = 0.0
            
            
            for j in tqdm(range(obs.size(0)-split_size)):
                
                Dataset_train = Dynamics_Dataset(obs[j,:,:],times)
                #Dataset_val = Dynamics_Dataset(obs[j-split_size,:,:],times)
                # loader = torch.utils.data.DataLoader(Dataset, batch_size = batch_size)
                # Dataset_val = Val_Dynamics_Dataset(Data,Val_Data_indices,times)

                # For the sampler
                train_sampler = SubsetRandomSampler(Train_Data_indices)
                #valid_sampler = SubsetRandomSampler(Val_Data_indices)

                # loader_val = torch.utils.data.DataLoader(Dataset, batch_size = args.batch_size)

                dataloaders = {'train': torch.utils.data.DataLoader(Dataset_train, sampler=train_sampler, batch_size = int(args.batch_size-1), drop_last=True),
                               #'val': torch.utils.data.DataLoader(Dataset_val, sampler=valid_sampler, batch_size = args.batch_size, drop_last=True),
                               #'test': torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np))),
                              }

                train_loader = dataloaders['train']
                #val_loader = dataloaders['val']
                #loader_test = dataloaders['test']

            #for obs_, ts_, ids_ in tqdm(train_loader): 
                obs_, ts_, ids_ = next(iter(train_loader))
                obs_ = obs_.to(args.device)
                ts_ = ts_.to(args.device)
                ids_ = ids_.to(args.device)
                # obs_, ts_, ids_ = next(iter(loader))

                ids_, indices = torch.sort(ids_)
                obs_ = obs_[indices,:]
                obs_ = torch.cat([obs[j,:1,:],obs_])
                ts_ = ts_[indices]
                ts_ = torch.cat([times[:1],ts_])
                if args.perturbation_to_obs0 is not None:
                       perturb = torch.normal(mean=torch.zeros(obs_.shape[1]).to(args.device),
                                              std=args.std_noise)#args.perturbation_to_obs0*obs_[:3,:].std(dim=0))
                else:
                    perturb = torch.zeros_like(obs_[0]).to(args.device)
                # print('obs_[:5]: ',obs_[:5])
                # print('ids_[:5]: ',ids_[:5])
                # print('ts_[:5]: ',ts_[:5])

                # print('obs_: ',obs_)
                # print('ids_: ',ids_)
                # print('ts_: ',ts_)

                # obs_, ts_ = obs_.squeeze(1), ts_.squeeze(1)
                
                
                y0 = obs_[0].flatten().to(device)
                c_func = lambda x: y0+f_func(torch.Tensor([x]).to(device))-f_func(torch.Tensor([0]).to(device))
                
                z_ = IESolver_monoidal(
                        x = ts_.to(device),
                        dim = args.dim, 
                        c = c_func, 
                        d = lambda x,y: torch.Tensor([1]).to(device), 
                        k = kernel, 
                        f = F_func,
                        G = G_NN,
                        lower_bound = lambda x: torch.Tensor([ts_[0]]).to(device),
                        upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                        max_iterations = args.max_iterations,
                        kernel_nn=True,
                        kernel_split = args.kernel_split,
                        G_nn = args.G_NN,
                        integration_dim = 0,
                        mc_samplings=1000,
                        num_internal_points = args.num_internal_points
                        ).solve()


                #loss_ts_ = get_times.select_times(ts_)[1]
                loss = F.mse_loss(z_[:,:], obs_.detach()[:,:]) #Original 
                # print('z_[:,:].to(args.device): ',z_[:,:].to(args.device))
                # print('obs_.to(args.device).detach()[:,:]: ',obs_.to(args.device).detach()[:,:])
                # loss = F.mse_loss(z_[:,:].to(args.device), obs_.to(args.device).detach()[:,:]) #Original 

        
                # ###############################
                # Loss_print.append(to_np(loss))
                # ###############################

                optimizer.zero_grad()
                loss.backward()#(retain_graph=True)
                optimizer.step()

                # n_iter += 1
                counter += 1
                train_loss += loss.item()
                
                if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                    scheduler.step()
                

            train_loss /= counter
            all_train_loss.append(train_loss)
            
            
            if  split_size==0 and args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
            
            del loss, train_loss, obs_, ts_, z_


            ## Validating
            if args.kernel_split is True:
                kernel.eval()
            else: 
                G_NN.eval()
            with torch.no_grad():

                    #Only do this if there is a validation dataset
                
                val_loss = 0.0
                counter = 0
                if split_size>0:
                    # for images, _, _, _, _ in tqdm(val_loader):   # frames, timevals, angular_velocity, mass_height, mass_xpos
                    for j in tqdm(range(obs.size(0)-split_size,obs.size(0))):
                        
                        valid_sampler = SubsetRandomSampler(Train_Data_indices)
                        Dataset_val = Dynamics_Dataset(obs[j,:,:],times)
                        val_loader = torch.utils.data.DataLoader(Dataset_val, sampler=valid_sampler, batch_size = int(args.batch_size-1), drop_last=True)
                    
                    #for obs_val, ts_val, ids_val in tqdm(val_loader):
                        obs_val, ts_val, ids_val = next(iter(val_loader))
                        obs_val = obs_val.to(args.device)
                        ts_val = ts_val.to(args.device)
                        ids_val = ids_val.to(args.device)

                        ids_val, indices = torch.sort(ids_val)
                        # print('indices: ',indices)
                        obs_val = obs_val[indices,:]
                        ts_val = ts_val[indices]
                        
                        obs_val = torch.cat([obs[j,:1,:],obs_val])
                        ts_val = torch.cat([times[:1],ts_val])

                        #Concatenate the first point of the train minibatch
                        # obs_[0],ts_
                        # print('\n In validation mode...')
                        # print('obs_[:5]: ',obs_[:5])
                        # print('ids_[:5]: ',ids_[:5])
                        # print('ts_[:5]: ',ts_[:5])
                        # print('ts_[0]:',ts_[0])

                        ## Below is to add initial data point to val
                        #obs_val = torch.cat((obs_[0][None,:],obs_val))
                        #ts_val = torch.hstack((ts_[0],ts_val))
                        #ids_val = torch.hstack((ids_[0],ids_val))

                        # obs_val, ts_val, ids_val = next(iter(loader_val))
                        # print('obs_val.shape: ',obs_val.shape)
                        # print('ids_val: ',ids_val)
                        # print('ts_val: ',ts_val)

                        # obs_val, ts_val = obs_val.squeeze(1), ts_val.squeeze(1)
                        
                        y0 = obs_val[0].flatten().to(device)
                        c_func = lambda x: y0+f_func(torch.Tensor([x]).to(device))-f_func(torch.Tensor([0]).to(device))

                        z_val = IESolver_monoidal(
                                    x = ts_val.to(device), 
                                    c = c_func, 
                                    d = lambda x,y: torch.Tensor([1]).to(device), 
                                    k = kernel, 
                                    f = F_func,
                                    G = G_NN,
                                    lower_bound = lambda x: torch.Tensor([ts_val[0]]).to(device),
                                    upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                                    max_iterations = args.max_iterations,
                                    kernel_nn=True,
                                    kernel_split = args.kernel_split,
                                    G_nn = args.G_NN,
                                    integration_dim = 0,
                                    mc_samplings=1000,
                                    num_internal_points = args.num_internal_points
                                    ).solve()

                        #validation_ts_ = get_times.select_times(ts_val)[1]
                        loss_validation = F.mse_loss(z_val[:,:], obs_val.detach()[:,:])
                        # Val_Loss.append(to_np(loss_validation))

                        del obs_val, ts_val, z_val
                        
                        counter += 1
                        val_loss += loss_validation.item()
                        
                        del loss_validation

                        #LRScheduler(loss_validation)
                        if args.lr_scheduler == 'ReduceLROnPlateau':
                            scheduler.step(val_loss)
                else: counter += 1

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            writer.add_scalar('train_loss', all_train_loss[-1], global_step=i)
            if len(all_val_loss)>0:
                writer.add_scalar('val_loss', all_val_loss[-1], global_step=i)
            if args.lr_scheduler == 'ReduceLROnPlateau':
                writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], global_step=i)
            elif args.lr_scheduler == 'CosineAnnealingLR':
                writer.add_scalar('Epoch/learning_rate', scheduler.get_last_lr()[0], global_step=i)

            
            with torch.no_grad():
                if i % args.plot_freq == 0 and args.test is True:
                    if obs.size(2)>2:
                        pca_proj = PCA(n_components=2)
                    for j in tqdm(range(obs.size(0))):
                        Dataset_all = Test_Dynamics_Dataset(Data[j,:,:],times)
                        loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np)))

                        obs_test, ts_test, ids_test = next(iter(loader_test))

                        ids_test, indices = torch.sort(ids_test)
                        # print('indices: ',indices)
                        obs_test = obs_test[indices,:]
                        ts_test = ts_test[indices]
                        # print('obs_test.shape: ',obs_test.shape)
                        # print('ids_test: ',ids_test)
                        # print('ts_test: ',ts_test)


                        obs_test = obs_test.to(args.device)
                        ts_test = ts_test.to(args.device)
                        ids_test = ids_test.to(args.device)
                        # print('obs_test.shape: ',obs_test.shape)
                        # print('ids_test: ',ids_test)
                        # print('ts_test: ',ts_test)
                        # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                        
                        y0 = obs_test[0].flatten().to(device)
                        c_func = lambda x: y0+f_func(torch.Tensor([x]).to(device))-f_func(torch.Tensor([0]).to(device))
                            
                        z_test = IESolver_monoidal(
                                        x = ts_test.to(device), 
                                        c = c_func, 
                                        d = lambda x,y: torch.Tensor([1]).to(device), 
                                        k = kernel, 
                                        f = F_func,
                                        G = G_NN,
                                        lower_bound = lambda x: torch.Tensor([ts_test[0]]).to(device),
                                        upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                                        max_iterations = args.max_iterations,
                                        kernel_nn=True,
                                        kernel_split = args.kernel_split,
                                        G_nn = args.G_NN,
                                        integration_dim = 0,
                                        mc_samplings=1000,
                                        num_internal_points = args.num_internal_points
                                        ).solve()
                        #print('Parameters are:',ide_trained.parameters)
                        #print(list(All_parameters))
                        plt.figure(0, figsize=(8,8),facecolor='w')
                        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
                        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
                        plt.plot(np.log10(all_train_loss),label='Train loss')
                        if split_size>0:
                            plt.plot(np.log10(all_val_loss),label='Val loss')
                        plt.xlabel("Epoch")
                        plt.ylabel("MSE Loss")
                        # timestr = time.strftime("%Y%m%d-%H%M%S")
                        plt.savefig(os.path.join(path_to_save_plots,'losses'))

                        new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                        plt.figure(figsize=(8,8),facecolor='w')
                        z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                        z_p = to_np(z_p)
                        obs_print = to_np(obs_test[:,:])

                        if obs.size(2)>2:
                            z_p = pca_proj.fit_transform(z_p)
                            obs_print = pca_proj.fit_transform(obs_print)                    

                        plt.figure(1, facecolor='w')
                        plt.plot(z_p[:,0],z_p[:,1],c='r', label='model')

                        # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                        plt.scatter(obs_print[:,0],obs_print[:,1],label='Data',c='blue', alpha=0.5)
                        plt.xlabel("dim 0")
                        plt.ylabel("dim 1")
                        #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                        plt.legend()
                        # plt.show()
                        # timestr = time.strftime("%Y%m%d-%H%M%S")
                        plt.savefig(os.path.join(path_to_save_plots,'plot_dim0vsdim1_epoch'+str(i)+'_'+str(j)))


                        if 'calcium_imaging' in args.experiment_name:
                            # Plot the first 20 frames
                            data_to_plot = obs_print[:20,:]*args.scaling_factor #Get the first 10 samples for a test 
                            predicted_to_plot = z_p[:20,:]*args.scaling_factor
                            data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                            predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                            predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                            data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                            fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                            c=0
                            for idx_row in range (2): 
                                for idx_col in range(10):
                                    ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row,idx_col].axis('off')
                                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                    ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                    ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row+1,idx_col].axis('off')
                                    c+=1
                            fig.tight_layout()
                            plt.savefig(os.path.join(path_to_save_plots, 'plot_first20frame_rec'+str(i)))


                            # Plot the last 20 frames  
                            data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                            predicted_to_plot = z_p[-20:,:]*args.scaling_factor
                            data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                            predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                            predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                            data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                            fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                            c=0
                            for idx_row in range (2): 
                                for idx_col in range(10):
                                    ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row,idx_col].axis('off')
                                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                    ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                    ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row+1,idx_col].axis('off')
                                    c+=1
                            fig.tight_layout()
                            plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_rec'+str(i)))


                            #Plot the R2 and MSE loss between the original data and the predicted overtime. 
                            data_to_plot = obs_print[:,:]*args.scaling_factor #Get the first 10 samples for a test 
                            predicted_to_plot = z_p[:,:]*args.scaling_factor
                            data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                            predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                            all_r2_scores = []
                            all_mse_scores = []

                            for idx_frames in range(len(data_to_plot)):
                                _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                                all_r2_scores.append(r_value)
                                # print('data_to_plot[idx_frames,:].flatten().shape: ',data_to_plot[idx_frames,:].flatten().shape)
                                # print('predicted_to_plot[idx_frames,:].flatten().shape: ',predicted_to_plot[idx_frames,:].flatten().shape)
                                tmp_mse_loss = mean_squared_error(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                                all_mse_scores.append(tmp_mse_loss)

                            fig,ax = plt.subplots(2,1, figsize=(15,5), sharex=True, facecolor='w')
                            ax[0].plot(np.arange(len(all_r2_scores)),all_r2_scores)
                            ax[1].plot(np.arange(len(all_mse_scores)),all_mse_scores)
                            ax[1].set_xlabel("Frames")
                            ax[0].set_ylabel("R2")
                            ax[1].set_ylabel("MSE")
                            fig.tight_layout()
                            plt.savefig(os.path.join(path_to_save_plots, 'plot_performance_rec'+str(i)))

                            #Plot integral and ode part separated
                            if ode_func is not None and F_func is not None:
                                Trained_Data_ode = odeint(ode_func,torch.Tensor(obs_print[0,:]).flatten().to(args.device),times.to(args.device),rtol=1e-4,atol=1e-4)
                                Trained_Data_ode_print = to_np(Trained_Data_ode)
                                Trained_Data_integral_print  = z_p - Trained_Data_ode_print
                                # print('Trained_Data_integral_print.max():',np.abs(Trained_Data_integral_print).max())
                                # print('Trained_Data_ode_print.max():',np.abs(Trained_Data_ode_print).max())

                                data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                                predicted_to_plot_ode = Trained_Data_ode_print[-20:,:]*args.scaling_factor
                                predicted_to_plot_ide = Trained_Data_integral_print[-20:,:]*args.scaling_factor
                                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                                predicted_to_plot_ode = args.fitted_pca.inverse_transform(predicted_to_plot_ode)
                                predicted_to_plot_ide = args.fitted_pca.inverse_transform(predicted_to_plot_ide)

                                predicted_to_plot_ode = predicted_to_plot_ode.reshape(predicted_to_plot_ode.shape[0],184, 208) # Add the original frame dimesion as input
                                predicted_to_plot_ide = predicted_to_plot_ide.reshape(predicted_to_plot_ide.shape[0],184, 208)
                                data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                                fig,ax = plt.subplots(6,10, figsize=(15,8), facecolor='w')
                                c=0
                                step = 0
                                for idx_row in range (2): 
                                    for idx_col in range(10):
                                        ax[2*idx_row+step,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row+step,idx_col].axis('off')

                                        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ode[c,:].flatten())
                                        ax[2*idx_row+1+step,idx_col].set_title('ODE R2: {:.3f}'.format(r_value**2))
                                        ax[2*idx_row+1+step,idx_col].imshow(predicted_to_plot_ode[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row+1+step,idx_col].axis('off')

                                        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ide[c,:].flatten())
                                        ax[2*idx_row+2+step,idx_col].set_title('IDE R2: {:.3f}'.format(r_value**2))
                                        ax[2*idx_row+2+step,idx_col].imshow(predicted_to_plot_ide[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row+2+step,idx_col].axis('off')
                                        c+=1
                                    step += 1
                                fig.tight_layout()
                                plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_ode_vs_ide_rec'+str(i)))




                        if F_func is not None and args.print_F_func is True:
                            # fig,ax = plt.subplots(1,1, figsize=(8,8), facecolor='w')
                            F_out = to_np(F_func.forward(z_test,ts_test))
                            # print('F_out.shape: ',F_out.shape)
                            # if F_out.shape[1]>2: 
                            #     # reducer = umap.UMAP(n_components=2, random_state=1) 
                            #     reducer = PCA(n_components=2)
                            #     F_out = reducer.fit_transform(F_out)
                            n_plots_x = int(np.ceil(np.sqrt(F_out.shape[1])))
                            n_plots_y = int(np.floor(np.sqrt(F_out.shape[1])))
                            fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=(10, 10), sharex=True, dpi=100, facecolor='w', edgecolor='k')
                            ax=ax.ravel()
                            for idx in range(args.num_dim_plot):
                                # plt.scatter(to_np(times)[:extrapolation_points],obs_print[:extrapolation_points,0]*scaling_factor,label='Data',c='blue')
                                ax[idx].scatter(to_np(times)[:],F_out[:,idx],label='F_out',c='blue', alpha=0.5)
                                ax[idx].set_xlabel("Time")
                                ax[idx].set_ylabel("F"+str(idx))
                                #plt.scatter(to_np(times)[extrapolation_points:],obs_print[extrapolation_points:,0,0],label='Data extr',c='red')
                                ax[idx].legend()
                                # timestr = time.strftime("%Y%m%d-%H%M%S")
                            fig.tight_layout()

                            # ax.plot(F_out[:,0],F_out[:,1])
                            # ax.scatter(F_out[:,0],F_out[:,1], c = np.arange(len(F_out)))
                            # ax.set_xlabel("F_0")
                            # ax.set_ylabel("F_1")
                            plt.savefig(os.path.join(path_to_save_plots, 'plot_F_func'+str(i)))
                        
                        del obs_test, ts_test, z_test, z_p

                        plt.close('all')

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            if args.kernel_split is True:
                model_state = {
                        'epoch': i + 1,
                        'state_dict': kernel.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }
            else:
                model_state = {
                        'epoch': i + 1,
                        'state_dict': G_NN.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            if split_size>0:
                if args.free_func_nn is False:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, G_NN, kernel, F_func, None)
                else:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, G_NN, kernel, F_func, f_func)
            else: 
                if args.free_func_nn is False:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, G_NN, kernel, F_func, None)
                else:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, G_NN, kernel, F_func, f_func)

            #lr_scheduler(loss_validation)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break


        end = time.time()
        # print(f"Training time: {(end-start)/60:.3f} minutes")
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),Loss_print)
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),Val_Loss)
        # # plt.savefig('trained.png')
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'trained'+timestr))
        # # plt.show()
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'final_losses'+timestr))
        # # plt.show()
    elif args.mode=='evaluate':
        print('Running in evaluation mode')
        ## Validating
        #model.eval()
        with torch.no_grad():
            splitting_size = int(args.training_split*Data.size(0))
            all_r2_scores = []
            all_mse = []
            for j in tqdm(range(Data.size(0)-splitting_size)):
                Dataset_all = Test_Dynamics_Dataset(Data[j,:,:],times)
                loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np)))

                obs_test, ts_test, ids_test = next(iter(loader_test))
                ids_test, indices = torch.sort(ids_test)
                # print('indices: ',indices)
                obs_test = obs_test[indices,:]
                ts_test = ts_test[indices]
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)


                obs_test = obs_test.to(args.device)
                ts_test = ts_test.to(args.device)
                ids_test = ids_test.to(args.device)
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)
                # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                
                y0 = obs_test[0].flatten().to(device)
                c_func = lambda x: y0+f_func(torch.Tensor([x]).to(device))-f_func(torch.Tensor([0]).to(device))
                
                z_test = IESolver_monoidal(
                                        x = ts_test.to(device), 
                                        c = lambda x: c_func, 
                                        d = lambda x,y: torch.Tensor([1]).to(device), 
                                        k = kernel, 
                                        f = F_func,
                                        G = G_NN,
                                        lower_bound = lambda x: torch.Tensor([ts_test[0]]).to(device),
                                        upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                                        max_iterations = args.max_iterations,
                                        kernel_nn=True,
                                        kernel_split = args.kernel_split,
                                        G_nn = args.G_NN,
                                        integration_dim = 0,
                                        mc_samplings=1000,
                                        num_internal_points = args.num_internal_points
                                        ).solve()
                
                # z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_test)
                obs_print = to_np(obs_test)
                
                new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                plt.figure(figsize=(8,8),facecolor='w')
                z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_p)

                plt.figure(1, facecolor='w')
                plt.plot(z_p[:,0],z_p[:,1],c='r', label='model')
                obs_print = to_np(obs_test[:,:])
                # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                plt.scatter(obs_print[:,0],obs_print[:,1],label='Data',c='blue', alpha=0.5)
                plt.xlabel("dim 0")
                plt.ylabel("dim 1")
                #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                plt.legend()
                
                _, _, r_value, _, _ = scipy.stats.linregress(z_p[:,:].flatten(),obs_print[:,:].flatten())
                mse_value = mean_squared_error(z_p[:,:].flatten(),obs_print[:,:].flatten())
                
                print('R2:',r_value)
                print('MSE:',mse_value)
                
                all_r2_scores.append(r_value)
                all_mse.append(mse_value)
            
            plt.figure(2,facecolor='w')
            plt.plot(np.linspace(0,len(all_r2_scores),len(all_r2_scores)),all_r2_scores)
            plt.xlabel("Dynamics")
            plt.ylabel("R2")
            plt.legend()
            
            plt.figure(3,facecolor='w')
            plt.plot(np.linspace(0,len(all_mse),len(all_mse)),all_mse)
            plt.xlabel("Dynamics")
            plt.ylabel("MSE")
            plt.legend()
            
            print("Average R2:",sum(all_r2_scores)/len(all_r2_scores))
            print("Average MSE:",sum(all_mse)/len(all_mse))
                
            for j in tqdm(range(Data.size(0)-splitting_size,Data.size(0))):
                Dataset_all = Test_Dynamics_Dataset(Data[j,:,:],times)
                loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np)))

                obs_test, ts_test, ids_test = next(iter(loader_test))
                ids_test, indices = torch.sort(ids_test)
                # print('indices: ',indices)
                obs_test = obs_test[indices,:]
                ts_test = ts_test[indices]
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)


                obs_test = obs_test.to(args.device)
                ts_test = ts_test.to(args.device)
                ids_test = ids_test.to(args.device)
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)
                # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                
                y0 = obs_test[0].flatten().to(device)
                c_func = lambda x: y0+f_func(torch.Tensor([x]).to(device))-f_func(torch.Tensor([0]).to(device))
                    
                z_test = IESolver_monoidal(
                                        x = ts_test.to(device), 
                                        c = c_func, 
                                        d = lambda x,y: torch.Tensor([1]).to(device), 
                                        k = kernel, 
                                        f = F_func,
                                        G = G_NN,
                                        lower_bound = lambda x: torch.Tensor([ts_test[0]]).to(device),
                                        upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                                        max_iterations = args.max_iterations,
                                        kernel_nn=True,
                                        kernel_split = args.kernel_split,
                                        G_nn = args.G_NN,
                                        integration_dim = 0,
                                        mc_samplings=1000,
                                        num_internal_points = args.num_internal_points
                                        ).solve()

                # z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_test)
                obs_print = to_np(obs_test)
                
                new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                plt.figure(figsize=(8,8),facecolor='w')
                z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_p)

                plt.figure(2, facecolor='w')
                plt.plot(z_p[:,0],z_p[:,1],c='r', label='model')
                obs_print = to_np(obs_test[:,:])
                # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                plt.scatter(obs_print[:,0],obs_print[:,1],label='Data',c='blue', alpha=0.5)
                plt.xlabel("dim 0")
                plt.ylabel("dim 1")
                #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                plt.legend()
                      
                print(scipy.stats.linregress(z_p[:,:].flatten(),obs_print[:,:].flatten()))
                print(mean_squared_error(z_p[:,:].flatten(),obs_print[:,:].flatten()))
                
                '''
                # Plot the last 20 frames  
                data_to_plot = obs_print[:,:]#*args.scaling_factor #Get the first 10 samples for a test 
                predicted_to_plot = z_p[:,:]*args.scaling_factor
                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                c=0
                for idx_row in range (2): 
                    for idx_col in range(10):
                        ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                        ax[2*idx_row,idx_col].axis('off')
                        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                        ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                        ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                        ax[2*idx_row+1,idx_col].axis('off')
                        c+=1
                fig.tight_layout()

                #Plot the R2 and MSE loss between the original data and the predicted overtime. 
                data_to_plot = obs_print[:,:]*args.scaling_factor #Get the first 10 samples for a test 
                predicted_to_plot = z_p[:,:]*args.scaling_factor
                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                all_r2_scores = []
                all_mse_scores = []

                for idx_frames in range(len(data_to_plot)):
                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                    all_r2_scores.append(r_value)
                    # print('data_to_plot[idx_frames,:].flatten().shape: ',data_to_plot[idx_frames,:].flatten().shape)
                    # print('predicted_to_plot[idx_frames,:].flatten().shape: ',predicted_to_plot[idx_frames,:].flatten().shape)
                    tmp_mse_loss = mean_squared_error(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                    all_mse_scores.append(tmp_mse_loss)

                fig,ax = plt.subplots(2,1, figsize=(15,5), sharex=True, facecolor='w')
                ax[0].plot(np.arange(len(all_r2_scores)),all_r2_scores)
                ax[1].plot(np.arange(len(all_mse_scores)),all_mse_scores)
                ax[1].set_xlabel("Frames")
                ax[0].set_ylabel("R2")
                ax[1].set_ylabel("MSE")
                fig.tight_layout()

                print('R2: ',all_r2_scores)
                print('MSE: ',all_mse_scores)
                '''

                
def Full_experiment_AttentionalIE_multiple_init_cond(model, Data, time_seq, index_np, mask, times, args, extrapolation_points): # experiment_name, plot_freq=1):
    # scaling_factor=1
    
    
    #metadata for saving checkpoints
    if args.model=='nie': 
        str_model_name = "nie"
    elif args.model=='node': 
        str_model_name = "node"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)

    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else: 
            num_experiments = [int(i[3:]) for i in txt]
            num_experiments = np.array(num_experiments).max()
         # -- logger location
        #writer = SummaryWriter(os.path.join(path_to_experiment,'run'+str(num_experiments+1)))
        #print('writer.log_dir: ',writer.log_dir)
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)
            
        #with open(os.path.join(writer.log_dir,'commandline_args.txt'), 'w') as f:
        #    for key, value in args.__dict__.items(): 
        #        f.write('%s:%s\n' % (key, value))



    obs = Data
    times = time_seq
    

    All_parameters = model.parameters()
    
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0,last_epoch=-1)# Emanuele's version
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,last_epoch=-1) #My first version
    #scheduler = LRScheduler(optimizer,patience = 20,min_lr=1e-12,factor=0.1)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0,last_epoch=-1)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        
        model, optimizer, scheduler, kernel, F_func, f_func = load_checkpoint(path, model, optimizer, scheduler, None, None,  None)


    
    if args.mode=='train':
        #lr_scheduler = LRScheduler(optimizer,patience = 50,min_lr=1e-5,factor=0.1)
        early_stopping = EarlyStopping(patience=1000,min_delta=0)

        # Loss_print = []
        # Val_Loss = []
        all_train_loss=[]
        all_val_loss=[]
        
            
        Data_splitting_indices = Train_val_split(np.copy(index_np)[1:],args.training_split)
        Train_Data_indices = Data_splitting_indices.train_IDs()
        Val_Data_indices = Data_splitting_indices.val_IDs()
        print('\nlen(Train_Data_indices): ',len(Train_Data_indices))
        print('Train_Data_indices: ',Train_Data_indices)
        print('\nlen(Val_Data_indices): ',len(Val_Data_indices))
        print('Val_Data_indices: ',Val_Data_indices)
        
        # Train Neural IDE
        get_times = Select_times_function(times,extrapolation_points)

        save_best_model = SaveBestModel()
        start = time.time()
        
        split_size = int(args.training_split*obs.size(0))
        
        
        for i in range(args.epochs):
            
            if args.support_tensors is True or args.support_test is True:
                if args.combine_points is True:
                    sampled_tensors,real_idx = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(0, 1))
                    temp_sampled_tensors = sampled_tensors
                    sampled_tensors = sampled_tensors.to(device)
                    #Check if there are duplicates and resample if there are
                    sampled_tensors = torch.cat([times,sampled_tensors])
                    dup=np.array([0])
                    while dup.size != 0:
                        u, c = np.unique(temp_sampled_tensors, return_counts=True)
                        dup = u[c > 1]
                        if dup.size != 0:
                            sampled_tensors,real_idx = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(0, 1))
                            sampled_tensors = sampled_tensors.to(device)
                            sampled_tensors = torch.cat([times,sampled_tensors])
                    dummy_times=sampled_tensors
                    real_idx=real_idx[:times.size(0)]
                if args.combine_points is False:
                        dummy_times = torch.linspace(times[0],times[-1],args.sampling_points)
            
            model.train()
            
            start_i = time.time()
            print('Epoch:',i)
            # GPUtil.showUtilization()
            counter=0
            train_loss = 0.0


            for j in tqdm(range(obs.size(0)-split_size)):
                
                Dataset_train = Dynamics_Dataset(obs[j,:,:],times)
                #Dataset_val = Dynamics_Dataset(obs[j-split_size,:,:],times)
                # loader = torch.utils.data.DataLoader(Dataset, batch_size = batch_size)
                # Dataset_val = Val_Dynamics_Dataset(Data,Val_Data_indices,times)

                # For the sampler
                train_sampler = SubsetRandomSampler(Train_Data_indices)
                #valid_sampler = SubsetRandomSampler(Val_Data_indices)

                # loader_val = torch.utils.data.DataLoader(Dataset, batch_size = args.batch_size)

                dataloaders = {'train': torch.utils.data.DataLoader(Dataset_train, sampler=train_sampler, batch_size = int(args.batch_size-1), drop_last=True),
                               #'val': torch.utils.data.DataLoader(Dataset_val, sampler=valid_sampler, batch_size = args.batch_size, drop_last=True),
                               #'test': torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np))),
                              }

                train_loader = dataloaders['train']
                #val_loader = dataloaders['val']
                #loader_test = dataloaders['test']

            #for obs_, ts_, ids_ in tqdm(train_loader): 
                obs_, ts_, ids_ = next(iter(train_loader))
                obs_ = obs_.to(args.device)
                ts_ = ts_.to(args.device)
                ids_ = ids_.to(args.device)
                # obs_, ts_, ids_ = next(iter(loader))

                ids_, indices = torch.sort(ids_)
                obs_ = obs_[indices,:]
                obs_ = torch.cat([obs[j,:1,:],obs_])
                ts_ = ts_[indices]
                ts_ = torch.cat([times[:1],ts_])
                if args.perturbation_to_obs0 is not None:
                       perturb = torch.normal(mean=torch.zeros(obs_.shape[1]).to(args.device),
                                              std=args.std_noise)#args.perturbation_to_obs0*obs_[:3,:].std(dim=0))
                else:
                    perturb = torch.zeros_like(obs_[0]).to(args.device)
                # print('obs_[:5]: ',obs_[:5])
                # print('ids_[:5]: ',ids_[:5])
                # print('ts_[:5]: ',ts_[:5])

                # print('obs_: ',obs_)
                # print('ids_: ',ids_)
                # print('ts_: ',ts_)

                # obs_, ts_ = obs_.squeeze(1), ts_.squeeze(1)
                
                if args.support_tensors is False:
                    z_ = Integral_attention_solver(
                            ts_.to(device),
                            obs_[0].unsqueeze(0).to(args.device),
                            sampling_points = ts_.size(0),
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            use_support=False,
                            ).solve()
                else:
                    z_ = Integral_attention_solver(
                            ts_.to(device),
                            obs_[0].unsqueeze(0).to(args.device),
                            c=None,
                            sampling_points = dummy_times.size(0),
                            support_tensors=dummy_times.to(device),
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            output_support_tensors=True
                            ).solve()
                    if args.combine_points is True:
                        z_ = z_[real_idx,:]
                    
                #loss_ts_ = get_times.select_times(ts_)[1]
                loss = F.mse_loss(z_[:,:], obs_.detach()[:,:]) #Original 
                # print('z_[:,:].to(args.device): ',z_[:,:].to(args.device))
                # print('obs_.to(args.device).detach()[:,:]: ',obs_.to(args.device).detach()[:,:])
                # loss = F.mse_loss(z_[:,:].to(args.device), obs_.to(args.device).detach()[:,:]) #Original 

                    
                # ###############################
                # Loss_print.append(to_np(loss))
                # ###############################

                optimizer.zero_grad()
                loss.backward()#(retain_graph=True)
                optimizer.step()

                # n_iter += 1
                counter += 1
                train_loss += loss.item()
                
                if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                    scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  split_size==0 and args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, obs_, ts_, z_, ids_

            ## Validating
                
            model.eval()
            with torch.no_grad():

                    #Only do this if there is a validation dataset
                
                val_loss = 0.0
                counter = 0
                if split_size>0:
                    # for images, _, _, _, _ in tqdm(val_loader):   # frames, timevals, angular_velocity, mass_height, mass_xpos
                    for j in tqdm(range(obs.size(0)-split_size,obs.size(0))):
                        
                        valid_sampler = SubsetRandomSampler(Train_Data_indices)
                        Dataset_val = Dynamics_Dataset(obs[j,:,:],times)
                        val_loader = torch.utils.data.DataLoader(Dataset_val, sampler=valid_sampler, batch_size = int(args.batch_size-1), drop_last=True)
                    
                    #for obs_val, ts_val, ids_val in tqdm(val_loader):
                        obs_val, ts_val, ids_val = next(iter(val_loader))
                        obs_val = obs_val.to(args.device)
                        ts_val = ts_val.to(args.device)
                        ids_val = ids_val.to(args.device)

                        ids_val, indices = torch.sort(ids_val)
                        # print('indices: ',indices)
                        obs_val = obs_val[indices,:]
                        ts_val = ts_val[indices]
                        
                        obs_val = torch.cat([obs[j,:1,:],obs_val])
                        ts_val = torch.cat([times[:1],ts_val])

                        #Concatenate the first point of the train minibatch
                        # obs_[0],ts_
                        # print('\n In validation mode...')
                        # print('obs_[:5]: ',obs_[:5])
                        # print('ids_[:5]: ',ids_[:5])
                        # print('ts_[:5]: ',ts_[:5])
                        # print('ts_[0]:',ts_[0])

                        ## Below is to add initial data point to val
                        #obs_val = torch.cat((obs_[0][None,:],obs_val))
                        #ts_val = torch.hstack((ts_[0],ts_val))
                        #ids_val = torch.hstack((ids_[0],ids_val))

                        # obs_val, ts_val, ids_val = next(iter(loader_val))
                        # print('obs_val.shape: ',obs_val.shape)
                        # print('ids_val: ',ids_val)
                        # print('ts_val: ',ts_val)

                        # obs_val, ts_val = obs_val.squeeze(1), ts_val.squeeze(1)
                        

                        if args.support_tensors is False:
                            z_val = Integral_attention_solver(
                                    ts_val.to(device),
                                    obs_val[0].unsqueeze(0).to(args.device),
                                    sampling_points = ts_val.size(0),
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    use_support=False,
                                    ).solve()
                        else:
                            z_val = Integral_attention_solver(
                                    ts_val.to(device),
                                    obs_[0].unsqueeze(0).to(args.device),
                                    sampling_points = dummy_times.size(0),
                                    support_tensors=dummy_times.to(device),
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    output_support_tensors=True
                                    ).solve()
                        
                            if args.combine_points is True:
                                z_val = z_val[real_idx,:]
                        #validation_ts_ = get_times.select_times(ts_val)[1]
                        loss_validation = F.mse_loss(z_val[:,:], obs_val.detach()[:,:])
                        # Val_Loss.append(to_np(loss_validation))
                        
                        del obs_val, ts_val, z_val, ids_val

                        counter += 1
                        val_loss += loss_validation.item()
                        
                        del loss_validation

                        #LRScheduler(loss_validation)
                        if args.lr_scheduler == 'ReduceLROnPlateau':
                            scheduler.step(val_loss)
                
                
                else: counter += 1

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            #writer.add_scalar('train_loss', all_train_loss[-1], global_step=i)
            #if len(all_val_loss)>0:
            #    writer.add_scalar('val_loss', all_val_loss[-1], global_step=i)
            #if args.lr_scheduler == 'ReduceLROnPlateau':
            #    writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], global_step=i)
            #elif args.lr_scheduler == 'CosineAnnealingLR':
            #    writer.add_scalar('Epoch/learning_rate', scheduler.get_last_lr()[0], global_step=i)

            
            with torch.no_grad():
                if i % args.plot_freq == 0:
                    if obs.size(2)>2:
                        pca_proj = PCA(n_components=2)
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
                    # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
                        
                    plt.plot(np.log10(all_train_loss),label='Train loss')
                    if split_size>0:
                        plt.plot(np.log10(all_val_loss),label='Val loss')
                    plt.xlabel("Epoch")
                    plt.ylabel("MSE Loss")
                    # timestr = time.strftime("%Y%m%d-%H%M%S")
                    #plt.show()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))

                    #for j in tqdm(range(obs.size(0))):
                    
                    #Dataset_all = Test_Dynamics_Dataset(Data[j,:,:],times)
                    Dataset_all = Test_Dynamics_Dataset(Data[0,:,:],times)
                    loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np)))

                    obs_test, ts_test, ids_test = next(iter(loader_test))

                    ids_test, indices = torch.sort(ids_test)
                    # print('indices: ',indices)
                    obs_test = obs_test[indices,:]
                    ts_test = ts_test[indices]
                    # print('obs_test.shape: ',obs_test.shape)
                    # print('ids_test: ',ids_test)
                    # print('ts_test: ',ts_test)


                    obs_test = obs_test.to(args.device)
                    ts_test = ts_test.to(args.device)
                    ids_test = ids_test.to(args.device)
                    # print('obs_test.shape: ',obs_test.shape)
                    # print('ids_test: ',ids_test)
                    # print('ts_test: ',ts_test)
                    # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)


                    if args.support_test is False:
                        z_test = Integral_attention_solver(
                                ts_test.to(device),
                                obs_test[0].unsqueeze(0).to(args.device),
                                sampling_points = ts_test.size(0),
                                mask=mask,
                                Encoder = model,
                                max_iterations = args.max_iterations,
                                #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                use_support=False,
                                ).solve()
                    else:
                        z_test = Integral_attention_solver(
                                ts_test.to(device),
                                obs_[0].unsqueeze(0).to(args.device),
                                sampling_points = dummy_times.size(0),
                                support_tensors=dummy_times.to(device),
                                mask=mask,
                                Encoder = model,
                                max_iterations = args.max_iterations,
                                #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                output_support_tensors=True,
                                ).solve()

                    #print('Parameters are:',ide_trained.parameters)
                    #print(list(All_parameters))


                    new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                    plt.figure(figsize=(8,8),facecolor='w')
                    z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                    z_p = to_np(z_p)
                    obs_print = to_np(obs_test[:,:])

                    if obs.size(2)>2:
                        z_p = pca_proj.fit_transform(z_p)
                        obs_print = pca_proj.fit_transform(obs_print)                    

                    plt.figure(1, facecolor='w')
                    plt.plot(z_p[:,0],z_p[:,1],c='r', label='model')
                    #plt.scatter(z_p[:,0],z_p[:,1],c='r',s=10)

                    # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                    plt.scatter(obs_print[:,0],obs_print[:,1],label='Data',c='blue', alpha=0.5)
                    plt.xlabel("dim 0")
                    plt.ylabel("dim 1")
                    #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                    plt.legend()
                    # plt.show()
                    # timestr = time.strftime("%Y%m%d-%H%M%S")
                    plt.savefig(os.path.join(path_to_save_plots,'plot_dim0vsdim1_epoch'+str(i)+'_'+str(j)))


                    if 'calcium_imaging' in args.experiment_name:
                        # Plot the first 20 frames
                        data_to_plot = obs_print[:20,:]*args.scaling_factor #Get the first 10 samples for a test 
                        predicted_to_plot = z_p[:20,:]*args.scaling_factor
                        data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                        predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                        predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                        data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                        fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                        c=0
                        for idx_row in range (2): 
                            for idx_col in range(10):
                                ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                ax[2*idx_row,idx_col].axis('off')
                                _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                ax[2*idx_row+1,idx_col].axis('off')
                                c+=1
                        fig.tight_layout()
                        plt.savefig(os.path.join(path_to_save_plots, 'plot_first20frame_rec'+str(i)))


                        # Plot the last 20 frames  
                        data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                        predicted_to_plot = z_p[-20:,:]*args.scaling_factor
                        data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                        predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                        predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                        data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                        fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                        c=0
                        for idx_row in range (2): 
                            for idx_col in range(10):
                                ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                ax[2*idx_row,idx_col].axis('off')
                                _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                ax[2*idx_row+1,idx_col].axis('off')
                                c+=1
                        fig.tight_layout()
                        plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_rec'+str(i)))


                        #Plot the R2 and MSE loss between the original data and the predicted overtime. 
                        data_to_plot = obs_print[:,:]*args.scaling_factor #Get the first 10 samples for a test 
                        predicted_to_plot = z_p[:,:]*args.scaling_factor
                        data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                        predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                        all_r2_scores = []
                        all_mse_scores = []

                        for idx_frames in range(len(data_to_plot)):
                            _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                            all_r2_scores.append(r_value)
                            # print('data_to_plot[idx_frames,:].flatten().shape: ',data_to_plot[idx_frames,:].flatten().shape)
                            # print('predicted_to_plot[idx_frames,:].flatten().shape: ',predicted_to_plot[idx_frames,:].flatten().shape)
                            tmp_mse_loss = mean_squared_error(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                            all_mse_scores.append(tmp_mse_loss)

                        fig,ax = plt.subplots(2,1, figsize=(15,5), sharex=True, facecolor='w')
                        ax[0].plot(np.arange(len(all_r2_scores)),all_r2_scores)
                        ax[1].plot(np.arange(len(all_mse_scores)),all_mse_scores)
                        ax[1].set_xlabel("Frames")
                        ax[0].set_ylabel("R2")
                        ax[1].set_ylabel("MSE")
                        fig.tight_layout()
                        plt.savefig(os.path.join(path_to_save_plots, 'plot_performance_rec'+str(i)))

                        #Plot integral and ode part separated
                        if ode_func is not None and F_func is not None:
                            Trained_Data_ode = odeint(ode_func,torch.Tensor(obs_print[0,:]).flatten().to(args.device),times.to(args.device),rtol=1e-4,atol=1e-4)
                            Trained_Data_ode_print = to_np(Trained_Data_ode)
                            Trained_Data_integral_print  = z_p - Trained_Data_ode_print
                            # print('Trained_Data_integral_print.max():',np.abs(Trained_Data_integral_print).max())
                            # print('Trained_Data_ode_print.max():',np.abs(Trained_Data_ode_print).max())

                            data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                            predicted_to_plot_ode = Trained_Data_ode_print[-20:,:]*args.scaling_factor
                            predicted_to_plot_ide = Trained_Data_integral_print[-20:,:]*args.scaling_factor
                            data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                            predicted_to_plot_ode = args.fitted_pca.inverse_transform(predicted_to_plot_ode)
                            predicted_to_plot_ide = args.fitted_pca.inverse_transform(predicted_to_plot_ide)

                            predicted_to_plot_ode = predicted_to_plot_ode.reshape(predicted_to_plot_ode.shape[0],184, 208) # Add the original frame dimesion as input
                            predicted_to_plot_ide = predicted_to_plot_ide.reshape(predicted_to_plot_ide.shape[0],184, 208)
                            data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                            fig,ax = plt.subplots(6,10, figsize=(15,8), facecolor='w')
                            c=0
                            step = 0
                            for idx_row in range (2): 
                                for idx_col in range(10):
                                    ax[2*idx_row+step,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row+step,idx_col].axis('off')

                                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ode[c,:].flatten())
                                    ax[2*idx_row+1+step,idx_col].set_title('ODE R2: {:.3f}'.format(r_value**2))
                                    ax[2*idx_row+1+step,idx_col].imshow(predicted_to_plot_ode[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row+1+step,idx_col].axis('off')

                                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ide[c,:].flatten())
                                    ax[2*idx_row+2+step,idx_col].set_title('IDE R2: {:.3f}'.format(r_value**2))
                                    ax[2*idx_row+2+step,idx_col].imshow(predicted_to_plot_ide[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row+2+step,idx_col].axis('off')
                                    c+=1
                                step += 1
                            fig.tight_layout()
                            plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_ode_vs_ide_rec'+str(i)))

                            del data_to_plot, predicted_to_plot
                            del z_to_print, time_to_print, obs_to_print

                    del obs_test, ts_test, z_test, z_p

                    plt.close('all')

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            if split_size>0:
                save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, None, None, None)
            else: 
                save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, model, None, None, None)

            #lr_scheduler(loss_validation)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break

        if args.support_tensors is True or args.support_test is True:
                del dummy_times
                
        end = time.time()
        # print(f"Training time: {(end-start)/60:.3f} minutes")
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),Loss_print)
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),Val_Loss)
        # # plt.savefig('trained.png')
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'trained'+timestr))
        # # plt.show()
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'final_losses'+timestr))
        # # plt.show()
    elif args.mode=='evaluate':
        print('Running in evaluation mode')
        ## Validating
        model.eval()
        
        t_min , t_max = args.time_interval
        n_points = args.test_points

        
        #test_times=torch.sort(torch.rand(n_points),0)[0].to(device)*(t_max-t_min)+t_min
        test_times=torch.linspace(t_min,t_max,n_points)
        
        #dummy_times = torch.cat([torch.Tensor([0.]).to(device),dummy_times])
        # print('times :',times)
        ###########################################################
        
        with torch.no_grad():
            splitting_size = int(args.training_split*Data.size(0))
            all_r2_scores = []
            all_mse = []
            for j in tqdm(range(Data.size(0)-splitting_size)):
                Dataset_all = Test_Dynamics_Dataset(Data[j,:,:],times)
                loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np)))

                obs_test, ts_test, ids_test = next(iter(loader_test))
                ids_test, indices = torch.sort(ids_test)
                # print('indices: ',indices)
                obs_test = obs_test[indices,:]
                ts_test = ts_test[indices]
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)


                obs_test = obs_test.to(args.device)
                ts_test = ts_test.to(args.device)
                ids_test = ids_test.to(args.device)
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)
                # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                
                y0 = obs_test[0].flatten().to(device)
                c_func = lambda x: y0+f_func(torch.Tensor([x]).to(device))-f_func(torch.Tensor([0]).to(device))
                
                z_test = Integral_attention_solver(
                                    test_times.to(device),
                                    obs_test[0].unsqueeze(0).to(args.device),
                                    sampling_points = test_times.size(0),
                                    #n_support_points=args.test_points, 
                                    support_tensors=test_times,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    use_support=False,
                                    output_support_tensors=False,
                                    ).solve()
                
                # z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_test)
                obs_print = to_np(obs_test)
                
                new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                plt.figure(j,figsize=(8,8),facecolor='w')
                z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_p)

                
                plt.plot(z_p[:,0],z_p[:,1],c='r', label='model')
                #plt.scatter(z_p[:,0],z_p[:,1],s=10,c='red', label='model')
                
                obs_print = to_np(obs_test[:,:])
                # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                plt.scatter(obs_print[:,0],obs_print[:,1],label='Data',c='blue', alpha=0.5)
                plt.xlabel("dim 0")
                plt.ylabel("dim 1")
                #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                plt.legend()
                
                plt.figure(j+100, facecolor='w')
                plt.plot(to_np(test_times),z_p[:,0],c='red',label='dim0')
                plt.scatter(to_np(times),obs_print[:,0],label='Data',c='blue', alpha=0.5)
                obs_print = to_np(obs_test[:,:])
                # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                #plt.xlabel("dim 0")
                #plt.ylabel("dim 1")
                #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
#                 plt.legend()
                
                plt.figure(j+300, facecolor='w')
                plt.plot(to_np(test_times),z_p[:,1],c='red',label='dim0')
                plt.scatter(to_np(times),obs_print[:,1],label='Data',c='blue', alpha=0.5)
                obs_print = to_np(obs_test[:,:])
                # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                #plt.xlabel("dim 0")
                #plt.ylabel("dim 1")
                #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
#                 plt.legend()
                
                _, _, r_value, _, _ = scipy.stats.linregress(z_p[:,:].flatten(),obs_print[:,:].flatten())
                mse_value = mean_squared_error(z_p[:,:].flatten(),obs_print[:,:].flatten())
                
                print('R2:',r_value)
                print('MSE:',mse_value)
                
                all_r2_scores.append(r_value)
                all_mse.append(mse_value)
            
            plt.figure(-1,facecolor='w')
            plt.plot(np.linspace(0,len(all_r2_scores),len(all_r2_scores)),all_r2_scores)
            plt.xlabel("Dynamics")
            plt.ylabel("R2")
            plt.legend()
            
            plt.figure(-2,facecolor='w')
            plt.plot(np.linspace(0,len(all_mse),len(all_mse)),all_mse)
            plt.xlabel("Dynamics")
            plt.ylabel("MSE")
            plt.legend()
            
            print("Average R2:",sum(all_r2_scores)/len(all_r2_scores))
            print("Average MSE:",sum(all_mse)/len(all_mse))
                
            for j in tqdm(range(Data.size(0)-splitting_size,Data.size(0))):
                Dataset_all = Test_Dynamics_Dataset(Data[j,:,:],times)
                loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np)))

                obs_test, ts_test, ids_test = next(iter(loader_test))
                ids_test, indices = torch.sort(ids_test)
                # print('indices: ',indices)
                obs_test = obs_test[indices,:]
                ts_test = ts_test[indices]
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)


                obs_test = obs_test.to(args.device)
                ts_test = ts_test.to(args.device)
                ids_test = ids_test.to(args.device)
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)
                # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                    
                z_test = Integral_attention_solver(
                                    test_times.to(device),
                                    obs_test[0].unsqueeze(0).to(args.device),
                                    sampling_points = test_times.size(0),
                                    #n_support_points=args.test_points, 
                                    support_tensors=test_times,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    output_support_tensors=True,
                                    ).solve()

                # z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_test)
                obs_print = to_np(obs_test)
                
                new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                plt.figure(j,figsize=(8,8),facecolor='w')
                z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_p)

                
                plt.plot(z_p[:,0],z_p[:,1],c='r', label='model')
                obs_print = to_np(obs_test[:,:])
                # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                plt.scatter(obs_print[:,0],obs_print[:,1],label='Data',c='blue', alpha=0.5)
                plt.xlabel("dim 0")
                plt.ylabel("dim 1")
                #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                plt.legend()
                      
                print(scipy.stats.linregress(z_p[:,:].flatten(),obs_print[:,:].flatten()))
                print(mean_squared_error(z_p[:,:].flatten(),obs_print[:,:].flatten()))
                
                '''
                # Plot the last 20 frames  
                data_to_plot = obs_print[:,:]#*args.scaling_factor #Get the first 10 samples for a test 
                predicted_to_plot = z_p[:,:]*args.scaling_factor
                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                c=0
                for idx_row in range (2): 
                    for idx_col in range(10):
                        ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                        ax[2*idx_row,idx_col].axis('off')
                        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                        ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                        ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                        ax[2*idx_row+1,idx_col].axis('off')
                        c+=1
                fig.tight_layout()

                #Plot the R2 and MSE loss between the original data and the predicted overtime. 
                data_to_plot = obs_print[:,:]*args.scaling_factor #Get the first 10 samples for a test 
                predicted_to_plot = z_p[:,:]*args.scaling_factor
                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                all_r2_scores = []
                all_mse_scores = []

                for idx_frames in range(len(data_to_plot)):
                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                    all_r2_scores.append(r_value)
                    # print('data_to_plot[idx_frames,:].flatten().shape: ',data_to_plot[idx_frames,:].flatten().shape)
                    # print('predicted_to_plot[idx_frames,:].flatten().shape: ',predicted_to_plot[idx_frames,:].flatten().shape)
                    tmp_mse_loss = mean_squared_error(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                    all_mse_scores.append(tmp_mse_loss)

                fig,ax = plt.subplots(2,1, figsize=(15,5), sharex=True, facecolor='w')
                ax[0].plot(np.arange(len(all_r2_scores)),all_r2_scores)
                ax[1].plot(np.arange(len(all_mse_scores)),all_mse_scores)
                ax[1].set_xlabel("Frames")
                ax[0].set_ylabel("R2")
                ax[1].set_ylabel("MSE")
                fig.tight_layout()

                print('R2: ',all_r2_scores)
                print('MSE: ',all_mse_scores)
                '''

def Full_Latent_IE_experiment(rec,dec,G_NN, kernel, F_func, f_func, Data, time_seq, index_np, args, extrapolation_points): # experiment_name, plot_freq=1):
    # scaling_factor=1
    
    #metadata for saving checkpoints
    if args.model=='nie': 
        str_model_name = "nie"
    elif args.model=='node': 
        str_model_name = "node"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)

    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else: 
            num_experiments = [int(i[3:]) for i in txt]
            num_experiments = np.array(num_experiments).max()
         # -- logger location
        writer = SummaryWriter(os.path.join(path_to_experiment,'run'+str(num_experiments+1)))
        print('writer.log_dir: ',writer.log_dir)
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)
            
        with open(os.path.join(writer.log_dir,'commandline_args.txt'), 'w') as f:
            for key, value in args.__dict__.items(): 
                f.write('%s:%s\n' % (key, value))



    obs = Data
    times = time_seq
    
    loss_meter = RunningAverageMeter()

    
    if args.kernel_split is True:
        if args.kernel_type_nn is True and args.free_func_nn is True:
            All_parameters = list(F_func.parameters()) + list(kernel.parameters()) + list(f_func.parameters())
        elif args.kernel_type_nn is True:
            All_parameters = list(F_func.parameters()) + list(kernel.parameters())
    else:
        if args.free_func_nn is True:
            All_parameters = list(G_NN.parameters()) + list(f_func.parameters())
        else:
            All_parameters = list(G_NN.parameters())
    
    All_parameters = All_parameters + list(rec.parameters()) + list(dec.parameters())
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0,last_epoch=-1)# Emanuele's version
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,last_epoch=-1) #My first version
    #scheduler = LRScheduler(optimizer,patience = 20,min_lr=1e-12,factor=0.1)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0,last_epoch=-1)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        if args.free_func_nn is True:
            G_NN, optimizer, scheduler, kernel, F_func, f_func = load_checkpoint(path, G_NN, optimizer, scheduler, kernel, F_func, f_func)
        else:
            G_NN, optimizer, scheduler, kernel, F_func, f_ = load_checkpoint(path, G_NN, optimizer, scheduler, kernel, F_func, None)
            f_func = f_func


    
    if args.mode=='train':
        #lr_scheduler = LRScheduler(optimizer,patience = 50,min_lr=1e-5,factor=0.1)
        early_stopping = EarlyStopping(patience=1000,min_delta=0)

        # Loss_print = []
        # Val_Loss = []
        all_train_loss=[]
        all_val_loss=[]
        
        
        Data_splitting_indices = Train_val_split(np.copy(index_np)[1:],0.)
        Train_Data_indices = Data_splitting_indices.train_IDs()
        Val_Data_indices = Data_splitting_indices.val_IDs()
        print('\nlen(Train_Data_indices): ',len(Train_Data_indices))
        print('Train_Data_indices: ',Train_Data_indices)
        print('\nlen(Val_Data_indices): ',len(Val_Data_indices))
        print('Val_Data_indices: ',Val_Data_indices)
        
        # Train Neural IDE
        get_times = Select_times_function(times,extrapolation_points)

        save_best_model = SaveBestModel()
        start = time.time()
        
        split_size = int(args.training_split*obs.size(0))
        
        for i in range(args.epochs):
            
            
            if args.kernel_split is True:
                kernel.train()
            else: 
                G_NN.train()
            start_i = time.time()
            print('Epoch:',i)
            # GPUtil.showUtilization()
            counter=0
            train_loss = 0.0
            
            
            for j in tqdm(range(obs.size(0)-split_size)):
                
                Dataset_train = Dynamics_Dataset(obs[j,:,:],times)
                #Dataset_val = Dynamics_Dataset(obs[j-split_size,:,:],times)
                # loader = torch.utils.data.DataLoader(Dataset, batch_size = batch_size)
                # Dataset_val = Val_Dynamics_Dataset(Data,Val_Data_indices,times)

                # For the sampler
                train_sampler = SubsetRandomSampler(Train_Data_indices)
                #valid_sampler = SubsetRandomSampler(Val_Data_indices)

                # loader_val = torch.utils.data.DataLoader(Dataset, batch_size = args.batch_size)

                dataloaders = {'train': torch.utils.data.DataLoader(Dataset_train, sampler=train_sampler, batch_size = int(args.batch_size-1), drop_last=True),
                               #'val': torch.utils.data.DataLoader(Dataset_val, sampler=valid_sampler, batch_size = args.batch_size, drop_last=True),
                               #'test': torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np))),
                              }

                train_loader = dataloaders['train']
                #val_loader = dataloaders['val']
                #loader_test = dataloaders['test']

            #for obs_, ts_, ids_ in tqdm(train_loader):
                
                obs_, ts_, ids_ = next(iter(train_loader))
                obs_ = obs_.to(args.device)
                ts_ = ts_.to(args.device)
                ids_ = ids_.to(args.device)
                # obs_, ts_, ids_ = next(iter(loader))

                ids_, indices = torch.sort(ids_)
                obs_ = obs_[indices,:]
                obs_ = torch.cat([obs[j,:1,:],obs_])
                ts_ = ts_[indices]
                ts_ = torch.cat([times[:1],ts_])
                if args.perturbation_to_obs0 is not None:
                       perturb = torch.normal(mean=torch.zeros(obs_.shape[1]).to(args.device),
                                              std=args.std_noise)#args.perturbation_to_obs0*obs_[:3,:].std(dim=0))
                else:
                    perturb = torch.zeros_like(obs_[0]).to(args.device)
                # print('obs_[:5]: ',obs_[:5])
                # print('ids_[:5]: ',ids_[:5])
                # print('ts_[:5]: ',ts_[:5])

                # print('obs_: ',obs_)
                # print('ids_: ',ids_)
                # print('ts_: ',ts_)

                # obs_, ts_ = obs_.squeeze(1), ts_.squeeze(1)
                
                h = rec.initHidden().to(device)
                for t in reversed(range(obs_.size(0))):
                    obs_dyn = obs_[t, :].unsqueeze(0)
                    out, h = rec.forward(obs_dyn, h)
                qz0_mean, qz0_logvar = out[:, :args.latent_dim], out[:,args.latent_dim:]
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                
                z_ = IESolver_monoidal(
                        x = ts_.to(device),
                        dim = args.dim,
                        c = lambda x: z0.flatten().to(device)+f_func(torch.Tensor([x]).to(device))-f_func(torch.Tensor([0]).to(device)), 
                        d = lambda x,y: torch.Tensor([1]).to(device), 
                        k = kernel, 
                        f = F_func,
                        G = G_NN,
                        lower_bound = lambda x: torch.Tensor([ts_[0]]).to(device),
                        upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                        max_iterations = args.max_iterations,
                        kernel_nn=True,
                        kernel_split = args.kernel_split,
                        Last_grad_only = args.Last_grad_only,
                        G_nn = args.G_NN,
                        integration_dim = 0,
                        mc_samplings=1000,
                        num_internal_points = args.num_internal_points
                        ).solve()

                pred_x = dec(z_)
                
                #loss = F.mse_loss(z_[:,:], obs_.detach()[:,:]) #Original 
                # compute loss
                noise_std_ = torch.zeros(pred_x.size()).to(device) + args.noise_std
                noise_logvar = 2. * torch.log(noise_std_).to(device)
                logpx = log_normal_pdf(
                    obs_, pred_x, noise_logvar).sum(-1).sum(-1)
                pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
                analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                        pz0_mean, pz0_logvar).sum(-1)
                loss = torch.mean(-logpx + analytic_kl, dim=0)


                # ###############################
                # Loss_print.append(to_np(loss))
                # ###############################

                optimizer.zero_grad()
                loss.backward()#(retain_graph=True)
                optimizer.step()

                # n_iter += 1
                counter += 1
                train_loss += loss.item()
                if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                    scheduler.step()
                    
                del obs_, ts_, z_

            train_loss /= counter
            all_train_loss.append(train_loss)
            if  split_size==0 and args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
            
            del loss, train_loss


            ## Validating
            if args.kernel_split is True:
                kernel.eval()
            else: 
                G_NN.eval()
            with torch.no_grad():

                    #Only do this if there is a validation dataset
                
                val_loss = 0.0
                counter = 0
                if split_size>0:
                    # for images, _, _, _, _ in tqdm(val_loader):   # frames, timevals, angular_velocity, mass_height, mass_xpos
                    for j in tqdm(range(obs.size(0)-split_size,obs.size(0))):
                        
                        valid_sampler = SubsetRandomSampler(Train_Data_indices)
                        Dataset_val = Dynamics_Dataset(obs[j,:,:],times)
                        val_loader = torch.utils.data.DataLoader(Dataset_val, sampler=valid_sampler, batch_size = int(args.batch_size-1), drop_last=True)
                    
                    #for obs_val, ts_val, ids_val in tqdm(val_loader):
                        obs_val, ts_val, ids_val = next(iter(val_loader))
                        obs_val = obs_val.to(args.device)
                        ts_val = ts_val.to(args.device)
                        ids_val = ids_val.to(args.device)

                        ids_val, indices = torch.sort(ids_val)
                        # print('indices: ',indices)
                        obs_val = obs_val[indices,:]
                        ts_val = ts_val[indices]
                        
                        obs_val = torch.cat([obs[j,:1,:],obs_val])
                        ts_val = torch.cat([times[:1],ts_val])

                        #Concatenate the first point of the train minibatch
                        # obs_[0],ts_
                        # print('\n In validation mode...')
                        # print('obs_[:5]: ',obs_[:5])
                        # print('ids_[:5]: ',ids_[:5])
                        # print('ts_[:5]: ',ts_[:5])
                        # print('ts_[0]:',ts_[0])

                        ## Below is to add initial data point to val
                        #obs_val = torch.cat((obs_[0][None,:],obs_val))
                        #ts_val = torch.hstack((ts_[0],ts_val))
                        #ids_val = torch.hstack((ids_[0],ids_val))

                        # obs_val, ts_val, ids_val = next(iter(loader_val))
                        # print('obs_val.shape: ',obs_val.shape)
                        # print('ids_val: ',ids_val)
                        # print('ts_val: ',ts_val)

                        # obs_val, ts_val = obs_val.squeeze(1), ts_val.squeeze(1)
                        
                        h = rec.initHidden().to(device)
                        for t in reversed(range(obs_val.size(0))):
                            obs_dyn_val = obs_val[t, :].unsqueeze(0)
                            out, h = rec.forward(obs_dyn_val, h)
                        qz0_mean, qz0_logvar = out[:, :args.latent_dim], out[:,args.latent_dim:]
                        epsilon = torch.randn(qz0_mean.size()).to(device)
                        z0_val = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                        z_val = IESolver_monoidal(
                                    x = ts_val.to(device),
                                    dim = args.dim,
                                    c = lambda x: z0_val.flatten().to(device)+f_func(torch.Tensor([x]).to(device))-f_func(torch.Tensor([0]).to(device)), 
                                    d = lambda x,y: torch.Tensor([1]).to(device), 
                                    k = kernel, 
                                    f = F_func,
                                    G = G_NN,
                                    lower_bound = lambda x: torch.Tensor([ts_val[0]]).to(device),
                                    upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                                    max_iterations = args.max_iterations,
                                    kernel_nn=True,
                                    kernel_split = args.kernel_split,
                                    Last_grad_only = args.Last_grad_only,
                                    G_nn = args.G_NN,
                                    integration_dim = 0,
                                    mc_samplings=1000,
                                    num_internal_points = args.num_internal_points
                                    ).solve()

                        
                        #loss_validation = F.mse_loss(z_val[:,:], obs_val.detach()[:,:])
                        pred_x = dec(z_val)
                
                        #loss = F.mse_loss(z_[:,:], obs_.detach()[:,:]) #Original 
                        # compute loss
                        noise_std_ = torch.zeros(pred_x.size()).to(device) + args.noise_std
                        noise_logvar = 2. * torch.log(noise_std_).to(device)
                        logpx = log_normal_pdf(
                            obs_val, pred_x, noise_logvar).sum(-1).sum(-1)
                        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
                        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                                pz0_mean, pz0_logvar).sum(-1)
                        loss_validation = torch.mean(-logpx + analytic_kl, dim=0)
                        
                        counter += 1
                        val_loss += loss_validation.item()
                        #LRScheduler(loss_validation)
                        if args.lr_scheduler == 'ReduceLROnPlateau':
                            scheduler.step(val_loss)
                            
                        del obs_val, ts_val, z_val

            
                else: counter += 1

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            writer.add_scalar('train_loss', all_train_loss[-1], global_step=i)
            if len(all_val_loss)>0:
                writer.add_scalar('val_loss', all_val_loss[-1], global_step=i)
            if args.lr_scheduler == 'ReduceLROnPlateau':
                writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], global_step=i)
            elif args.lr_scheduler == 'CosineAnnealingLR':
                writer.add_scalar('Epoch/learning_rate', scheduler.get_last_lr()[0], global_step=i)

            
            with torch.no_grad():
                if i % args.plot_freq == 0 and args.test is True:
                    if obs.size(2)>2:
                        pca_proj = PCA(n_components=2)
                    for j in tqdm(range(obs.size(0))):
                        Dataset_all = Test_Dynamics_Dataset(Data[j,:,:],times)
                        loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np)))

                        obs_test, ts_test, ids_test = next(iter(loader_test))

                        ids_test, indices = torch.sort(ids_test)
                        # print('indices: ',indices)
                        obs_test = obs_test[indices,:]
                        ts_test = ts_test[indices]
                        # print('obs_test.shape: ',obs_test.shape)
                        # print('ids_test: ',ids_test)
                        # print('ts_test: ',ts_test)


                        obs_test = obs_test.to(args.device)
                        ts_test = ts_test.to(args.device)
                        ids_test = ids_test.to(args.device)
                        # print('obs_test.shape: ',obs_test.shape)
                        # print('ids_test: ',ids_test)
                        # print('ts_test: ',ts_test)
                        # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                        
                        h = rec.initHidden().to(device)
                        for t in reversed(range(obs_test.size(0))):
                            obs_dyn_test = obs_test[t, :].unsqueeze(0)
                            out, h = rec.forward(obs_dyn_test, h)
                        qz0_mean, qz0_logvar = out[:, :args.latent_dim], out[:,args.latent_dim:]
                        epsilon = torch.randn(qz0_mean.size()).to(device)
                        z0_test = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                        
                        z_test = IESolver_monoidal(
                                        x = ts_test.to(device), 
                                        dim = args.dim,
                                        c = lambda x: z0_test.flatten().to(device)+f_func(torch.Tensor([x]).to(device))-f_func(torch.Tensor([0]).to(device)), 
                                        d = lambda x,y: torch.Tensor([1]).to(device), 
                                        k = kernel, 
                                        f = F_func,
                                        G = G_NN,
                                        lower_bound = lambda x: torch.Tensor([ts_test[0]]).to(device),
                                        upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                                        max_iterations = args.max_iterations,
                                        kernel_nn=True,
                                        kernel_split = args.kernel_split,
                                        Last_grad_only = args.Last_grad_only,
                                        G_nn = args.G_NN,
                                        integration_dim = 0,
                                        mc_samplings=1000,
                                        num_internal_points = args.num_internal_points
                                        ).solve()
                        
                        z_test = dec(z_test)
                
                        #loss = F.mse_loss(z_[:,:], obs_.detach()[:,:]) #Original 
                        # compute loss
                        
                        #print('Parameters are:',ide_trained.parameters)
                        #print(list(All_parameters))
                        plt.figure(0, figsize=(8,8),facecolor='w')
                        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
                        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
                        plt.plot(np.log10(all_train_loss),label='Train loss')
                        if split_size>0:
                            plt.plot(np.log10(all_val_loss),label='Val loss')
                        plt.xlabel("Epoch")
                        plt.ylabel("MSE Loss")
                        # timestr = time.strftime("%Y%m%d-%H%M%S")
                        plt.savefig(os.path.join(path_to_save_plots,'losses'))

                        new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                        plt.figure(figsize=(8,8),facecolor='w')
                        z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                        z_p = to_np(z_p)
                        obs_print = to_np(obs_test[:,:])

                        if obs.size(2)>2:
                            z_p = pca_proj.fit_transform(z_p)
                            obs_print = pca_proj.fit_transform(obs_print)                    

                        plt.figure(1, facecolor='w')
                        plt.plot(z_p[:,0],z_p[:,1],c='r', label='model')

                        # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                        plt.scatter(obs_print[:,0],obs_print[:,1],label='Data',c='blue', alpha=0.5)
                        plt.xlabel("dim 0")
                        plt.ylabel("dim 1")
                        #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                        plt.legend()
                        # plt.show()
                        # timestr = time.strftime("%Y%m%d-%H%M%S")
                        plt.savefig(os.path.join(path_to_save_plots,'plot_dim0vsdim1_epoch'+str(i)+'_'+str(j)))


                        if 'calcium_imaging' in args.experiment_name:
                            # Plot the first 20 frames
                            data_to_plot = obs_print[:20,:]*args.scaling_factor #Get the first 10 samples for a test 
                            predicted_to_plot = z_p[:20,:]*args.scaling_factor
                            data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                            predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                            predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                            data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                            fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                            c=0
                            for idx_row in range (2): 
                                for idx_col in range(10):
                                    ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row,idx_col].axis('off')
                                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                    ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                    ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row+1,idx_col].axis('off')
                                    c+=1
                            fig.tight_layout()
                            plt.savefig(os.path.join(path_to_save_plots, 'plot_first20frame_rec'+str(i)))


                            # Plot the last 20 frames  
                            data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                            predicted_to_plot = z_p[-20:,:]*args.scaling_factor
                            data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                            predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                            predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                            data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                            fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                            c=0
                            for idx_row in range (2): 
                                for idx_col in range(10):
                                    ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row,idx_col].axis('off')
                                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                    ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                    ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row+1,idx_col].axis('off')
                                    c+=1
                            fig.tight_layout()
                            plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_rec'+str(i)))


                            #Plot the R2 and MSE loss between the original data and the predicted overtime. 
                            data_to_plot = obs_print[:,:]*args.scaling_factor #Get the first 10 samples for a test 
                            predicted_to_plot = z_p[:,:]*args.scaling_factor
                            data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                            predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                            all_r2_scores = []
                            all_mse_scores = []

                            for idx_frames in range(len(data_to_plot)):
                                _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                                all_r2_scores.append(r_value)
                                # print('data_to_plot[idx_frames,:].flatten().shape: ',data_to_plot[idx_frames,:].flatten().shape)
                                # print('predicted_to_plot[idx_frames,:].flatten().shape: ',predicted_to_plot[idx_frames,:].flatten().shape)
                                tmp_mse_loss = mean_squared_error(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                                all_mse_scores.append(tmp_mse_loss)

                            fig,ax = plt.subplots(2,1, figsize=(15,5), sharex=True, facecolor='w')
                            ax[0].plot(np.arange(len(all_r2_scores)),all_r2_scores)
                            ax[1].plot(np.arange(len(all_mse_scores)),all_mse_scores)
                            ax[1].set_xlabel("Frames")
                            ax[0].set_ylabel("R2")
                            ax[1].set_ylabel("MSE")
                            fig.tight_layout()
                            plt.savefig(os.path.join(path_to_save_plots, 'plot_performance_rec'+str(i)))

                            #Plot integral and ode part separated
                            if ode_func is not None and F_func is not None:
                                Trained_Data_ode = odeint(ode_func,torch.Tensor(obs_print[0,:]).flatten().to(args.device),times.to(args.device),rtol=1e-4,atol=1e-4)
                                Trained_Data_ode_print = to_np(Trained_Data_ode)
                                Trained_Data_integral_print  = z_p - Trained_Data_ode_print
                                # print('Trained_Data_integral_print.max():',np.abs(Trained_Data_integral_print).max())
                                # print('Trained_Data_ode_print.max():',np.abs(Trained_Data_ode_print).max())

                                data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                                predicted_to_plot_ode = Trained_Data_ode_print[-20:,:]*args.scaling_factor
                                predicted_to_plot_ide = Trained_Data_integral_print[-20:,:]*args.scaling_factor
                                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                                predicted_to_plot_ode = args.fitted_pca.inverse_transform(predicted_to_plot_ode)
                                predicted_to_plot_ide = args.fitted_pca.inverse_transform(predicted_to_plot_ide)

                                predicted_to_plot_ode = predicted_to_plot_ode.reshape(predicted_to_plot_ode.shape[0],184, 208) # Add the original frame dimesion as input
                                predicted_to_plot_ide = predicted_to_plot_ide.reshape(predicted_to_plot_ide.shape[0],184, 208)
                                data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                                fig,ax = plt.subplots(6,10, figsize=(15,8), facecolor='w')
                                c=0
                                step = 0
                                for idx_row in range (2): 
                                    for idx_col in range(10):
                                        ax[2*idx_row+step,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row+step,idx_col].axis('off')

                                        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ode[c,:].flatten())
                                        ax[2*idx_row+1+step,idx_col].set_title('ODE R2: {:.3f}'.format(r_value**2))
                                        ax[2*idx_row+1+step,idx_col].imshow(predicted_to_plot_ode[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row+1+step,idx_col].axis('off')

                                        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ide[c,:].flatten())
                                        ax[2*idx_row+2+step,idx_col].set_title('IDE R2: {:.3f}'.format(r_value**2))
                                        ax[2*idx_row+2+step,idx_col].imshow(predicted_to_plot_ide[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row+2+step,idx_col].axis('off')
                                        c+=1
                                    step += 1
                                fig.tight_layout()
                                plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_ode_vs_ide_rec'+str(i)))




                        if F_func is not None and args.print_F_func is True:
                            # fig,ax = plt.subplots(1,1, figsize=(8,8), facecolor='w')
                            F_out = to_np(F_func.forward(z_test,ts_test))
                            # print('F_out.shape: ',F_out.shape)
                            # if F_out.shape[1]>2: 
                            #     # reducer = umap.UMAP(n_components=2, random_state=1) 
                            #     reducer = PCA(n_components=2)
                            #     F_out = reducer.fit_transform(F_out)
                            n_plots_x = int(np.ceil(np.sqrt(F_out.shape[1])))
                            n_plots_y = int(np.floor(np.sqrt(F_out.shape[1])))
                            fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=(10, 10), sharex=True, dpi=100, facecolor='w', edgecolor='k')
                            ax=ax.ravel()
                            for idx in range(args.num_dim_plot):
                                # plt.scatter(to_np(times)[:extrapolation_points],obs_print[:extrapolation_points,0]*scaling_factor,label='Data',c='blue')
                                ax[idx].scatter(to_np(times)[:],F_out[:,idx],label='F_out',c='blue', alpha=0.5)
                                ax[idx].set_xlabel("Time")
                                ax[idx].set_ylabel("F"+str(idx))
                                #plt.scatter(to_np(times)[extrapolation_points:],obs_print[extrapolation_points:,0,0],label='Data extr',c='red')
                                ax[idx].legend()
                                # timestr = time.strftime("%Y%m%d-%H%M%S")
                            fig.tight_layout()

                            # ax.plot(F_out[:,0],F_out[:,1])
                            # ax.scatter(F_out[:,0],F_out[:,1], c = np.arange(len(F_out)))
                            # ax.set_xlabel("F_0")
                            # ax.set_ylabel("F_1")
                            plt.savefig(os.path.join(path_to_save_plots, 'plot_F_func'+str(i)))


                        plt.close('all')
                        
                        del obs_test, ts_test, z_test

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            if args.kernel_split is True:
                model_state = {
                        'epoch': i + 1,
                        'state_dict': kernel.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }
            else:
                model_state = {
                        'epoch': i + 1,
                        'state_dict': G_NN.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            if split_size>0:
                if args.free_func_nn is False:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, G_NN, kernel, F_func, None)
                else:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, G_NN, kernel, F_func, f_func)
            else: 
                if args.free_func_nn is False:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, G_NN, kernel, F_func, None)
                else:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, G_NN, kernel, F_func, f_func)

            #lr_scheduler(loss_validation)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break


        end = time.time()
        # print(f"Training time: {(end-start)/60:.3f} minutes")
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),Loss_print)
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),Val_Loss)
        # # plt.savefig('trained.png')
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'trained'+timestr))
        # # plt.show()
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'final_losses'+timestr))
        # # plt.show()
    elif args.mode=='evaluate':
        print('Running in evaluation mode')
        ## Validating
        model.eval()
        with torch.no_grad():
            splitting_size = int(args.training_split*Data.size(0))
            all_r2_scores = []
            all_mse = []
            for j in tqdm(range(Data.size(0)-splitting_size)):
                Dataset_all = Test_Dynamics_Dataset(Data[j,:,:],times)
                loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np)))

                obs_test, ts_test, ids_test = next(iter(loader_test))
                ids_test, indices = torch.sort(ids_test)
                # print('indices: ',indices)
                obs_test = obs_test[indices,:]
                ts_test = ts_test[indices]
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)


                obs_test = obs_test.to(args.device)
                ts_test = ts_test.to(args.device)
                ids_test = ids_test.to(args.device)
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)
                # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                h = rec.initHidden().to(device)
                for t in reversed(range(obs_test.size(0))):
                    obs_dyn_test = obs_test[t, :].unsqueeze(0)
                    out, h = rec.forward(obs_dyn_test, h)
                qz0_mean, qz0_logvar = out[:, :args.latent_dim], out[:,args.latent_dim:]
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0_val = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                        
                z_test = IESolver_monoidal(
                                x = ts_test.to(device), 
                                dim = args.dim,
                                c = lambda x: obs_test[0].flatten().to(device)+f_func(torch.Tensor([x]).to(device))-f_func(torch.Tensor([0]).to(device)), 
                                d = lambda x,y: torch.Tensor([1]).to(device), 
                                k = kernel, 
                                f = F_func,
                                G = G_NN,
                                lower_bound = lambda x: torch.Tensor([ts_test[0]]).to(device),
                                upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                                max_iterations = args.max_iterations,
                                kernel_nn=True,
                                kernel_split = args.kernel_split,
                                Last_grad_only = args.Last_grad_only,
                                G_nn = args.G_NN,
                                integration_dim = 0,
                                mc_samplings=1000,
                                num_internal_points = args.num_internal_points
                                ).solve()

                z_test = dec(z_test)
                # z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_test)
                obs_print = to_np(obs_test)
                
                new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                plt.figure(j,figsize=(8,8),facecolor='w')
                z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_p)

                plt.figure(1, facecolor='w')
                plt.plot(z_p[:,0],z_p[:,1],c='r', label='model')
                obs_print = to_np(obs_test[:,:])
                # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                plt.scatter(obs_print[:,0],obs_print[:,1],label='Data',c='blue', alpha=0.5)
                plt.xlabel("dim 0")
                plt.ylabel("dim 1")
                #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                plt.legend()
                
                _, _, r_value, _, _ = scipy.stats.linregress(z_p[:,:].flatten(),obs_print[:,:].flatten())
                mse_value = mean_squared_error(z_p[:,:].flatten(),obs_print[:,:].flatten())
                
                print('R2:',r_value)
                print('MSE:',mse_value)
                
                all_r2_scores.append(r_value)
                all_mse.append(mse_value)
            
            plt.figure(-2,facecolor='w')
            plt.plot(np.linspace(0,len(all_r2_scores),len(all_r2_scores)),all_r2_scores)
            plt.xlabel("Dynamics")
            plt.ylabel("R2")
            plt.legend()
            
            plt.figure(-1,facecolor='w')
            plt.plot(np.linspace(0,len(all_mse),len(all_mse)),all_mse)
            plt.xlabel("Dynamics")
            plt.ylabel("MSE")
            plt.legend()
            
            print("Average R2:",sum(all_r2_scores)/len(all_r2_scores))
            print("Average MSE:",sum(all_mse)/len(all_mse))
                
            for j in tqdm(range(Data.size(0)-splitting_size,Data.size(0))):
                Dataset_all = Test_Dynamics_Dataset(Data[j,:,:],times)
                loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np)))

                obs_test, ts_test, ids_test = next(iter(loader_test))
                ids_test, indices = torch.sort(ids_test)
                # print('indices: ',indices)
                obs_test = obs_test[indices,:]
                ts_test = ts_test[indices]
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)


                obs_test = obs_test.to(args.device)
                ts_test = ts_test.to(args.device)
                ids_test = ids_test.to(args.device)
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)
                # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                h = rec.initHidden().to(device)
                for t in reversed(range(obs_test.size(0))):
                    obs_dyn_test = obs_test[t, :].unsqueeze(0)
                    out, h = rec.forward(obs_dyn_test, h)
                qz0_mean, qz0_logvar = out[:, :args.latent_dim], out[:,args.latent_dim:]
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0_val = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                        
                z_test = IESolver_monoidal(
                                x = ts_test.to(device), 
                                dim = args.dim,
                                c = lambda x: obs_test[0].flatten().to(device)+f_func(torch.Tensor([x]).to(device))-f_func(torch.Tensor([0]).to(device)), 
                                d = lambda x,y: torch.Tensor([1]).to(device), 
                                k = kernel, 
                                f = F_func,
                                G = G_NN,
                                lower_bound = lambda x: torch.Tensor([ts_test[0]]).to(device),
                                upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                                max_iterations = args.max_iterations,
                                kernel_nn=True,
                                kernel_split = args.kernel_split,
                                Last_grad_only = args.Last_grad_only,
                                G_nn = args.G_NN,
                                integration_dim = 0,
                                mc_samplings=1000,
                                num_internal_points = args.num_internal_points
                                ).solve()

                z_test = dec(z_test)
                
                z_p = to_np(z_test)
                obs_print = to_np(obs_test)
                
                new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                plt.figure(j,figsize=(8,8),facecolor='w')
                z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_p)

                
                plt.plot(z_p[:,0],z_p[:,1],c='r', label='model')
                obs_print = to_np(obs_test[:,:])
                # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                plt.scatter(obs_print[:,0],obs_print[:,1],label='Data',c='blue', alpha=0.5)
                plt.xlabel("dim 0")
                plt.ylabel("dim 1")
                #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                plt.legend()
                      
                print(scipy.stats.linregress(z_p[:,:].flatten(),obs_print[:,:].flatten()))
                print(mean_squared_error(z_p[:,:].flatten(),obs_print[:,:].flatten()))
                
                '''
                # Plot the last 20 frames  
                data_to_plot = obs_print[:,:]#*args.scaling_factor #Get the first 10 samples for a test 
                predicted_to_plot = z_p[:,:]*args.scaling_factor
                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                c=0
                for idx_row in range (2): 
                    for idx_col in range(10):
                        ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                        ax[2*idx_row,idx_col].axis('off')
                        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                        ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                        ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                        ax[2*idx_row+1,idx_col].axis('off')
                        c+=1
                fig.tight_layout()

                #Plot the R2 and MSE loss between the original data and the predicted overtime. 
                data_to_plot = obs_print[:,:]*args.scaling_factor #Get the first 10 samples for a test 
                predicted_to_plot = z_p[:,:]*args.scaling_factor
                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                all_r2_scores = []
                all_mse_scores = []

                for idx_frames in range(len(data_to_plot)):
                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                    all_r2_scores.append(r_value)
                    # print('data_to_plot[idx_frames,:].flatten().shape: ',data_to_plot[idx_frames,:].flatten().shape)
                    # print('predicted_to_plot[idx_frames,:].flatten().shape: ',predicted_to_plot[idx_frames,:].flatten().shape)
                    tmp_mse_loss = mean_squared_error(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                    all_mse_scores.append(tmp_mse_loss)

                fig,ax = plt.subplots(2,1, figsize=(15,5), sharex=True, facecolor='w')
                ax[0].plot(np.arange(len(all_r2_scores)),all_r2_scores)
                ax[1].plot(np.arange(len(all_mse_scores)),all_mse_scores)
                ax[1].set_xlabel("Frames")
                ax[0].set_ylabel("R2")
                ax[1].set_ylabel("MSE")
                fig.tight_layout()

                print('R2: ',all_r2_scores)
                print('MSE: ',all_mse_scores)
                '''

                
                

def Classification_experiment(G_NN, kernel, F_func, f_func, Data, labels, Data_test, labels_test, args, Encoder = None, Decoder = None): # experiment_name, plot_freq=1):
    # scaling_factor=1
        
    # -- metadata for saving checkpoints
     
    str_model_name = "nie"
    
    # str_model_type = "linear_encoder" # options: linear_encoder; conv_encoder (not implemented yet)
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)

    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        # Check how many runs are already there
        # num_experiments = len(os.listdir(os.path.join(str_log_dir,str_model_name)))
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else: 
            num_experiments = [int(i[3:]) for i in txt]
            num_experiments = np.array(num_experiments).max()
         # -- logger location
        writer = SummaryWriter(os.path.join(path_to_experiment,'run'+str(num_experiments+1)))
        print('writer.log_dir: ',writer.log_dir)
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)
            
        with open(os.path.join(writer.log_dir,'commandline_args.txt'), 'w') as f:
            for key, value in args.__dict__.items(): 
                f.write('%s:%s\n' % (key, value))


        # logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)

    obs = Data
    target = labels
    obs_test=Data_test
    target_test=labels_test
    pca_proj = PCA(n_components=2)
    #times = time_seq
    

    
    if args.kernel_split is True:
        if args.kernel_type_nn is True and args.free_func_nn is True:
            All_parameters = list(F_func.parameters()) + list(kernel.parameters()) + list(f_func.parameters())
        elif args.kernel_type_nn is True:
            All_parameters = list(F_func.parameters()) + list(kernel.parameters())
    else:
        if args.free_func_nn is True:
            All_parameters = list(G_NN.parameters()) + list(f_func.parameters())
        else:
            All_parameters = list(G_NN.parameters())
            
    if Encoder is not None:
        All_parameters = All_parameters + list(Encoder.parameters())
        
    if Decoder is not None:
        All_parameters = All_parameters + list(Decoder.parameters())
    
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0,last_epoch=-1)# Emanuele's version
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,last_epoch=-1) #My first version
    #scheduler = LRScheduler(optimizer,patience = 20,min_lr=1e-12,factor=0.1)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0,last_epoch=-1)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        if args.free_func_nn is True:
            G_NN, optimizer, scheduler, kernel, F_func, f_func = load_checkpoint(path, G_NN, optimizer, scheduler, kernel, F_func, f_func)
        else:
            G_NN, optimizer, scheduler, kernel, F_func, f_ = load_checkpoint(path, G_NN, optimizer, scheduler, kernel, F_func, None)
            f_func = f_func


    
    if args.mode=='train':
        #lr_scheduler = LRScheduler(optimizer,patience = 50,min_lr=1e-5,factor=0.1)
        early_stopping = EarlyStopping(patience=1000,min_delta=0)

        # Loss_print = []
        # Val_Loss = []
        all_train_loss=[]
        all_val_loss=[]
        test_losses=[]
        accuracy=[]
        
        #Data_splitting_indices = Train_val_split(np.copy(index_np)[1:],0.)
        #Train_Data_indices = Data_splitting_indices.train_IDs()
        #Val_Data_indices = Data_splitting_indices.val_IDs()
        #print('\nlen(Train_Data_indices): ',len(Train_Data_indices))
        #print('Train_Data_indices: ',Train_Data_indices)
        #print('\nlen(Val_Data_indices): ',len(Val_Data_indices))
        #print('Val_Data_indices: ',Val_Data_indices)
        
        # Train Neural IDE
        #get_times = Select_times_function(times,extrapolation_points)

        save_best_model = SaveBestModel()
        start = time.time()
        
        split_size = int(args.training_split*obs.size(0))
        
        criterion = torch.nn.CrossEntropyLoss()
        soft_max = torch.nn.Softmax(dim=1)
        
        for i in range(args.epochs):
            
            
            if args.kernel_split is True:
                kernel.train()
            else: 
                G_NN.train()
            start_i = time.time()
            print('Epoch:',i)
            # GPUtil.showUtilization()
            counter=0
            train_loss = 0.0
            
        
            for j in tqdm(range(obs.size(0)-split_size)):   
                obs_, target_= obs[j,:], torch.tensor([target[j]]).to(device)#next(iter(train_loader))
                obs_ = obs_.to(args.device)
                


                # obs_, ts_ = obs_.squeeze(1), ts_.squeeze(1)
                if Encoder is not None:
                    obs_ = Encoder(obs_)
                
                c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,obs_.size(0)).to(device), obs_)
                interpolation = NaturalCubicSpline(c_coeffs)
                c = lambda x: interpolation.evaluate(x).squeeze()
                
                z_ = IESolver_monoidal(
                                    torch.linspace(0,1,2).to(device),
                                    #y_0 = obs_[0].flatten().to(device), 
                                    #c = lambda x: obs_[0].flatten().to(device),
                                    c = c,
                                    d = lambda x,y: torch.Tensor([1]).to(device), 
                                    k = kernel, 
                                    f = F_func,
                                    G = G_NN,
                                    lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                                    max_iterations = args.max_iterations,
                                    kernel_nn=True,
                                    kernel_split = args.kernel_split,
                                    G_nn = args.G_NN,
                                    integration_dim = 0,
                                    mc_samplings=1000,
                                    num_internal_points = args.num_internal_points,
                                    Last_grad_only=False,
                                    ).solve()
                
                z_ = z_[-1:,:]
                if Decoder is not None:
                    z_ = Decoder(z_)
                
                z_ = soft_max(z_)
                
                
                #loss_ts_ = get_times.select_times(ts_)[1]
                loss = criterion(z_, target_.detach()) #Original 
                # print('z_[:,:].to(args.device): ',z_[:,:].to(args.device))
                # print('obs_.to(args.device).detach()[:,:]: ',obs_.to(args.device).detach()[:,:])
                # loss = F.mse_loss(z_[:,:].to(args.device), obs_.to(args.device).detach()[:,:]) #Original 


                # ###############################
                # Loss_print.append(to_np(loss))
                # ###############################

                optimizer.zero_grad()
                loss.backward()#(retain_graph=True)
                optimizer.step()

                # n_iter += 1
                counter += 1
                train_loss += loss.item()
                if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                        scheduler.step()
                

            train_loss /= counter
            all_train_loss.append(train_loss)
            if  split_size > 0 and args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)


            ## Validating
            if args.kernel_split is True:
                kernel.eval()
            else: 
                G_NN.eval()
            with torch.no_grad():

                    #Only do this if there is a validation dataset
                
                val_loss = 0.0
                counter = 0
                Data_val = obs[-split_size:,]
                if Data_val is not None:
                    # for images, _, _, _, _ in tqdm(val_loader):   # frames, timevals, angular_velocity, mass_height, mass_xpos
                    
                    
                    for j in tqdm(range(obs.size(0)-split_size,obs.size(0))):
                        
                        obs_val, target_val= obs[j,:,:], torch.tensor([labels[j]]).to(device)#next(iter(train_loader))
                        obs_val = obs_val.to(args.device)
                        target_val = target_val.to(device)
        
                        
                        if Encoder is not None:
                            obs_val = Encoder(obs_val)
                        
                        c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,obs_val.size(0)).to(device), obs_val)
                        interpolation = NaturalCubicSpline(c_coeffs)
                        c = lambda x: interpolation.evaluate(x).squeeze()

                        z_val = IESolver_monoidal(
                                        torch.linspace(0,1,2).to(device),
                                        #y_0 = obs_[0].flatten().to(device), 
                                        #c = lambda x: obs_val[0].flatten().to(device), 
                                        c = c,
                                        d = lambda x,y: torch.Tensor([1]).to(device), 
                                        k = kernel, 
                                        f = F_func,
                                        G = G_NN,
                                        lower_bound = lambda x: torch.Tensor([0]).to(device),
                                        upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                                        max_iterations = args.max_iterations,
                                        kernel_nn=True,
                                        kernel_split = args.kernel_split,
                                        G_nn = args.G_NN,
                                        integration_dim = 0,
                                        mc_samplings=1000,
                                        num_internal_points = args.num_internal_points
                                        ).solve()

                        z_val = z_val[-1:,:]
                        if Decoder is not None:
                            z_val = Decoder(z_val)

                        z_val = soft_max(z_val)

                        #validation_ts_ = get_times.select_times(ts_val)[1]
                        loss_validation = criterion(z_val, target_val.detach())
                        # Val_Loss.append(to_np(loss_validation))

                        #LRScheduler(loss_validation)
                        if args.lr_scheduler == 'ReduceLROnPlateau':
                            scheduler.step(val_loss)

                        counter += 1
                        val_loss += loss_validation.item()
                else: counter += 1

                val_loss /= counter
                all_val_loss.append(val_loss)

            writer.add_scalar('train_loss', all_train_loss[-1], global_step=i)
            if len(all_val_loss)>0:
                writer.add_scalar('val_loss', all_val_loss[-1], global_step=i)
            if args.lr_scheduler == 'ReduceLROnPlateau':
                writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], global_step=i)
            elif args.lr_scheduler == 'CosineAnnealingLR':
                writer.add_scalar('Epoch/learning_rate', scheduler.get_last_lr()[0], global_step=i)


            if args.plot_freq is not None and i % args.plot_freq == 0 and Data_test is not None:
                
                test_loss = 0
                correct = 0
                
                
                
                for j in tqdm(range(Data_test.size(0))):
                    #Dataset_all = Test_Dynamics_Dataset(obs[j,:],times)
                    #loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np)))

                    

                    #ids_test, indices = torch.sort(ids_test)
                    # print('indices: ',indices)
                    #obs_test = obs_test[:]
                    #ts_test = ts_test[indices]
                    # print('obs_test.shape: ',obs_test.shape)
                    # print('ids_test: ',ids_test)
                    # print('ts_test: ',ts_test)


                    obs_test, target_test= Data_test[j,:,:], torch.tensor([labels_test[j]]).to(device)#next(iter(train_loader))
                    obs_test = obs_test.to(args.device)
                    target_test = target_test.to(device)
                    
                    # print('obs_test.shape: ',obs_test.shape)
                    # print('ids_test: ',ids_test)
                    # print('ts_test: ',ts_test)
                    # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                    if Encoder is not None:
                        obs_test = Encoder(obs_test)
                        
                    c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,obs_test.size(0)).to(device), obs_test)
                    interpolation = NaturalCubicSpline(c_coeffs)
                    c = lambda x: interpolation.evaluate(x).squeeze()
                
                    z_test = IESolver_monoidal(
                                    torch.linspace(0,1,2).to(device),
                                    #y_0 = obs_[0].flatten().to(device), 
                                    #c = lambda x: obs_test[0].flatten().to(device), 
                                    c = c,
                                    d = lambda x,y: torch.Tensor([1]).to(device), 
                                    k = kernel, 
                                    f = F_func,
                                    G = G_NN,
                                    lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                                    max_iterations = args.max_iterations,
                                    kernel_nn=True,
                                    kernel_split = args.kernel_split,
                                    G_nn = args.G_NN,
                                    integration_dim = 0,
                                    mc_samplings=1000,
                                    num_internal_points = args.num_internal_points
                                    ).solve()
                    
                    z_test = z_test[-1:,:]
                    if Decoder is not None:
                        z_test = Decoder(z_test)
                    z_test = soft_max(z_test)
                    
                    test_loss += criterion(z_test, target_test).item()
                    pred = z_test.max(1, keepdim=True)[1]
                    correct += pred.eq(target_test.view_as(pred)).sum()
                    #print('Parameters are:',ide_trained.parameters)
                    #print(list(All_parameters))
                    plt.figure(0, facecolor='w')
                    # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
                    # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
                    plt.plot(np.log10(all_train_loss),label='Train loss')
                    if split_size>0:
                        plt.plot(np.log10(all_val_loss),label='Val loss')
                    plt.xlabel("Epoch")
                    plt.ylabel("MSE Loss")
                    # timestr = time.strftime("%Y%m%d-%H%M%S")
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))
                    '''
                    new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                    plt.figure(figsize=(8,8),facecolor='w')
                    z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                    z_p = to_np(z_p)
                    obs_print = to_np(obs_test[:,:])
                    
                    z_p = pca_proj.fit_transform(z_p)
                    obs_print = pca_proj.fit_transform(obs_print)                    

                    plt.figure(1, facecolor='w')
                    plt.plot(z_p[:,0],z_p[:,1],c='r', label='model')
                    
                    # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                    plt.scatter(obs_print[:,0],obs_print[:,1],label='Data',c='blue', alpha=0.5)
                    plt.xlabel("dim 0")
                    plt.ylabel("dim 1")
                    #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                    plt.legend()
                    # plt.show()
                    # timestr = time.strftime("%Y%m%d-%H%M%S")
                    plt.savefig(os.path.join(path_to_save_plots,'plot_dim0vsdim1_epoch'+str(i)+'_'+str(j)))

                    #Plot the other dimensions vs time
                    if obs_print.shape[1]<args.num_dim_plot: args.num_dim_plot=obs_print.shape[1]
                    n_plots_x = int(np.ceil(np.sqrt(args.num_dim_plot)))
                    n_plots_y = int(np.floor(np.sqrt(args.num_dim_plot)))
                    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=(10, 10), sharex=True, dpi=100, facecolor='w', edgecolor='k')
                    ax=ax.ravel()
                    for idx in range(args.num_dim_plot):
                        ax[idx].plot(new_times,z_p[:,idx],c='r', label='model')
                        # plt.scatter(to_np(times)[:extrapolation_points],obs_print[:extrapolation_points,0]*scaling_factor,label='Data',c='blue')
                        ax[idx].scatter(to_np(times)[:],obs_print[:,idx],label='Data',c='blue', alpha=0.5)
                        ax[idx].set_xlabel("Time")
                        ax[idx].set_ylabel("dim"+str(idx))
                        #plt.scatter(to_np(times)[extrapolation_points:],obs_print[extrapolation_points:,0,0],label='Data extr',c='red')
                        ax[idx].legend()
                        # timestr = time.strftime("%Y%m%d-%H%M%S")
                    fig.tight_layout()
                    plt.savefig(os.path.join(path_to_save_plots, 'plot_ndim_epoch'+str(i)))

                    if 'calcium_imaging' in args.experiment_name:
                        # Plot the first 20 frames
                        data_to_plot = obs_print[:20,:]*args.scaling_factor #Get the first 10 samples for a test 
                        predicted_to_plot = z_p[:20,:]*args.scaling_factor
                        data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                        predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                        predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                        data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                        fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                        c=0
                        for idx_row in range (2): 
                            for idx_col in range(10):
                                ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                ax[2*idx_row,idx_col].axis('off')
                                _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                ax[2*idx_row+1,idx_col].axis('off')
                                c+=1
                        fig.tight_layout()
                        plt.savefig(os.path.join(path_to_save_plots, 'plot_first20frame_rec'+str(i)))


                        # Plot the last 20 frames  
                        data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                        predicted_to_plot = z_p[-20:,:]*args.scaling_factor
                        data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                        predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                        predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                        data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                        fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                        c=0
                        for idx_row in range (2): 
                            for idx_col in range(10):
                                ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                ax[2*idx_row,idx_col].axis('off')
                                _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                ax[2*idx_row+1,idx_col].axis('off')
                                c+=1
                        fig.tight_layout()
                        plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_rec'+str(i)))


                        #Plot the R2 and MSE loss between the original data and the predicted overtime. 
                        data_to_plot = obs_print[:,:]*args.scaling_factor #Get the first 10 samples for a test 
                        predicted_to_plot = z_p[:,:]*args.scaling_factor
                        data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                        predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                        all_r2_scores = []
                        all_mse_scores = []

                        for idx_frames in range(len(data_to_plot)):
                            _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                            all_r2_scores.append(r_value)
                            # print('data_to_plot[idx_frames,:].flatten().shape: ',data_to_plot[idx_frames,:].flatten().shape)
                            # print('predicted_to_plot[idx_frames,:].flatten().shape: ',predicted_to_plot[idx_frames,:].flatten().shape)
                            tmp_mse_loss = mean_squared_error(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                            all_mse_scores.append(tmp_mse_loss)

                        fig,ax = plt.subplots(2,1, figsize=(15,5), sharex=True, facecolor='w')
                        ax[0].plot(np.arange(len(all_r2_scores)),all_r2_scores)
                        ax[1].plot(np.arange(len(all_mse_scores)),all_mse_scores)
                        ax[1].set_xlabel("Frames")
                        ax[0].set_ylabel("R2")
                        ax[1].set_ylabel("MSE")
                        fig.tight_layout()
                        plt.savefig(os.path.join(path_to_save_plots, 'plot_performance_rec'+str(i)))

                        #Plot integral and ode part separated
                        if ode_func is not None and F_func is not None:
                            Trained_Data_ode = odeint(ode_func,torch.Tensor(obs_print[0,:]).flatten().to(args.device),times.to(args.device),rtol=1e-4,atol=1e-4)
                            Trained_Data_ode_print = to_np(Trained_Data_ode)
                            Trained_Data_integral_print  = z_p - Trained_Data_ode_print
                            # print('Trained_Data_integral_print.max():',np.abs(Trained_Data_integral_print).max())
                            # print('Trained_Data_ode_print.max():',np.abs(Trained_Data_ode_print).max())

                            data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                            predicted_to_plot_ode = Trained_Data_ode_print[-20:,:]*args.scaling_factor
                            predicted_to_plot_ide = Trained_Data_integral_print[-20:,:]*args.scaling_factor
                            data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                            predicted_to_plot_ode = args.fitted_pca.inverse_transform(predicted_to_plot_ode)
                            predicted_to_plot_ide = args.fitted_pca.inverse_transform(predicted_to_plot_ide)

                            predicted_to_plot_ode = predicted_to_plot_ode.reshape(predicted_to_plot_ode.shape[0],184, 208) # Add the original frame dimesion as input
                            predicted_to_plot_ide = predicted_to_plot_ide.reshape(predicted_to_plot_ide.shape[0],184, 208)
                            data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                            fig,ax = plt.subplots(6,10, figsize=(15,8), facecolor='w')
                            c=0
                            step = 0
                            for idx_row in range (2): 
                                for idx_col in range(10):
                                    ax[2*idx_row+step,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row+step,idx_col].axis('off')

                                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ode[c,:].flatten())
                                    ax[2*idx_row+1+step,idx_col].set_title('ODE R2: {:.3f}'.format(r_value**2))
                                    ax[2*idx_row+1+step,idx_col].imshow(predicted_to_plot_ode[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row+1+step,idx_col].axis('off')

                                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ide[c,:].flatten())
                                    ax[2*idx_row+2+step,idx_col].set_title('IDE R2: {:.3f}'.format(r_value**2))
                                    ax[2*idx_row+2+step,idx_col].imshow(predicted_to_plot_ide[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row+2+step,idx_col].axis('off')
                                    c+=1
                                step += 1
                            fig.tight_layout()
                            plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_ode_vs_ide_rec'+str(i)))




                    if F_func is not None:
                        # fig,ax = plt.subplots(1,1, figsize=(8,8), facecolor='w')
                        F_out = to_np(F_func.forward(z_test,ts_test))
                        # print('F_out.shape: ',F_out.shape)
                        # if F_out.shape[1]>2: 
                        #     # reducer = umap.UMAP(n_components=2, random_state=1) 
                        #     reducer = PCA(n_components=2)
                        #     F_out = reducer.fit_transform(F_out)
                        n_plots_x = int(np.ceil(np.sqrt(F_out.shape[1])))
                        n_plots_y = int(np.floor(np.sqrt(F_out.shape[1])))
                        fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=(10, 10), sharex=True, dpi=100, facecolor='w', edgecolor='k')
                        ax=ax.ravel()
                        for idx in range(args.num_dim_plot):
                            # plt.scatter(to_np(times)[:extrapolation_points],obs_print[:extrapolation_points,0]*scaling_factor,label='Data',c='blue')
                            ax[idx].scatter(to_np(times)[:],F_out[:,idx],label='F_out',c='blue', alpha=0.5)
                            ax[idx].set_xlabel("Time")
                            ax[idx].set_ylabel("F"+str(idx))
                            #plt.scatter(to_np(times)[extrapolation_points:],obs_print[extrapolation_points:,0,0],label='Data extr',c='red')
                            ax[idx].legend()
                            # timestr = time.strftime("%Y%m%d-%H%M%S")
                        fig.tight_layout()

                        # ax.plot(F_out[:,0],F_out[:,1])
                        # ax.scatter(F_out[:,0],F_out[:,1], c = np.arange(len(F_out)))
                        # ax.set_xlabel("F_0")
                        # ax.set_ylabel("F_1")
                        plt.savefig(os.path.join(path_to_save_plots, 'plot_F_func'+str(i)))


                    plt.close('all')
                    '''
                test_loss /= Data_test.size(0)
                test_losses.append(test_loss)
                print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, Data_test.size(0),
                    100. * correct / Data_test.size(0)))
                accuracy.append(100. * to_np(correct) / Data_test.size(0))
                plt.figure(1,facecolor='w')
                plt.plot(accuracy,label='Acc %')
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.savefig(os.path.join(path_to_save_plots,'Accuracy'))
                
            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            if args.kernel_split is True:
                model_state = {
                        'epoch': i + 1,
                        'state_dict': kernel.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }
            else:
                model_state = {
                        'epoch': i + 1,
                        'state_dict': G_NN.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            if split_size>0:
                if args.free_func_nn is False:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, G_NN, kernel, F_func, None)
                else:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, G_NN, kernel, F_func, f_func)
            else: 
                if args.free_func_nn is False:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, G_NN, kernel, F_func, None)
                else:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, G_NN, kernel, F_func, f_func)

            #lr_scheduler(loss_validation)
            if len(all_val_loss)>0:
                early_stopping(all_val_loss[-1])
            else:
                early_stopping(all_train_loss[-1])
            if early_stopping.early_stop:
                break


        end = time.time()
        # print(f"Training time: {(end-start)/60:.3f} minutes")
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),Loss_print)
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),Val_Loss)
        # # plt.savefig('trained.png')
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'trained'+timestr))
        # # plt.show()
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'final_losses'+timestr))
        # # plt.show()
    elif args.mode=='evaluate':
        print('Running in evaluation mode')
        ## Validating
        model.eval()
        with torch.no_grad():
            splitting_size = int(args.training_split*Data.size(0))
            all_r2_scores = []
            all_mse = []
            for j in tqdm(range(Data.size(0)-splitting_size)):
                Dataset_all = Test_Dynamics_Dataset(Data[j,:,:],times)
                loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np)))

                obs_test, ts_test, ids_test = next(iter(loader_test))
                ids_test, indices = torch.sort(ids_test)
                # print('indices: ',indices)
                obs_test = obs_test[indices,:]
                ts_test = ts_test[indices]
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)


                obs_test = obs_test.to(args.device)
                ts_test = ts_test.to(args.device)
                ids_test = ids_test.to(args.device)
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)
                # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                
                if Encoder is not None:
                        obs_test = Encoder(obs_test)
                
                c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,obs_test.size(0)).to(device), obs_test)
                interpolation = NaturalCubicSpline(c_coeffs)
                c = lambda x: interpolation.evaluate(x).squeeze()

                z_test = IESolver_monoidal(
                                torch.linspace(0,1,2).to(device),
                                #y_0 = obs_[0].flatten().to(device), 
                                #c = lambda x: obs_test[0].flatten().to(device), 
                                c = c,
                                d = lambda x,y: torch.Tensor([1]).to(device), 
                                k = kernel, 
                                f = F_func,
                                G = G_NN,
                                lower_bound = lambda x: torch.Tensor([0]).to(device),
                                upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                                max_iterations = args.max_iterations,
                                kernel_nn=True,
                                kernel_split = args.kernel_split,
                                G_nn = args.G_NN,
                                integration_dim = 0,
                                mc_samplings=1000,
                                num_internal_points = args.num_internal_points
                                ).solve()
                
                z_test = z_test[-1:,:]
                if Decoder is not None:
                    z_test = Decoder(z_test)
                z_test = soft_max(z_test)
                # z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_test)
                obs_print = to_np(obs_test)
                
                new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                plt.figure(figsize=(8,8),facecolor='w')
                z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_p)

                plt.figure(1, facecolor='w')
                plt.plot(z_p[:,0],z_p[:,1],c='r', label='model')
                obs_print = to_np(obs_test[:,:])
                # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                plt.scatter(obs_print[:,0],obs_print[:,1],label='Data',c='blue', alpha=0.5)
                plt.xlabel("dim 0")
                plt.ylabel("dim 1")
                #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                plt.legend()
                
                _, _, r_value, _, _ = scipy.stats.linregress(z_p[:,:].flatten(),obs_print[:,:].flatten())
                mse_value = mean_squared_error(z_p[:,:].flatten(),obs_print[:,:].flatten())
                
                print('R2:',r_value)
                print('MSE:',mse_value)
                
                all_r2_scores.append(r_value)
                all_mse.append(mse_value)
            
            plt.figure(2,facecolor='w')
            plt.plot(np.linspace(0,len(all_r2_scores),len(all_r2_scores)),all_r2_scores)
            plt.xlabel("Dynamics")
            plt.ylabel("R2")
            plt.legend()
            
            plt.figure(3,facecolor='w')
            plt.plot(np.linspace(0,len(all_mse),len(all_mse)),all_mse)
            plt.xlabel("Dynamics")
            plt.ylabel("MSE")
            plt.legend()
            
            print("Average R2:",sum(all_r2_scores)/len(all_r2_scores))
            print("Average MSE:",sum(all_mse)/len(all_mse))
                
            for j in tqdm(range(Data.size(0)-splitting_size,Data.size(0))):
                Dataset_all = Test_Dynamics_Dataset(Data[j,:,:],times)
                loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np)))

                obs_test, ts_test, ids_test = next(iter(loader_test))
                ids_test, indices = torch.sort(ids_test)
                # print('indices: ',indices)
                obs_test = obs_test[indices,:]
                ts_test = ts_test[indices]
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)


                obs_test = obs_test.to(args.device)
                ts_test = ts_test.to(args.device)
                ids_test = ids_test.to(args.device)
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)
                # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                
                if Encoder is not None:
                        obs_test = Encoder(obs_test)
                
                c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,obs_test.size(0)).to(device), obs_test)
                interpolation = NaturalCubicSpline(c_coeffs)
                c = lambda x: interpolation.evaluate(x).squeeze()

                z_test = IESolver_monoidal(
                                torch.linspace(0,1,2).to(device),
                                #y_0 = obs_[0].flatten().to(device), 
                                #c = lambda x: obs_test[0].flatten().to(device), 
                                c = c,
                                d = lambda x,y: torch.Tensor([1]).to(device), 
                                k = kernel, 
                                f = F_func,
                                G = G_NN,
                                lower_bound = lambda x: torch.Tensor([0]).to(device),
                                upper_bound = lambda x: x,#torch.Tensor([t_max]).to(device),
                                max_iterations = args.max_iterations,
                                kernel_nn=True,
                                kernel_split = args.kernel_split,
                                G_nn = args.G_NN,
                                integration_dim = 0,
                                mc_samplings=1000,
                                num_internal_points = args.num_internal_points
                                ).solve()
                
                z_test = z_test[-1:,:]
                if Decoder is not None:
                    z_test = Decoder(z_test)
                z_test = soft_max(z_test)
                # z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_test)
                obs_print = to_np(obs_test)
                
                new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                plt.figure(figsize=(8,8),facecolor='w')
                z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_p)

                plt.figure(2, facecolor='w')
                plt.plot(z_p[:,0],z_p[:,1],c='r', label='model')
                obs_print = to_np(obs_test[:,:])
                # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                plt.scatter(obs_print[:,0],obs_print[:,1],label='Data',c='blue', alpha=0.5)
                plt.xlabel("dim 0")
                plt.ylabel("dim 1")
                #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                plt.legend()
                      
                print(scipy.stats.linregress(z_p[:,:].flatten(),obs_print[:,:].flatten()))
                print(mean_squared_error(z_p[:,:].flatten(),obs_print[:,:].flatten()))
                
                '''
                # Plot the last 20 frames  
                data_to_plot = obs_print[:,:]#*args.scaling_factor #Get the first 10 samples for a test 
                predicted_to_plot = z_p[:,:]*args.scaling_factor
                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                c=0
                for idx_row in range (2): 
                    for idx_col in range(10):
                        ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                        ax[2*idx_row,idx_col].axis('off')
                        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                        ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                        ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                        ax[2*idx_row+1,idx_col].axis('off')
                        c+=1
                fig.tight_layout()

                #Plot the R2 and MSE loss between the original data and the predicted overtime. 
                data_to_plot = obs_print[:,:]*args.scaling_factor #Get the first 10 samples for a test 
                predicted_to_plot = z_p[:,:]*args.scaling_factor
                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                all_r2_scores = []
                all_mse_scores = []

                for idx_frames in range(len(data_to_plot)):
                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                    all_r2_scores.append(r_value)
                    # print('data_to_plot[idx_frames,:].flatten().shape: ',data_to_plot[idx_frames,:].flatten().shape)
                    # print('predicted_to_plot[idx_frames,:].flatten().shape: ',predicted_to_plot[idx_frames,:].flatten().shape)
                    tmp_mse_loss = mean_squared_error(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                    all_mse_scores.append(tmp_mse_loss)

                fig,ax = plt.subplots(2,1, figsize=(15,5), sharex=True, facecolor='w')
                ax[0].plot(np.arange(len(all_r2_scores)),all_r2_scores)
                ax[1].plot(np.arange(len(all_mse_scores)),all_mse_scores)
                ax[1].set_xlabel("Frames")
                ax[0].set_ylabel("R2")
                ax[1].set_ylabel("MSE")
                fig.tight_layout()

                print('R2: ',all_r2_scores)
                print('MSE: ',all_mse_scores)
                '''
                
def Full_experiment_AttentionalIE_PDE(model, Encoder, Decoder, Data, time_seq, index_np, mask, times, args, extrapolation_points): # experiment_name, plot_freq=1):
    # scaling_factor=1
    
    
    #metadata for saving checkpoints
    if args.model=='nie': 
        str_model_name = "nie"
    elif args.model=='node': 
        str_model_name = "node"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)

    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else: 
            num_experiments = [int(i[3:]) for i in txt]
            num_experiments = np.array(num_experiments).max()
         # -- logger location
        #writer = SummaryWriter(os.path.join(path_to_experiment,'run'+str(num_experiments+1)))
        #print('writer.log_dir: ',writer.log_dir)
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)
            
#         with open(os.path.join(writer.log_dir,'commandline_args.txt'), 'w') as f:
#             for key, value in args.__dict__.items(): 
#                 f.write('%s:%s\n' % (key, value))



    obs = Data
    times = time_seq
    
    
    if Encoder is None and Decoder is None:
        All_parameters = model.parameters()
    elif Encoder is not None and Decoder is None:
        All_parameters = list(model.parameters())+list(Encoder.parameters())
    elif Decoder is not None and Encoder is None:
        All_parameters = list(model.parameters())+list(Decoder.parameters())
    else:
        All_parameters = list(model.parameters())+list(Encoder.parameters())+list(Decoder.parameters())
    
     
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0,last_epoch=-1)# Emanuele's version
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,last_epoch=-1) #My first version
    #scheduler = LRScheduler(optimizer,patience = 20,min_lr=1e-12,factor=0.1)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0,last_epoch=-1)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)
    elif args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        
        if Encoder is None or Decoder is None:
            model, optimizer, scheduler, pos_enc, pos_dec, f_func = load_checkpoint(path, model, optimizer, scheduler, None, None,  None)
        else:
            G_NN, optimizer, scheduler, model, Encoder, Decoder = load_checkpoint(path, None, optimizer, scheduler, model, Encoder, Decoder)


    
    if args.mode=='train':
        #lr_scheduler = LRScheduler(optimizer,patience = 50,min_lr=1e-5,factor=0.1)
        early_stopping = EarlyStopping(patience=1000,min_delta=0)

        # Loss_print = []
        # Val_Loss = []
        all_train_loss=[]
        all_val_loss=[]
        
            
        Data_splitting_indices = Train_val_split(np.copy(index_np),0)
        Train_Data_indices = Data_splitting_indices.train_IDs()
        Val_Data_indices = Data_splitting_indices.val_IDs()
        print('\nlen(Train_Data_indices): ',len(Train_Data_indices))
        print('Train_Data_indices: ',Train_Data_indices)
        print('\nlen(Val_Data_indices): ',len(Val_Data_indices))
        print('Val_Data_indices: ',Val_Data_indices)
        
        # Train Neural IDE
        get_times = Select_times_function(times,extrapolation_points)

        save_best_model = SaveBestModel()
        start = time.time()
        
        split_size = int(args.training_split*obs.size(0))
        
        obs_train = obs[:obs.size(0)-split_size,:,:]
        
        for i in range(args.epochs):
            
            if args.support_tensors is True or args.support_test is True:
                if args.combine_points is True:
                    sampled_tensors,real_idx = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(0, 1))
                    temp_sampled_tensors = sampled_tensors
                    sampled_tensors = sampled_tensors.to(device)
                    #Check if there are duplicates and resample if there are
                    sampled_tensors = torch.cat([times,sampled_tensors])
                    dup=np.array([0])
                    while dup.size != 0:
                        u, c = np.unique(temp_sampled_tensors, return_counts=True)
                        dup = u[c > 1]
                        if dup.size != 0:
                            sampled_tensors,real_idx = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(0, 1))
                            sampled_tensors = sampled_tensors.to(device)
                            sampled_tensors = torch.cat([times,sampled_tensors])
                    dummy_times=sampled_tensors
                    real_idx=real_idx[:times.size(0)]
                if args.combine_points is False:
                        dummy_times = torch.linspace(times[0],times[-1],args.sampling_points)
            
            model.train()
            if Encoder is not None:
                Encoder.train()
            if Decoder is not None:    
                Decoder.train()
            
            start_i = time.time()
            print('Epoch:',i)
            # GPUtil.showUtilization()
            counter=0
            train_loss = 0.0
            
            if args.n_batch>1:
                obs_shuffle = obs_train[torch.randperm(obs_train.size(0)),:,:]
                
            for j in tqdm(range(0,obs.size(0)-split_size,args.n_batch)):
                
                if args.n_batch==1:
                    Dataset_train = Dynamics_Dataset(obs_train[j,:,:],times)
                else:
                    Dataset_train = Dynamics_Dataset(obs_shuffle[j:j+args.n_batch,:,:],times,args.n_batch)
                #Dataset_val = Dynamics_Dataset(obs[j-split_size,:,:],times)
                # loader = torch.utils.data.DataLoader(Dataset, batch_size = batch_size)
                # Dataset_val = Val_Dynamics_Dataset(Data,Val_Data_indices,times)

                # For the sampler
                train_sampler = SubsetRandomSampler(Train_Data_indices)
                #valid_sampler = SubsetRandomSampler(Val_Data_indices)

                # loader_val = torch.utils.data.DataLoader(Dataset, batch_size = args.batch_size)

                dataloaders = {'train': torch.utils.data.DataLoader(Dataset_train, sampler=train_sampler,\
                                                                    batch_size = args.n_batch, drop_last=True),
                              }

                train_loader = dataloaders['train']
                #val_loader = dataloaders['val']
                #loader_test = dataloaders['test']

            #for obs_, ts_, ids_ in tqdm(train_loader): 
                obs_, ts_, ids_ = Dataset_train.__getitem__(index_np)#next(iter(train_loader))
                
#                 obs_ = obs_.to(args.device)
#                 ts_ = times.to(args.device)
#                 ids_ = torch.from_numpy(ids_).to(args.device)
#                 # obs_, ts_, ids_ = next(iter(loader))

#                 ids_, indices = torch.sort(ids_)
#                 ts_ = ts_[indices]
#                 ts_ = torch.cat([times[:1],ts_])
#                 if args.n_batch==1:
#                     obs_ = obs_[indices,:]
#                     #obs_ = torch.cat([obs[j,:1,:],obs_])
#                 else:
#                     obs_ = obs_[:,indices,:]
#                     #obs_ = torch.cat([obs[j:j+args.n_batch,:1,:],obs_],1)
                    
#                 if args.perturbation_to_obs0 is not None:
#                        perturb = torch.normal(mean=torch.zeros(obs_.shape[1]).to(args.device),
#                                               std=args.std_noise)#args.perturbation_to_obs0*obs_[:3,:].std(dim=0))
#                 else:
#                     perturb = torch.zeros_like(obs_[0]).to(args.device)
                
                if args.n_batch==1:
                    c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_[:,:1])
                    interpolation = NaturalCubicSpline(c_coeffs)
                    c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,args.time_points).unsqueeze(-1)
                else:
                    if Encoder is None and args.fourier_transform is True:
                        c = lambda x: torch.fft.rfft(obs_[:,:,:1],dim=1)\
                                      [:,:args.modes,:].unsqueeze(-1)\
                                      .repeat(1,1,args.time_points,1)\
                                      .to(args.device)
                    else:
#                         c = lambda x: Encoder(obs_[:,:,:1].permute(0,2,1).requires_grad_(True))\
#                                     .permute(0,2,1)\
#                                     .repeat(1,1,args.time_points)\
#                                     .unsqueeze(-1).contiguous().to(args.device)
                        
#                         y_0 = Encoder(obs_[:,:,:1].permute(0,2,1))\
#                                     .permute(0,2,1)\
#                                     .unsqueeze(-1)[:,:,:1,:]
                          c = lambda x: Encoder(obs_[:,:,:1].permute(0,2,1).requires_grad_(True))\
                                        .permute(0,2,1).unsqueeze(-2).repeat(1,1,args.time_points,1)\
                                        .contiguous().to(args.device)
                          y_0 = Encoder(obs_[:,:,:1].permute(0,2,1).requires_grad_(True))\
                                .permute(0,2,1).unsqueeze(-2)
                        
                
                if Encoder is None:
                    y_0 = obs_[:,:,:1].unsqueeze(-1)
                    
                if args.fourier_transform is True:
                    y_0 = torch.fft.rfft(obs_[:,:,:1],dim=1)\
                          [:,:args.modes,:].unsqueeze(-1).to(args.device)
                    
                if args.ts_integration is None:
                    times_integration = torch.linspace(0,1,args.time_points).to(args.device)
                else:
                    times_integration = args.ts_integration.to(args.device)
                
                
                if args.n_batch==1:
                    z_ = Integral_spatial_attention_solver(
                            torch.linspace(0,1,args.time_points).to(device),
                            obs_[0].unsqueeze(1).to(args.device),
                            c=c,
                            sampling_points = args.time_points,
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            spatial_integration=True,
                            spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                            spatial_domain_dim=1,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            smoothing_factor=args.smoothing_factor,
                            use_support=False,
                            ).solve()
                else:
                    z_ = Integral_spatial_attention_solver_multbatch(
                            times_integration,
                            y_0.to(args.device),
                            c=c,
                            sampling_points = args.time_points,
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            spatial_integration=True,
                            spatial_domain= torch.meshgrid(\
                                            [torch.linspace(0,1,args.n_points) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                            spatial_domain_dim=1,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            smoothing_factor=args.smoothing_factor,
                            use_support=args.support_tensors,
                            accumulate_grads=True
                            ).solve()
                        
                if Encoder is None:
                    z_ = z_.view(args.n_batch,args.n_points,args.time_points)
                if args.burgers_t==2:
                    if args.n_batch==1:
                        z_ = torch.cat([z_[:,:1],z_[:,-1:]],-1)
                    else:
                        z_ = torch.cat([z_[:,:,:1],z_[:,:,-1:]],-1)
                if Decoder is not None:
                    z_ = Decoder(z_.requires_grad_(True))
                else:
                    z_ = z_.view(args.n_batch,Data.shape[1],args.time_points)
                   
                #loss_ts_ = get_times.select_times(ts_)[1]
                loss = F.mse_loss(z_, obs_.detach()) #Original 
                # print('z_[:,:].to(args.device): ',z_[:,:].to(args.device))
                # print('obs_.to(args.device).detach()[:,:]: ',obs_.to(args.device).detach()[:,:])
                # loss = F.mse_loss(z_[:,:].to(args.device), obs_.to(args.device).detach()[:,:]) #Original 

                   
                # ###############################
                # Loss_print.append(to_np(loss))
                # ###############################

                optimizer.zero_grad()
                loss.backward()#(retain_graph=True)
                optimizer.step()

                # n_iter += 1
                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  split_size==0 and args.lr_scheduler == 'StepLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, obs_, ts_, z_, ids_

            ## Validating
                
            model.eval()
            if Encoder is not None:
                Encoder.eval()
            if Decoder is not None:
                Decoder.eval()
                
            with torch.no_grad():

                    #Only do this if there is a validation dataset
                
                val_loss = 0.0
                counter = 0
                if split_size>0:
                    # for images, _, _, _, _ in tqdm(val_loader):   # frames, timevals, angular_velocity, mass_height, mass_xpos
                    for j in tqdm(range(obs.size(0)-split_size,obs.size(0),args.n_batch)):
                        
                        valid_sampler = SubsetRandomSampler(Train_Data_indices)
                        if args.n_batch==1:
                            Dataset_val = Dynamics_Dataset(obs[j,:,:],times)
                        else:
                            Dataset_val = Dynamics_Dataset(obs[j:j+args.n_batch,:,:],times,args.n_batch)
                        
                        val_loader = torch.utils.data.DataLoader(Dataset_val, sampler=valid_sampler,\
                                                                 batch_size = args.n_batch, drop_last=True)
                    
                    #for obs_val, ts_val, ids_val in tqdm(val_loader):
                        obs_val, ts_val, ids_val = Dataset_val.__getitem__(index_np)#next(iter(val_loader))
                        obs_val = obs_val.to(args.device)
                        ts_val = ts_val.to(args.device)
                        
                        ids_val = torch.from_numpy(ids_val).to(args.device)

                        ids_val, indices = torch.sort(ids_val)
                        # print('indices: ',indices)
                        if args.n_batch ==1:
                            obs_val = obs_val[indices,:]
                        else:
                            obs_val = obs_val[:,indices,:]
                        ts_val = ts_val[indices]
                                             

                        #Concatenate the first point of the train minibatch
                        # obs_[0],ts_
                        # print('\n In validation mode...')
                        # print('obs_[:5]: ',obs_[:5])
                        # print('ids_[:5]: ',ids_[:5])
                        # print('ts_[:5]: ',ts_[:5])
                        # print('ts_[0]:',ts_[0])

                        ## Below is to add initial data point to val
                        #obs_val = torch.cat((obs_[0][None,:],obs_val))
                        #ts_val = torch.hstack((ts_[0],ts_val))
                        #ids_val = torch.hstack((ids_[0],ids_val))

                        # obs_val, ts_val, ids_val = next(iter(loader_val))
                        # print('obs_val.shape: ',obs_val.shape)
                        # print('ids_val: ',ids_val)
                        # print('ts_val: ',ts_val)

                        # obs_val, ts_val = obs_val.squeeze(1), ts_val.squeeze(1)
                        if args.n_batch==1:
                            c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_val[:,:1])
                            interpolation = NaturalCubicSpline(c_coeffs)
                            c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,args.time_points).unsqueeze(-1)
                        else:
                            if Encoder is None:
                                c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_val[:,:,:1])
                                interpolation = NaturalCubicSpline(c_coeffs)
                                c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,1,args.time_points).unsqueeze(-1)
                            else:
#                                 c = lambda x: Encoder(obs_val[:,:,:1].permute(0,2,1))\
#                                             .permute(0,2,1)\
#                                             .repeat(1,1,args.time_points)\
#                                             .unsqueeze(-1).contiguous().to(args.device)

#                                 y_0 = Encoder(obs_val[:,:,:1].permute(0,2,1))\
#                                             .permute(0,2,1)\
#                                             .unsqueeze(-1)[:,:,:1,:]
                                  c = lambda x: Encoder(obs_val[:,:,:1].permute(0,2,1).requires_grad_(True))\
                                                .permute(0,2,1).unsqueeze(-2).repeat(1,1,args.time_points,1)\
                                                .contiguous().to(args.device)
                                  y_0 = Encoder(obs_val[:,:,:1].permute(0,2,1).requires_grad_(True))\
                                        .permute(0,2,1).unsqueeze(-2)
                        
                        if Encoder is None:
                            y_0 = obs_val[:,:,:1].unsqueeze(-1)
                            
                        if args.ts_integration is None:
                            times_integration = torch.linspace(0,1,args.time_points).to(args.device)
                        else:
                            times_integration = args.ts_integration.to(args.device)
                            
                        
                        if args.n_batch==1:
                            z_val = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_val[0].unsqueeze(1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()
                        else:
                            z_val = Integral_spatial_attention_solver_multbatch(
                                times_integration,
                                y_0.to(args.device),
                                c=c,
                                sampling_points = args.time_points,
                                mask=mask,
                                Encoder = model,
                                max_iterations = args.max_iterations,
                                spatial_integration=True,
                                spatial_domain= torch.meshgrid(\
                                            [torch.linspace(0,1,args.n_points) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                spatial_domain_dim=1,
                                #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                smoothing_factor=args.smoothing_factor,
                                use_support=args.support_tensors,
                                ).solve()
                        
                        if Encoder is None:  
                            z_val = z_val.view(args.n_batch,args.n_points,args.time_points)
                        if args.burgers_t==2:
                            if args.n_batch==1:
                                z_val = torch.cat([z_val[:,:1],z_val[:,-1:]],-1)
                            else:
                                z_val = torch.cat([z_val[:,:,:1],z_val[:,:,-1:]],-1)
                        if Decoder is not None:
                            z_val = Decoder(z_val)
                        else:
                            z_val = z_val.view(args.n_batch,Data.shape[1],args.time_points)
                        #validation_ts_ = get_times.select_times(ts_val)[1]
                        loss_validation = F.mse_loss(z_val, obs_val.detach())
                        # Val_Loss.append(to_np(loss_validation))
                        
                        del obs_val, ts_val, z_val, ids_val

                        counter += 1
                        val_loss += loss_validation.item()
                        
                        del loss_validation

                        #LRScheduler(loss_validation)
                        if args.lr_scheduler == 'ReduceLROnPlateau':
                            scheduler.step(val_loss)
                
                
                else: counter += 1

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            #writer.add_scalar('train_loss', all_train_loss[-1], global_step=i)
            #if len(all_val_loss)>0:
            #    writer.add_scalar('val_loss', all_val_loss[-1], global_step=i)
            #if args.lr_scheduler == 'ReduceLROnPlateau':
            #    writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], global_step=i)
            #elif args.lr_scheduler == 'CosineAnnealingLR':
            #    writer.add_scalar('Epoch/learning_rate', scheduler.get_last_lr()[0], global_step=i)

            
            with torch.no_grad():
                
                model.eval()
                if Encoder is not None:
                    Encoder.eval()
                if Decoder is not None:
                    Decoder.eval()

                if i % args.plot_freq == 0:
                    
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
                    # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
                        
                    plt.plot(np.log10(all_train_loss),label='Train loss')
                    if split_size>0:
                        plt.plot(np.log10(all_val_loss),label='Val loss')
                    plt.xlabel("Epoch")
                    plt.ylabel("MSE Loss")
                    # timestr = time.strftime("%Y%m%d-%H%M%S")
                    #plt.show()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))

                    for j in tqdm(range(0,args.n_batch,args.n_batch)):
                        if args.n_batch==1:
                            Dataset_all = Dynamics_Dataset(Data[j,:,:],times)
                        else:
                            Dataset_all = Dynamics_Dataset(obs[j:j+args.n_batch,:,:],times,args.n_batch)
                            
                        loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = args.n_batch)

                        obs_test, ts_test, ids_test = Dataset_all.__getitem__(index_np)#next(iter(loader_test))

                        ids_test, indices = torch.sort(torch.from_numpy(ids_test))
                        # print('indices: ',indices)
                        if args.n_batch==1:
                            obs_test = obs_test[indices,:]
                        else:
                            obs_test = obs_test[:,indices,:]
                        ts_test = ts_test[indices]
                        # print('obs_test.shape: ',obs_test.shape)
                        # print('ids_test: ',ids_test)
                        # print('ts_test: ',ts_test)


                        obs_test = obs_test.to(args.device)
                        ts_test = ts_test.to(args.device)
                        ids_test = ids_test.to(args.device)
                        # print('obs_test.shape: ',obs_test.shape)
                        # print('ids_test: ',ids_test)
                        # print('ts_test: ',ts_test)
                        # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                        if args.n_batch ==1:
                            c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_test[:,:1])
                            interpolation = NaturalCubicSpline(c_coeffs)
                            c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,args.time_points).unsqueeze(-1)
                        else:
                            if Encoder is None:
                                c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_test[:,:,:1])
                                interpolation = NaturalCubicSpline(c_coeffs)
                                c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,1,args.time_points).unsqueeze(-1)
                            else:
#                                 c = lambda x: Encoder(obs_test[:,:,:1].permute(0,2,1))\
#                                             .permute(0,2,1)\
#                                             .repeat(1,1,args.time_points)\
#                                             .unsqueeze(-1).contiguous().to(args.device)

#                                 y_0 = Encoder(obs_test[:,:,:1].permute(0,2,1))\
#                                             .permute(0,2,1)\
#                                             .unsqueeze(-1)[:,:,:1,:]
                                  c = lambda x: Encoder(obs_test[:,:,:1].permute(0,2,1).requires_grad_(True))\
                                                .permute(0,2,1).unsqueeze(-2).repeat(1,1,args.time_points,1)\
                                                .contiguous().to(args.device)
                                  y_0 = Encoder(obs_test[:,:,:1].permute(0,2,1).requires_grad_(True))\
                                        .permute(0,2,1).unsqueeze(-2)
        
                        if Encoder is None:
                            y_0 = obs_test[:,:,:1].unsqueeze(-1) 
                            
                        if args.ts_integration is None:
                            times_integration = torch.linspace(0,1,args.time_points).to(args.device)
                        else:
                            times_integration = args.ts_integration.to(args.device)
                        
                        
                        if args.n_batch ==1:
                            z_test = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_test[0].unsqueeze(1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                    spatial_domain_dim=1,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()
                        else:
                            z_test = Integral_spatial_attention_solver_multbatch(
                                times_integration,
                                y_0.to(args.device),
                                c=c,
                                sampling_points = args.time_points,
                                mask=mask,
                                Encoder = model,
                                max_iterations = args.max_iterations,
                                spatial_integration=True,
                                spatial_domain= torch.meshgrid(\
                                            [torch.linspace(0,1,args.n_points) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                spatial_domain_dim=1,
                                #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                smoothing_factor=args.smoothing_factor,
                                use_support=args.support_tensors,
                                ).solve()
                        
                            
                        #print('Parameters are:',ide_trained.parameters)
                        #print(list(All_parameters))
                        if Decoder is not None:
                            z_test = Decoder(z_test)
                        else:
                            z_test = z_test.view(args.n_batch,Data.shape[1],args.time_points)
                            
                        if args.print_ts is True:
                            z_ts = z_test[0,:,:]
                            
                        if args.n_batch== 1:
                            z_test = torch.cat([z_test[:,:1],z_test[:,-1:]],-1)
                        else:
                            z_test = torch.cat([z_test[:,:,:1],z_test[:,:,-1:]],-1)
                            
                        new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                        #plt.figure(figsize=(8,8),facecolor='w')
                        
                        z_p = z_test
                        if args.n_batch >1:
                            z_p = z_test[0,:,:]
                        z_p = to_np(z_p)
                        
                        if args.n_batch >1:
                            obs_print = to_np(obs_test[0,:,:])
                        else:
                            obs_print = to_np(obs_test[:,:])
                  
                        
                        
                        plt.figure(3, facecolor='w')
                        
                        plt.plot(torch.linspace(0,1,Data.shape[1]),z_p[:,0],c='green', label='model_t0',linewidth=3)
                        #plt.scatter(torch.linspace(0,1,args.n_points),z_p[:,0],c='green',s=10)
                        plt.plot(torch.linspace(0,1,Data.shape[1]),z_p[:,-1],c='orange', label='model_t1',linewidth=3)
                        #plt.scatter(torch.linspace(0,1,args.n_points),z_p[:,1],c='orange',s=10)
                       
                        
                        plt.scatter(torch.linspace(0,1,obs_test.shape[1]),obs_print[:,0],label='Data_t0',c='red', alpha=0.5)
                        plt.scatter(torch.linspace(0,1,obs_test.shape[1]),obs_print[:,-1],label='Data_t1',c='blue', alpha=0.5)
                        
                        plt.legend()
                        
                        plt.savefig(os.path.join(path_to_save_plots,'plot_t0t1_epoch'+str(i)+'_'+str(j)))
                        
                        if args.print_ts is True and i%args.freq_print_ts == 0:
                            z_ts_print = to_np(z_ts)
                            
                            plt.figure(4, facecolor='w')
                            for plot_i in range(args.time_points):
                                plt.plot(
                                    torch.linspace(0,1,Data.shape[1]),
                                    z_ts_print[:,plot_i],
                                    c='green', label='model_t'+str(i))
                                
                                plt.scatter(
                                    torch.linspace(0,1,obs_test.shape[1]),
                                    obs_print[:,plot_i],
                                    label='Data_t'+str(i),c='red', alpha=0.5)
                                

                            #plt.legend()
                            plt.savefig(os.path.join(path_to_save_plots,'plot_ts_epoch'+str(i)+'_'+str(j)))
                            

                        if 'calcium_imaging' in args.experiment_name:
                            # Plot the first 20 frames
                            data_to_plot = obs_print[:20,:]*args.scaling_factor #Get the first 10 samples for a test 
                            predicted_to_plot = z_p[:20,:]*args.scaling_factor
                            data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                            predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                            predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                            data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                            fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                            c=0
                            for idx_row in range (2): 
                                for idx_col in range(10):
                                    ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row,idx_col].axis('off')
                                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                    ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                    ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row+1,idx_col].axis('off')
                                    c+=1
                            fig.tight_layout()
                            plt.savefig(os.path.join(path_to_save_plots, 'plot_first20frame_rec'+str(i)))


                            # Plot the last 20 frames  
                            data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                            predicted_to_plot = z_p[-20:,:]*args.scaling_factor
                            data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                            predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                            predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                            data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                            fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                            c=0
                            for idx_row in range (2): 
                                for idx_col in range(10):
                                    ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row,idx_col].axis('off')
                                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                    ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                    ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                    ax[2*idx_row+1,idx_col].axis('off')
                                    c+=1
                            fig.tight_layout()
                            plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_rec'+str(i)))


                            #Plot the R2 and MSE loss between the original data and the predicted overtime. 
                            data_to_plot = obs_print[:,:]*args.scaling_factor #Get the first 10 samples for a test 
                            predicted_to_plot = z_p[:,:]*args.scaling_factor
                            data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                            predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                            all_r2_scores = []
                            all_mse_scores = []

                            for idx_frames in range(len(data_to_plot)):
                                _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                                all_r2_scores.append(r_value)
                                # print('data_to_plot[idx_frames,:].flatten().shape: ',data_to_plot[idx_frames,:].flatten().shape)
                                # print('predicted_to_plot[idx_frames,:].flatten().shape: ',predicted_to_plot[idx_frames,:].flatten().shape)
                                tmp_mse_loss = mean_squared_error(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                                all_mse_scores.append(tmp_mse_loss)

                            fig,ax = plt.subplots(2,1, figsize=(15,5), sharex=True, facecolor='w')
                            ax[0].plot(np.arange(len(all_r2_scores)),all_r2_scores)
                            ax[1].plot(np.arange(len(all_mse_scores)),all_mse_scores)
                            ax[1].set_xlabel("Frames")
                            ax[0].set_ylabel("R2")
                            ax[1].set_ylabel("MSE")
                            fig.tight_layout()
                            plt.savefig(os.path.join(path_to_save_plots, 'plot_performance_rec'+str(i)))

                            #Plot integral and ode part separated
                            if ode_func is not None and F_func is not None:
                                Trained_Data_ode = odeint(ode_func,torch.Tensor(obs_print[0,:]).flatten().to(args.device),times.to(args.device),rtol=1e-4,atol=1e-4)
                                Trained_Data_ode_print = to_np(Trained_Data_ode)
                                Trained_Data_integral_print  = z_p - Trained_Data_ode_print
                                # print('Trained_Data_integral_print.max():',np.abs(Trained_Data_integral_print).max())
                                # print('Trained_Data_ode_print.max():',np.abs(Trained_Data_ode_print).max())

                                data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                                predicted_to_plot_ode = Trained_Data_ode_print[-20:,:]*args.scaling_factor
                                predicted_to_plot_ide = Trained_Data_integral_print[-20:,:]*args.scaling_factor
                                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                                predicted_to_plot_ode = args.fitted_pca.inverse_transform(predicted_to_plot_ode)
                                predicted_to_plot_ide = args.fitted_pca.inverse_transform(predicted_to_plot_ide)

                                predicted_to_plot_ode = predicted_to_plot_ode.reshape(predicted_to_plot_ode.shape[0],184, 208) # Add the original frame dimesion as input
                                predicted_to_plot_ide = predicted_to_plot_ide.reshape(predicted_to_plot_ide.shape[0],184, 208)
                                data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                                fig,ax = plt.subplots(6,10, figsize=(15,8), facecolor='w')
                                c=0
                                step = 0
                                for idx_row in range (2): 
                                    for idx_col in range(10):
                                        ax[2*idx_row+step,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row+step,idx_col].axis('off')

                                        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ode[c,:].flatten())
                                        ax[2*idx_row+1+step,idx_col].set_title('ODE R2: {:.3f}'.format(r_value**2))
                                        ax[2*idx_row+1+step,idx_col].imshow(predicted_to_plot_ode[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row+1+step,idx_col].axis('off')

                                        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ide[c,:].flatten())
                                        ax[2*idx_row+2+step,idx_col].set_title('IDE R2: {:.3f}'.format(r_value**2))
                                        ax[2*idx_row+2+step,idx_col].imshow(predicted_to_plot_ide[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row+2+step,idx_col].axis('off')
                                        c+=1
                                    step += 1
                                fig.tight_layout()
                                plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_ode_vs_ide_rec'+str(i)))

                                del data_to_plot, predicted_to_plot
                                del z_to_print, time_to_print, obs_to_print
                        
                        del obs_test, ts_test, z_test, z_p

                        plt.close('all')

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            if split_size>0:
                if Encoder is None:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, None, None, None)
                else:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, None, model, Encoder, Decoder)
            else: 
                if Encoder is None:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, model, None, None, None)
                else:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, None, model, Encoder, Decoder)

            #lr_scheduler(loss_validation)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break

        if args.support_tensors is True or args.support_test is True:
                del dummy_times
                
        end = time.time()
        # print(f"Training time: {(end-start)/60:.3f} minutes")
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),Loss_print)
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),Val_Loss)
        # # plt.savefig('trained.png')
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'trained'+timestr))
        # # plt.show()
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'final_losses'+timestr))
        # # plt.show()
    elif args.mode=='evaluate':
        print('Running in evaluation mode')
        ## Validating
        model.eval()
        if Encoder is not None:
            Encoder.eval()
        if Decoder is not None:
            Decoder.eval()
        
        t_min , t_max = args.time_interval
        n_points = args.test_points

        
        test_times=torch.sort(torch.rand(n_points),0)[0].to(device)*(t_max-t_min)+t_min
        #test_times=torch.linspace(t_min,t_max,n_points)
        
        #dummy_times = torch.cat([torch.Tensor([0.]).to(device),dummy_times])
        # print('times :',times)
        ###########################################################
        
        with torch.no_grad():
            splitting_size = int(args.training_split*Data.size(0))
            all_r2_scores = []
            all_mse = []
            
            for j in tqdm(range(0,Data.shape[0],args.n_batch)):
                
                Dataset_all = Dynamics_Dataset(obs[j:j+args.n_batch,:,:],times,args.n_batch)

                loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = args.n_batch)

                obs_test, ts_test, ids_test = Dataset_all.__getitem__(index_np)#next(iter(loader_test))

                ids_test, indices = torch.sort(torch.from_numpy(ids_test))
                # print('indices: ',indices)
                if args.n_batch==1:
                    obs_test = obs_test[indices,:]
                else:
                    obs_test = obs_test[:,indices,:]
                ts_test = ts_test[indices]
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)


                obs_test = obs_test.to(args.device)
                ts_test = ts_test.to(args.device)
                ids_test = ids_test.to(args.device)
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)
                # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                if args.n_batch ==1:
                    c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_test[:,:1])
                    interpolation = NaturalCubicSpline(c_coeffs)
                    c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,args.time_points).unsqueeze(-1)
                else:
                    if Encoder is None:
                        c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_test[:,:,:1])
                        interpolation = NaturalCubicSpline(c_coeffs)
                        c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,1,args.time_points).unsqueeze(-1)
                    else:
#                         c = lambda x: Encoder(obs_test[:,:,:1].repeat(1,1,args.time_points)\
#                                     .permute(0,2,1).requires_grad_(True))\
#                                     .permute(0,2,1).unsqueeze(-1).contiguous()
#                         y_0 = Encoder(obs_test[:,:,:1].repeat(1,1,args.time_points).permute(0,2,1))\
#                                     .permute(0,2,1).unsqueeze(-1)[:,:,:1,:]
                      c = lambda x: Encoder(obs_test[:,:,:1].permute(0,2,1).requires_grad_(True))\
                                    .permute(0,2,1).unsqueeze(-2).repeat(1,1,args.time_points,1)\
                                    .contiguous().to(args.device)
                      y_0 = Encoder(obs_test[:,:,:1].permute(0,2,1).requires_grad_(True))\
                            .permute(0,2,1).unsqueeze(-2)
        
                if Encoder is None:
                    y_0 = obs_test[:,:,:1].unsqueeze(-1) 
                    
                if args.ts_integration is None:
                    times_integration = torch.linspace(0,1,args.time_points).to(args.device)
                else:
                    times_integration = args.ts_integration.to(args.device)

                
                if args.n_batch ==1:
                    z_test = Integral_spatial_attention_solver(
                            torch.linspace(0,1,args.time_points).to(device),
                            obs_test[0].unsqueeze(1).to(args.device),
                            c=c,
                            sampling_points = args.time_points,
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            spatial_integration=True,
                            spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                            spatial_domain_dim=1,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            smoothing_factor=args.smoothing_factor,
                            use_support=False,
                            ).solve()
                else:
                    z_test = Integral_spatial_attention_solver_multbatch(
                                times_integration,
                                y_0.to(args.device),
                                c=c,
                                sampling_points = args.time_points,
                                mask=mask,
                                Encoder = model,
                                max_iterations = args.max_iterations,
                                spatial_integration=True,
                                spatial_domain= torch.meshgrid(\
                                            [torch.linspace(0,1,args.n_points) for i in range(1)])[0]\
                                            .unsqueeze(-1).to(device),
                                spatial_domain_dim=1,
                                #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                smoothing_factor=args.smoothing_factor,
                                use_support=args.support_tensors,
                                ).solve()
                

                #print('Parameters are:',ide_trained.parameters)
                #print(list(All_parameters))
                if Decoder is not None:
                    z_test = Decoder(z_test)
                else:
                    z_test = z_test.view(args.n_batch,Data.shape[1],args.time_points)
                    
                
                new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                #plt.figure(figsize=(8,8),facecolor='w')

                z_p = z_test
                z_p = to_np(z_p)

                
                obs_print = to_np(obs_test[:,:,:])

                if args.plot_eval is True:
                    for i in range(args.n_batch):
                        if args.plot_as_image is False:
                            plt.figure(j+i, facecolor='w')

                            plt.plot(torch.linspace(0,1,Data.shape[1]),z_p[i,:,0],c='green', label='model_t0',linewidth=3)
                            #plt.scatter(torch.linspace(0,1,args.n_points),z_p[:,0],c='green',s=10)
                            plt.plot(torch.linspace(0,1,Data.shape[1]),z_p[i,:,-1],c='orange', label='model_t1',linewidth=3)
                            #plt.scatter(torch.linspace(0,1,args.n_points),z_p[:,1],c='orange',s=10)


                            plt.scatter(torch.linspace(0,1,obs_test.shape[1]),obs_print[i,:,0],label='Data_t0',c='red', alpha=0.5)
                            plt.scatter(torch.linspace(0,1,obs_test.shape[1]),obs_print[i,:,-1],label='Data_t1',c='blue', alpha=0.5)


                            plt.legend()
                            plt.show()

                        else:
                            for in_batch_indx in range(args.n_batch):

                                obs_print = to_np(obs_test.permute(1,2,0))
                                z_p = to_np(z_test.permute(1,2,0))

                                #plot_reconstruction(obs_print, z_p, None, path_to_save_plots, 'plot_epoch_', i, args)
                                plot_reconstruction(obs_print, z_p, None, None, None, None, args)

                                plt.close('all')

                                del z_p, obs_print
                
                
                _, _, r_value, _, _ = scipy.stats.linregress(z_p[:,:,:].flatten(),obs_print[:,:,:].flatten())
                mse_value = mean_squared_error(z_p[:,:,:].flatten(),obs_print[:,:,:].flatten())

                print('R2:',r_value)
                print('MSE:',mse_value)

                all_r2_scores.append(r_value)
                all_mse.append(mse_value)

                del z_test, z_p, obs_test, obs_print
            
            print("Average R2:",sum(all_r2_scores)/len(all_r2_scores))
            print("Average MSE:",sum(all_mse)/len(all_mse))
                

                
                
def Full_experiment_AttentionalIE_PDE_Navier_Stokes(model, Encoder, Decoder, Data, time_seq, index_np, mask, times, args, extrapolation_points): # experiment_name, plot_freq=1):
    # scaling_factor=1
    
    
    #metadata for saving checkpoints
    if args.model=='nie': 
        str_model_name = "nie"
    elif args.model=='node': 
        str_model_name = "node"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)

    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else: 
            num_experiments = [int(i[3:]) for i in txt]
            num_experiments = np.array(num_experiments).max()
         # -- logger location
        #writer = SummaryWriter(os.path.join(path_to_experiment,'run'+str(num_experiments+1)))
        #print('writer.log_dir: ',writer.log_dir)
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)
            
        #with open(os.path.join(writer.log_dir,'commandline_args.txt'), 'w') as f:
        #    for key, value in args.__dict__.items(): 
        #        f.write('%s:%s\n' % (key, value))



    obs = Data
    times = time_seq
    
    
    if Encoder is None and Decoder is None:
        All_parameters = model.parameters()
    elif Encoder is not None and Decoder is None:
        All_parameters = list(model.parameters())+list(Encoder.parameters())
    elif Decoder is not None and Encoder is None:
        All_parameters = list(model.parameters())+list(Decoder.parameters())
    else:
        All_parameters = list(model.parameters())+list(Encoder.parameters())+list(Decoder.parameters())
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0,last_epoch=-1)# Emanuele's version
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,last_epoch=-1) #My first version
    #scheduler = LRScheduler(optimizer,patience = 20,min_lr=1e-12,factor=0.1)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0,last_epoch=-1)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        if Encoder is None or Decoder is None:
            model, optimizer, scheduler, pos_enc, pos_dec, f_func = load_checkpoint(path, model, optimizer, scheduler, None, None,  None)
        else:
            G_NN, optimizer, scheduler, model, Encoder, Decoder = load_checkpoint(path, None, optimizer, scheduler, model, Encoder, Decoder)

    
    if args.eqn_type=='Navier-Stokes':
        spatial_domain_xy = torch.meshgrid([torch.linspace(0,1,args.n_points) for i in range(2)])
        
        x_space = spatial_domain_xy[0].flatten().unsqueeze(-1)
        y_space = spatial_domain_xy[1].flatten().unsqueeze(-1)
        
        spatial_domain = torch.cat([x_space,y_space],-1)
    
    
    if args.mode=='train':
        #lr_scheduler = LRScheduler(optimizer,patience = 50,min_lr=1e-5,factor=0.1)
        early_stopping = EarlyStopping(patience=1000,min_delta=0)

        # Loss_print = []
        # Val_Loss = []
        all_train_loss=[]
        all_val_loss=[]
        
            
        Data_splitting_indices = Train_val_split(np.copy(index_np),0)
        Train_Data_indices = Data_splitting_indices.train_IDs()
        Val_Data_indices = Data_splitting_indices.val_IDs()
        print('\nlen(Train_Data_indices): ',len(Train_Data_indices))
        print('Train_Data_indices: ',Train_Data_indices)
        print('\nlen(Val_Data_indices): ',len(Val_Data_indices))
        print('Val_Data_indices: ',Val_Data_indices)
        
        # Train Neural IDE
        get_times = Select_times_function(times,extrapolation_points)

        save_best_model = SaveBestModel()
        start = time.time()
        
        split_size = int(args.training_split*obs.size(0))
        
        if args.eqn_type == 'Burgers':
            obs_train = obs[:obs.size(0)-split_size,:,:]
        else:
            obs_train = obs[:obs.size(0)-split_size,:,:,:]
            
        for i in range(args.epochs):
            
            if args.support_tensors is True or args.support_test is True:
                if args.combine_points is True:
                    sampled_tensors,real_idx = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(0, 1))
                    temp_sampled_tensors = sampled_tensors
                    sampled_tensors = sampled_tensors.to(device)
                    #Check if there are duplicates and resample if there are
                    sampled_tensors = torch.cat([times,sampled_tensors])
                    dup=np.array([0])
                    while dup.size != 0:
                        u, c = np.unique(temp_sampled_tensors, return_counts=True)
                        dup = u[c > 1]
                        if dup.size != 0:
                            sampled_tensors,real_idx = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(0, 1))
                            sampled_tensors = sampled_tensors.to(device)
                            sampled_tensors = torch.cat([times,sampled_tensors])
                    dummy_times=sampled_tensors
                    real_idx=real_idx[:times.size(0)]
                if args.combine_points is False:
                        dummy_times = torch.linspace(times[0],times[-1],args.sampling_points)
            
            model.train()
            if Encoder is not None:
                Encoder.train()
            if Decoder is not None:
                Decoder.train()
            
            start_i = time.time()
            print('Epoch:',i)
            # GPUtil.showUtilization()
            counter=0
            train_loss = 0.0
            
            if args.n_batch>1:
                if args.eqn_type == 'Burgers':
                    obs_shuffle = obs_train[torch.randperm(obs_train.size(0)),:,:]
                else:
                    obs_shuffle = obs_train[torch.randperm(obs_train.size(0)),:,:,:]
                
            for j in tqdm(range(0,obs.size(0)-split_size,args.n_batch)):
                
                if args.n_batch==1:
                    if args.eqn_type == 'Burgers':
                        Dataset_train = Dynamics_Dataset(obs_train[j,:,:],times)
                    else:
                        Dataset_train = Dynamics_Dataset(obs_train[j,:,:,:],times)
                else:
                    if args.eqn_type == 'Burgers':
                        Dataset_train = Dynamics_Dataset(obs_shuffle[j:j+args.n_batch,:,:],times,args.n_batch)
                    else:
                        Dataset_train = Dynamics_Dataset(obs_shuffle[j:j+args.n_batch,:,:,:],times,args.n_batch)
                #Dataset_val = Dynamics_Dataset(obs[j-split_size,:,:],times)
                # loader = torch.utils.data.DataLoader(Dataset, batch_size = batch_size)
                # Dataset_val = Val_Dynamics_Dataset(Data,Val_Data_indices,times)

                # For the sampler
                train_sampler = SubsetRandomSampler(Train_Data_indices)
                #valid_sampler = SubsetRandomSampler(Val_Data_indices)

                # loader_val = torch.utils.data.DataLoader(Dataset, batch_size = args.batch_size)

                dataloaders = {'train': torch.utils.data.DataLoader(Dataset_train, sampler=train_sampler,\
                                                                    batch_size = args.n_batch, drop_last=True),
                              }

                train_loader = dataloaders['train']
                #val_loader = dataloaders['val']
                #loader_test = dataloaders['test']

            #for obs_, ts_, ids_ in tqdm(train_loader): 
                obs_, ts_, ids_ = Dataset_train.__getitem__(index_np)#next(iter(train_loader))
                
                obs_ = obs_.to(args.device)
                ts_ = ts_.to(args.device)
                ids_ = torch.from_numpy(ids_).to(args.device)
                # obs_, ts_, ids_ = next(iter(loader))

                ids_, indices = torch.sort(ids_)
                ts_ = ts_[indices]
                ts_ = torch.cat([times[:1],ts_])
                if args.n_batch==1:
                    if args.eqn_type == 'Burgers':
                        obs_ = obs_[indices,:]
                    else:
                        if Encoder is None:
                            obs_ = obs_[indices,:,:]
                            obs_ = obs_[:,indices,:]
                else:
                    if args.eqn_type == 'Burgers':
                        obs_ = obs_[:,indices,:]
                    else:
                        if Encoder is None:
                            obs_ = obs_[:,indices,:,:]
                            obs_ = obs_[:,:,indices,:]
                            
                    
                if args.perturbation_to_obs0 is not None:
                       perturb = torch.normal(mean=torch.zeros(obs_.shape[1]).to(args.device),
                                              std=args.std_noise)#args.perturbation_to_obs0*obs_[:3,:].std(dim=0))
                else:
                    perturb = torch.zeros_like(obs_[0]).to(args.device)
                # print('obs_[:5]: ',obs_[:5])
                # print('ids_[:5]: ',ids_[:5])
                # print('ts_[:5]: ',ts_[:5])

                # print('obs_: ',obs_)
                # print('ids_: ',ids_)
                # print('ts_: ',ts_)

                # obs_, ts_ = obs_.squeeze(1), ts_.squeeze(1)
                if args.initialization is False:
                    y_init = None
                    
                    if args.n_batch==1:
                        if args.eqn_type == 'Burgers':
                            c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_[:,:1])
                            interpolation = NaturalCubicSpline(c_coeffs)
                            c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,args.time_points).unsqueeze(-1)
                        else:
                            c= lambda x: obs_[:,:,:1].repeat(1,1,args.time_points).unsqueeze(-1).to(device)
                    else:
                        if args.eqn_type == 'Burgers':
                            c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_[:,:,:1])
                            interpolation = NaturalCubicSpline(c_coeffs)
                            c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,1,args.time_points).unsqueeze(-1)
                        else:
    #                         c= lambda x: \
    #                         Encoder(obs_[:,:,:,:1].repeat(1,1,1,args.time_points)\
    #                         .permute(0,3,1,2).requires_grad_(True)).unsqueeze(-1)\
    #                         .permute(0,2,3,1,4).contiguous().to(args.device)
                            c= lambda x: \
                            Encoder(obs_[:,:,:,:1].permute(0,3,1,2).requires_grad_(True))\
                                    .permute(0,2,3,1).unsqueeze(-2)\
                                    .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)
        
                else:
                    y_init = Encoder(obs_[:,:,:,:1].permute(0,3,1,2).requires_grad_(True))\
                                    .permute(0,2,3,1).unsqueeze(-2)\
                                    .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)
                    c = lambda x: torch.zeros_like(y_init).to(args.device)
                    
                
                #if args.patches is True:
                if args.eqn_type == 'Navier-Stokes':
#                     y_0 = Encoder(obs_[:,:,:,:1].repeat(1,1,1,args.time_points)\
#                         .permute(0,3,1,2)).unsqueeze(-1)\
#                         .permute(0,2,3,1,4)[:,:,:,:1,:]
                    y_0 =  Encoder(obs_[:,:,:,:1].permute(0,3,1,2))\
                                .permute(0,2,3,1).unsqueeze(-2)
                    
                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)
                
                if args.support_tensors is False:
                    if args.n_batch==1:
                        if args.eqn_type == 'Burgers':
                            z_ = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_[0].unsqueeze(1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                    spatial_domain_dim=1,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()
                        else:
                            z_ = Integral_spatial_attention_solver(
                                    times_integration.to(args.device),
                                    obs_[:,:,0].unsqueeze(-1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(device),
                                    spatial_domain_dim=2,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()
                    else:
                        if args.eqn_type == 'Burgers':
                            z_ = Integral_spatial_attention_solver_multbatch(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_[:,0].unsqueeze(-1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                    spatial_domain_dim=1,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()
                        else:
                            z_ = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    y_init=y_init,
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(args.device),
                                    spatial_domain_dim=2,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=True,
                                    initialization=args.initialization
                                    ).solve()
                else:
                    z_ = Integral_spatial_attention_solver(
                            torch.linspace(0,1,args.time_points).to(device),
                            obs_[0].unsqueeze(0).to(args.device),
                            c=c,
                            sampling_points = args.time_points,
                            support_tensors=dummy_times.to(device),
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            spatial_integration=True,
                            spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                            spatial_domain_dim=1,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            smoothing_factor=args.smoothing_factor,
                            output_support_tensors=True
                            ).solve()
                    if args.combine_points is True:
                        z_ = z_[real_idx,:]
                
                
                if args.eqn_type=='Burgers':
                    if args.n_batch==1:
                            z_ = z_.view(args.n_points,args.time_points)
                            z_ = torch.cat([z_[:,:1],z_[:,-1:]],-1)
                    else:
                            z_ = z_.view(args.n_batch,args.n_points,args.time_points)
                            z_ = torch.cat([z_[:,:,:1],z_[:,:,-1:]],-1)
                else:
                    if args.n_batch==1:
                        z_ = z_.view(args.n_points,args.n_points,args.time_points)
                    else:
                        z_ = z_.view(z_.shape[0],args.n_points,args.n_points,args.time_points,args.dim)
                    
                    if Decoder is not None:
#                         z_ = z_.squeeze(-1).permute(0,3,1,2)
#                         z_ = Decoder(z_.requires_grad_(True)).permute(0,2,3,1)
                        z_ = Decoder(z_.requires_grad_(True))
                    else:
                        z_ = z_.view(args.n_batch,Data.shape[1],Data.shape[2],args.time_points)
                    if args.initial_t is False:
                        obs_ = obs_[:,:,:,1:]
                     
                #loss_ts_ = get_times.select_times(ts_)[1]
                loss = F.mse_loss(z_, obs_.detach()) #Original 
                # print('z_[:,:].to(args.device): ',z_[:,:].to(args.device))
                # print('obs_.to(args.device).detach()[:,:]: ',obs_.to(args.device).detach()[:,:])
                # loss = F.mse_loss(z_[:,:].to(args.device), obs_.to(args.device).detach()[:,:]) #Original 

                
                # ###############################
                # Loss_print.append(to_np(loss))
                # ###############################

                optimizer.zero_grad()
                loss.backward()#(retain_graph=True)
                optimizer.step()

                # n_iter += 1
                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  split_size==0 and args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, obs_, ts_, z_, ids_

            ## Validating
                
            model.eval()
            if Encoder is not None:
                Encoder.eval()
            if Decoder is not None:
                Decoder.eval()
                
            with torch.no_grad():

                    #Only do this if there is a validation dataset
                
                val_loss = 0.0
                counter = 0
                if split_size>0:
                    # for images, _, _, _, _ in tqdm(val_loader):   # frames, timevals, angular_velocity, mass_height, mass_xpos
                    for j in tqdm(range(obs.size(0)-split_size,obs.size(0),args.n_batch)):
                        
                        valid_sampler = SubsetRandomSampler(Train_Data_indices)
                        if args.n_batch==1:
                            if args.eqn_type == 'Burgers':
                                Dataset_val = Dynamics_Dataset(obs[j,:,:],times)
                            else:
                                Dataset_val = Dynamics_Dataset(obs[j,:,:,:],times)
                        else:
                            if args.eqn_type == 'Burgers':
                                Dataset_val = Dynamics_Dataset(obs[j:j+args.n_batch,:,:],times,args.n_batch)
                            else:
                                Dataset_val = Dynamics_Dataset(obs[j:j+args.n_batch,:,:,:],times,args.n_batch)
                        
                        val_loader = torch.utils.data.DataLoader(Dataset_val, sampler=valid_sampler,\
                                                                 batch_size = args.n_batch, drop_last=True)
                    
                    #for obs_val, ts_val, ids_val in tqdm(val_loader):
                        obs_val, ts_val, ids_val = Dataset_val.__getitem__(index_np)#next(iter(val_loader))
                        obs_val = obs_val.to(args.device)
                        ts_val = ts_val.to(args.device)
                        
                        ids_val = torch.from_numpy(ids_val).to(args.device)

                        ids_val, indices = torch.sort(ids_val)
                        # print('indices: ',indices)
                        if args.n_batch ==1:
                            if args.eqn_type == 'Burgers':
                                obs_val = obs_val[indices,:]
                            else:
                                if Encoder is None:
                                    obs_val = obs_val[indices,:,:]
                                    obs_val = obs_val[:,indices,:]
                        else:
                            if args.eqn_type == 'Burgers':
                                obs_val = obs_val[:,indices,:]
                            else:
                                if Encoder is None:
                                    obs_val = obs_val[:,indices,:,:]
                                    obs_val = obs_val[:,:,indices,:]
                        
                        ts_val = ts_val[indices]
                                             

                        #Concatenate the first point of the train minibatch
                        # obs_[0],ts_
                        # print('\n In validation mode...')
                        # print('obs_[:5]: ',obs_[:5])
                        # print('ids_[:5]: ',ids_[:5])
                        # print('ts_[:5]: ',ts_[:5])
                        # print('ts_[0]:',ts_[0])

                        ## Below is to add initial data point to val
                        #obs_val = torch.cat((obs_[0][None,:],obs_val))
                        #ts_val = torch.hstack((ts_[0],ts_val))
                        #ids_val = torch.hstack((ids_[0],ids_val))

                        # obs_val, ts_val, ids_val = next(iter(loader_val))
                        # print('obs_val.shape: ',obs_val.shape)
                        # print('ids_val: ',ids_val)
                        # print('ts_val: ',ts_val)

                        # obs_val, ts_val = obs_val.squeeze(1), ts_val.squeeze(1)
                        if args.initialization is False:
                            y_init=None
                            if args.n_batch==1:
                                if args.eqn_type == 'Burgers':
                                    c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_val[:,:1])
                                    interpolation = NaturalCubicSpline(c_coeffs)
                                    c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,args.time_points).unsqueeze(-1)
                                else:
                                    c= lambda x: obs_val[:,:,:1].repeat(1,1,args.time_points).unsqueeze(-1).to(device)
                            else:
                                if args.eqn_type == 'Burgers':
                                    c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_val[:,:,:1])
                                    interpolation = NaturalCubicSpline(c_coeffs)
                                    c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,1,args.time_points).unsqueeze(-1)
                                else:
    #                                 c= lambda x: \
    #                                             Encoder(obs_val[:,:,:,:1].repeat(1,1,1,args.time_points)\
    #                                             .permute(0,3,1,2)).unsqueeze(-1)\
    #                                             .permute(0,2,3,1,4).contiguous().to(args.device)
                                    c= lambda x: \
                                        Encoder(obs_val[:,:,:,:1].permute(0,3,1,2))\
                                                .permute(0,2,3,1).unsqueeze(-2)\
                                                .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)
                        else:
                            y_init = Encoder(obs_val[:,:,:,:1].permute(0,3,1,2))\
                                                .permute(0,2,3,1).unsqueeze(-2)\
                                                .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)
                            c = lambda x: torch.zeros_like(y_init).to(args.device)
                        
                        if args.eqn_type == 'Navier-Stokes':
#                             y_0 = Encoder(obs_val[:,:,:,:1].repeat(1,1,1,args.time_points)\
#                             .permute(0,3,1,2)).unsqueeze(-1)\
#                             .permute(0,2,3,1,4)[:,:,:,:1,:]
                            y_0 = Encoder(obs_val[:,:,:,:1].permute(0,3,1,2))\
                                            .permute(0,2,3,1).unsqueeze(-2)\
                                            .to(args.device)
                            
                            
                        if args.ts_integration is not None:
                            times_integration = args.ts_integration
                        else:
                            times_integration = torch.linspace(0,1,args.time_points)
                    
                        if args.support_tensors is False:
                            if args.n_batch==1:
                                if args.eqn_type == 'Burgers':
                                    z_val = Integral_spatial_attention_solver(
                                            torch.linspace(0,1,args.time_points).to(device),
                                            obs_val[0].unsqueeze(1).to(args.device),
                                            c=c,
                                            sampling_points = args.time_points,
                                            mask=mask,
                                            Encoder = model,
                                            max_iterations = args.max_iterations,
                                            spatial_integration=True,
                                            spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                            spatial_domain_dim=1,
                                            smoothing_factor=args.smoothing_factor,
                                            use_support=False,
                                            ).solve()
                                else:
                                    z_val = Integral_spatial_attention_solver(
                                            torch.linspace(0,1,args.time_points).to(device),
                                            obs_val[:,:,0].unsqueeze(-1).to(args.device),
                                            c=c,
                                            sampling_points = args.time_points,
                                            mask=mask,
                                            Encoder = model,
                                            max_iterations = args.max_iterations,
                                            spatial_integration=True,
                                            spatial_domain= spatial_domain.to(device),
                                            spatial_domain_dim=2,
                                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                            smoothing_factor=args.smoothing_factor,
                                            use_support=False,
                                            ).solve()
                                    
                            else:
                                if args.eqn_type == 'Burgers':
                                    z_val = Integral_spatial_attention_solver_multbatch(
                                        torch.linspace(0,1,args.time_points).to(device),
                                        obs_val[:,0].unsqueeze(-1).to(args.device),
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                        spatial_domain_dim=1,
                                        #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                        #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        ).solve()
                                else:
                                    z_val = Integral_spatial_attention_solver_multbatch(
                                            times_integration.to(args.device),
                                            y_0.to(args.device),
                                            y_init=y_init,
                                            c=c,
                                            sampling_points = args.time_points,
                                            mask=mask,
                                            Encoder = model,
                                            max_iterations = args.max_iterations,
                                            spatial_integration=True,
                                            spatial_domain= spatial_domain.to(args.device),
                                            spatial_domain_dim=2,
                                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                            smoothing_factor=args.smoothing_factor,
                                            use_support=False,
                                            initialization=args.initialization
                                            ).solve()
                                
                        else:
                            z_val = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_[0].unsqueeze(1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    support_tensors=dummy_times.to(device),
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                    spatial_domain_dim=1,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    output_support_tensors=True
                                    ).solve()
                        
                            if args.combine_points is True:
                                z_val = z_val[real_idx,:]
                          
                        if args.eqn_type=='Burgers':
                            if args.n_batch==1:
                                    z_val = z_val.view(args.n_points,args.time_points)
                                    z_val = torch.cat([z_val[:,:1],z_val[:,-1:]],-1)
                            else:
                                    z_val = z_val.view(args.n_batch,args.n_points,args.time_points)
                                    z_val = torch.cat([z_val[:,:,:1],z_val[:,:,-1:]],-1)
                        else:
                            if args.n_batch==1:
                                z_val = z_val.view(args.n_points,args.n_points,args.time_points)
                            else:
                                z_val = z_val.view(z_val.shape[0],args.n_points,args.n_points,args.time_points,args.dim)
                            
                            if Decoder is not None:
#                                 z_val = z_val.squeeze(-1).permute(0,3,1,2)
#                                 z_val = Decoder(z_val).permute(0,2,3,1)
                                z_val = Decoder(z_val)
                            else:
                                z_val = z_val.view(args.n_batch,Data.shape[1],Data.shape[2],args.time_points)
                            if args.initial_t is False:
                                obs_val = obs_val[:,:,:,1:]
                            
                        #validation_ts_ = get_times.select_times(ts_val)[1]
                        loss_validation = F.mse_loss(z_val, obs_val.detach())
                        # Val_Loss.append(to_np(loss_validation))
                        
                        del obs_val, ts_val, z_val, ids_val

                        counter += 1
                        val_loss += loss_validation.item()
                        
                        del loss_validation

                        #LRScheduler(loss_validation)
                        if args.lr_scheduler == 'ReduceLROnPlateau':
                            scheduler.step(val_loss)
                
                
                else: counter += 1

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            #writer.add_scalar('train_loss', all_train_loss[-1], global_step=i)
            #if len(all_val_loss)>0:
            #    writer.add_scalar('val_loss', all_val_loss[-1], global_step=i)
            #if args.lr_scheduler == 'ReduceLROnPlateau':
            #    writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], global_step=i)
            #elif args.lr_scheduler == 'CosineAnnealingLR':
            #    writer.add_scalar('Epoch/learning_rate', scheduler.get_last_lr()[0], global_step=i)

            
            with torch.no_grad():
                
                model.eval()
                if Encoder is not None:
                    Encoder.eval()
                if Decoder is not None:
                    Decoder.eval()
                
                if i % args.plot_freq == 0 and i != 0:
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
                    # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
                        
                    plt.plot(np.log10(all_train_loss),label='Train loss')
                    if split_size>0:
                        plt.plot(np.log10(all_val_loss),label='Val loss')
                    plt.xlabel("Epoch")
                    plt.ylabel("MSE Loss")
                    # timestr = time.strftime("%Y%m%d-%H%M%S")
                    #plt.show()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))

                    #for j in tqdm(range(0,obs.size(0),args.n_batch)):
                    for j in tqdm(range(1)):
                        if args.n_batch==1:
                            if args.eqn_type == 'Burgers':
                                Dataset_all = Dynamics_Dataset(Data[j,:,:],times)
                            else:
                                Dataset_all = Dynamics_Dataset(Data[j,:,:,:],times)
                        else:
                            if args.eqn_type == 'Burgers':
                                Dataset_all = Dynamics_Dataset(obs[j:j+args.n_batch,:,:],times,args.n_batch)
                            else:
                                Dataset_all = Dynamics_Dataset(obs[j:j+args.n_batch,:,:,:],times,args.n_batch)
                                
                        loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = args.n_batch)

                        obs_test, ts_test, ids_test = Dataset_all.__getitem__(index_np)#next(iter(loader_test))

                        ids_test, indices = torch.sort(torch.from_numpy(ids_test))
                        # print('indices: ',indices)
                        if args.n_batch==1:
                            if args.eqn_type == 'Burgers':
                                obs_test = obs_test[indices,:]
                            else:
                                if Encoder is None:
                                    obs_test = obs_test[indices,:,:]
                                    obs_test = obs_test[:,indices,:]
                        else:
                            if args.eqn_type == 'Burgers':
                                obs_test = obs_test[:,indices,:]
                            else:
                                if Encoder is None:
                                    obs_test = obs_test[:,indices,:,:]
                                    obs_test = obs_test[:,:,indices,:]
                        ts_test = ts_test[indices]
                        # print('obs_test.shape: ',obs_test.shape)
                        # print('ids_test: ',ids_test)
                        # print('ts_test: ',ts_test)


                        obs_test = obs_test.to(args.device)
                        ts_test = ts_test.to(args.device)
                        ids_test = ids_test.to(args.device)
                        # print('obs_test.shape: ',obs_test.shape)
                        # print('ids_test: ',ids_test)
                        # print('ts_test: ',ts_test)
                        # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                        if args.initialization is False:
                            y_init=None
                            if args.n_batch ==1:
                                if args.eqn_type == 'Burgers':
                                    c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_test[:,:1])
                                    interpolation = NaturalCubicSpline(c_coeffs)
                                    c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,args.time_points).unsqueeze(-1)
                                else:
                                    c = lambda x: obs_test[:,:,:1].repeat(1,1,args.time_points).unsqueeze(-1).to(device)
                            else:
                                if args.eqn_type == 'Burgers':
                                    c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_test[:,:,:1])
                                    interpolation = NaturalCubicSpline(c_coeffs)
                                    c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,1,args.time_points).unsqueeze(-1)
                                else:
    #                                 c= lambda x: \
    #                                             Encoder(obs_test[:,:,:,:1].repeat(1,1,1,args.time_points)\
    #                                             .permute(0,3,1,2)).unsqueeze(-1)\
    #                                             .permute(0,2,3,1,4).contiguous().to(args.device)
                                    c= lambda x: Encoder(obs_test[:,:,:,:1].permute(0,3,1,2))\
                                                .permute(0,2,3,1).unsqueeze(-2)\
                                                .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)
                        else:
                            y_init = Encoder(obs_test[:,:,:,:1].permute(0,3,1,2))\
                                                .permute(0,2,3,1).unsqueeze(-2)\
                                                .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)
                            c = lambda x: torch.zeros_like(y_init).to(args.device)
                        if args.eqn_type == 'Navier-Stokes':
#                             y_0 = Encoder(obs_test[:,:,:,:1].repeat(1,1,1,args.time_points)\
#                             .permute(0,3,1,2)).unsqueeze(-1)\
#                             .permute(0,2,3,1,4)[:,:,:,:1,:]
                            y_0 = Encoder(obs_test[:,:,:,:1].permute(0,3,1,2))\
                                            .permute(0,2,3,1).unsqueeze(-2)\
                                            .to(args.device)
                            
                        if args.ts_integration is not None:
                            times_integration = args.ts_integration
                        else:
                            times_integration = torch.linspace(0,1,args.time_points)
                                  
                        if args.support_test is False:
                            if args.n_batch==1:
                                if args.eqn_type == 'Burgers':
                                    z_test = Integral_spatial_attention_solver(
                                            torch.linspace(0,1,args.time_points).to(device),
                                            obs_test[0].unsqueeze(1).to(args.device),
                                            c=c,
                                            sampling_points = args.time_points,
                                            mask=mask,
                                            Encoder = model,
                                            max_iterations = args.max_iterations,
                                            spatial_integration=True,
                                            spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                            spatial_domain_dim=1,
                                            smoothing_factor=args.smoothing_factor,
                                            use_support=False,
                                            ).solve()
                                else:
                                    z_test = Integral_spatial_attention_solver(
                                            torch.linspace(0,1,args.time_points).to(device),
                                            obs_test[:,:,0].unsqueeze(-1).to(args.device),
                                            c=c,
                                            sampling_points = args.time_points,
                                            mask=mask,
                                            Encoder = model,
                                            max_iterations = args.max_iterations,
                                            spatial_integration=True,
                                            spatial_domain= spatial_domain.to(device),
                                            spatial_domain_dim=2,
                                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                            smoothing_factor=args.smoothing_factor,
                                            use_support=False,
                                            ).solve()
                                    
                            else:
                                if args.eqn_type == 'Burgers':
                                    z_test = Integral_spatial_attention_solver_multbatch(
                                        torch.linspace(0,1,args.time_points).to(device),
                                        obs_test[:,0].unsqueeze(-1).to(args.device),
                                        c=c,
                                        sampling_points = args.time_points,
                                        mask=mask,
                                        Encoder = model,
                                        max_iterations = args.max_iterations,
                                        spatial_integration=True,
                                        spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                        spatial_domain_dim=1,
                                        #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                        #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                        smoothing_factor=args.smoothing_factor,
                                        use_support=False,
                                        ).solve()
                                else:
                                    z_test = Integral_spatial_attention_solver_multbatch(
                                            times_integration.to(args.device),
                                            y_0.to(args.device),
                                            y_init=y_init,
                                            c=c,
                                            sampling_points = args.time_points,
                                            mask=mask,
                                            Encoder = model,
                                            max_iterations = args.max_iterations,
                                            spatial_integration=True,
                                            spatial_domain= spatial_domain.to(args.device),
                                            spatial_domain_dim=2,
                                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                            smoothing_factor=args.smoothing_factor,
                                            use_support=False,
                                            initialization=args.initialization
                                            ).solve()
                        else:
                            z_test = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_[0].unsqueeze(1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    support_tensors=dummy_times.to(device),
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                    spatial_domain_dim=1,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    output_support_tensors=True,
                                    ).solve()
                            
                        #print('Parameters are:',ide_trained.parameters)
                        #print(list(All_parameters))
                        
                        
                        if args.eqn_type == 'Burgers':
                            if args.n_batch== 1:
                                z_test = z_test.view(args.n_points,args.time_points)
                                z_test = torch.cat([z_test[:,:1],z_test[:,-1:]],-1)
                            else:
                                z_test = z_test.view(args.n_batch,args.n_points,args.time_points)
                                z_test = torch.cat([z_test[:,:,:1],z_test[:,:,-1:]],-1)
                            new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                            plt.figure(1,facecolor='w')

                            z_p = z_test
                            if args.n_batch >1:
                                z_p = z_test[0,:,:]
                            z_p = to_np(z_p)

                            if args.n_batch >1:
                                obs_print = to_np(obs_test[0,:,:])
                            else:
                                obs_print = to_np(obs_test[:,:])

#                             if obs.size(2)>2:
#                                 z_p = pca_proj.fit_transform(z_p)
#                                 obs_print = pca_proj.fit_transform(obs_print)                    

                            plt.figure(1, facecolor='w')
                            plt.plot(torch.linspace(0,1,args.n_points),z_p[:,0],c='r', label='model')
                            plt.scatter(torch.linspace(0,1,args.n_points),z_p[:,0],c='r',s=10)


                            # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                            plt.scatter(torch.linspace(0,1,obs_test.size(1)),obs_print[:,0],label='Data',c='blue', alpha=0.5)
                            #plt.xlabel("dim 0")
                            #plt.ylabel("dim 1")
                            #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                            plt.legend()
                            # plt.show()
                            # timestr = time.strftime("%Y%m%d-%H%M%S")
                            plt.savefig(os.path.join(path_to_save_plots,'plot_t0_epoch'+str(i)+'_'+str(j)))

                            plt.figure(2, facecolor='w')

                            plt.plot(torch.linspace(0,1,args.n_points),z_p[:,1], label='model')
                            plt.scatter(torch.linspace(0,1,args.n_points),z_p[:,1],s=10)


                            # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                            plt.scatter(torch.linspace(0,1,obs_test.size(1)),obs_print[:,1],label='Data',c='blue', alpha=0.5)
                            #plt.xlabel("dim 0")
                            #plt.ylabel("dim 1")
                            #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                            plt.legend()
                            # plt.show()
                            # timestr = time.strftime("%Y%m%d-%H%M%S")
                            plt.savefig(os.path.join(path_to_save_plots,'plot_t1_epoch'+str(i)+'_'+str(j)))


                            plt.figure(3, facecolor='w')

                            plt.plot(torch.linspace(0,1,args.n_points),z_p[:,0],c='green', label='model_t0')
                            #plt.scatter(torch.linspace(0,1,args.n_points),z_p[:,0],c='green',s=10)
                            plt.plot(torch.linspace(0,1,args.n_points),z_p[:,1],c='orange', label='model_t1')
                            #plt.scatter(torch.linspace(0,1,args.n_points),z_p[:,1],c='orange',s=10)


                            # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                            plt.scatter(torch.linspace(0,1,obs_test.size(1)),obs_print[:,0],label='Data_t0',c='red', alpha=0.5)
                            plt.scatter(torch.linspace(0,1,obs_test.size(1)),obs_print[:,1],label='Data_t1',c='blue', alpha=0.5)
                            #plt.xlabel("dim 0")
                            #plt.ylabel("dim 1")
                            #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                            plt.legend()
                            # plt.show()
                            # timestr = time.strftime("%Y%m%d-%H%M%S")
                            plt.savefig(os.path.join(path_to_save_plots,'plot_t0t1_epoch'+str(i)+'_'+str(j)))

                            if 'calcium_imaging' in args.experiment_name:
                                # Plot the first 20 frames
                                data_to_plot = obs_print[:20,:]*args.scaling_factor #Get the first 10 samples for a test 
                                predicted_to_plot = z_p[:20,:]*args.scaling_factor
                                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                                predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                                data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                                fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                                c=0
                                for idx_row in range (2): 
                                    for idx_col in range(10):
                                        ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row,idx_col].axis('off')
                                        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                        ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                        ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row+1,idx_col].axis('off')
                                        c+=1
                                fig.tight_layout()
                                plt.savefig(os.path.join(path_to_save_plots, 'plot_first20frame_rec'+str(i)))


                                # Plot the last 20 frames  
                                data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                                predicted_to_plot = z_p[-20:,:]*args.scaling_factor
                                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                                predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                                data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                                fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                                c=0
                                for idx_row in range (2): 
                                    for idx_col in range(10):
                                        ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row,idx_col].axis('off')
                                        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                                        ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                                        ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                        ax[2*idx_row+1,idx_col].axis('off')
                                        c+=1
                                fig.tight_layout()
                                plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_rec'+str(i)))


                                #Plot the R2 and MSE loss between the original data and the predicted overtime. 
                                data_to_plot = obs_print[:,:]*args.scaling_factor #Get the first 10 samples for a test 
                                predicted_to_plot = z_p[:,:]*args.scaling_factor
                                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                                all_r2_scores = []
                                all_mse_scores = []

                                for idx_frames in range(len(data_to_plot)):
                                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                                    all_r2_scores.append(r_value)
                                    # print('data_to_plot[idx_frames,:].flatten().shape: ',data_to_plot[idx_frames,:].flatten().shape)
                                    # print('predicted_to_plot[idx_frames,:].flatten().shape: ',predicted_to_plot[idx_frames,:].flatten().shape)
                                    tmp_mse_loss = mean_squared_error(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                                    all_mse_scores.append(tmp_mse_loss)

                                fig,ax = plt.subplots(2,1, figsize=(15,5), sharex=True, facecolor='w')
                                ax[0].plot(np.arange(len(all_r2_scores)),all_r2_scores)
                                ax[1].plot(np.arange(len(all_mse_scores)),all_mse_scores)
                                ax[1].set_xlabel("Frames")
                                ax[0].set_ylabel("R2")
                                ax[1].set_ylabel("MSE")
                                fig.tight_layout()
                                plt.savefig(os.path.join(path_to_save_plots, 'plot_performance_rec'+str(i)))

                                #Plot integral and ode part separated
                                if ode_func is not None and F_func is not None:
                                    Trained_Data_ode = odeint(ode_func,torch.Tensor(obs_print[0,:]).flatten().to(args.device),times.to(args.device),rtol=1e-4,atol=1e-4)
                                    Trained_Data_ode_print = to_np(Trained_Data_ode)
                                    Trained_Data_integral_print  = z_p - Trained_Data_ode_print
                                    # print('Trained_Data_integral_print.max():',np.abs(Trained_Data_integral_print).max())
                                    # print('Trained_Data_ode_print.max():',np.abs(Trained_Data_ode_print).max())

                                    data_to_plot = obs_print[-20:,:]*args.scaling_factor #Get the first 10 samples for a test 
                                    predicted_to_plot_ode = Trained_Data_ode_print[-20:,:]*args.scaling_factor
                                    predicted_to_plot_ide = Trained_Data_integral_print[-20:,:]*args.scaling_factor
                                    data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                                    predicted_to_plot_ode = args.fitted_pca.inverse_transform(predicted_to_plot_ode)
                                    predicted_to_plot_ide = args.fitted_pca.inverse_transform(predicted_to_plot_ide)

                                    predicted_to_plot_ode = predicted_to_plot_ode.reshape(predicted_to_plot_ode.shape[0],184, 208) # Add the original frame dimesion as input
                                    predicted_to_plot_ide = predicted_to_plot_ide.reshape(predicted_to_plot_ide.shape[0],184, 208)
                                    data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                                    fig,ax = plt.subplots(6,10, figsize=(15,8), facecolor='w')
                                    c=0
                                    step = 0
                                    for idx_row in range (2): 
                                        for idx_col in range(10):
                                            ax[2*idx_row+step,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                            ax[2*idx_row+step,idx_col].axis('off')

                                            _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ode[c,:].flatten())
                                            ax[2*idx_row+1+step,idx_col].set_title('ODE R2: {:.3f}'.format(r_value**2))
                                            ax[2*idx_row+1+step,idx_col].imshow(predicted_to_plot_ode[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                            ax[2*idx_row+1+step,idx_col].axis('off')

                                            _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot_ide[c,:].flatten())
                                            ax[2*idx_row+2+step,idx_col].set_title('IDE R2: {:.3f}'.format(r_value**2))
                                            ax[2*idx_row+2+step,idx_col].imshow(predicted_to_plot_ide[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                                            ax[2*idx_row+2+step,idx_col].axis('off')
                                            c+=1
                                        step += 1
                                    fig.tight_layout()
                                    plt.savefig(os.path.join(path_to_save_plots, 'plot_last20frame_ode_vs_ide_rec'+str(i)))

                                    del data_to_plot, predicted_to_plot
                                    del z_to_print, time_to_print, obs_to_print

                            del obs_test, ts_test, z_test, z_p

                            plt.close('all')
                            
                        else:
                            
                            #z_test = z_test.view(args.n_batch,args.n_points,args.n_points,args.time_points,args.dim)
                            if Decoder is not None:
#                                 z_test = z_test.squeeze(-1).permute(0,3,1,2)
#                                 z_test = Decoder(z_test).permute(0,2,3,1)
                                z_test = Decoder(z_test)
                            else:
                                z_test = z_test.view(z_test.shape[0],Data.shape[1],Data.shape[2],args.time_points)
                            if args.initial_t is False:
                                obs_test = obs_test[:,:,:,1:]
                            
                            z_p = z_test
                            if args.n_batch >1:
                                z_p = z_test[0,:,:,:]
                            z_p = to_np(z_p)

                            if args.n_batch >1:
                                obs_print = to_np(obs_test[0,:,:,:])
                            else:
                                obs_print = to_np(obs_test[:,:,:])

                            plot_reconstruction(obs_print, z_p, None, path_to_save_plots, 'plot_epoch_', i, args)
                            
                            plt.close('all')
                            del z_p, z_test, obs_print
#                             plt.figure(1, facecolor='w')
#                             plt.plot(torch.linspace(0,1,args.time_points),z_p[7,5,:],c='green', label='model_t')
                            
                            
#                             plt.scatter(torch.linspace(0,1,obs_test.size(-1)),obs_print[7,5,:],label='Data_t',c='blue', alpha=0.5)
                            
                            
#                             plt.legend()
                            
#                             plt.savefig(os.path.join(path_to_save_plots,'plot_epoch'+str(i)+'_'+str(j)))
                            
#                             del obs_print, z_p
                            
#                             plt.close('all')

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            if split_size>0:
                if Encoder is None:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, None, None, None)
                else:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, None, model, Encoder, Decoder)
            else: 
                if Encoder is None:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, model, None, None, None)
                else:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, None, model, Encoder, Decoder)

            #lr_scheduler(loss_validation)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break

        if args.support_tensors is True or args.support_test is True:
                del dummy_times
                
        end = time.time()
        # print(f"Training time: {(end-start)/60:.3f} minutes")
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),Loss_print)
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),Val_Loss)
        # # plt.savefig('trained.png')
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'trained'+timestr))
        # # plt.show()
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'final_losses'+timestr))
        # # plt.show()
    elif args.mode=='evaluate':
        print('Running in evaluation mode')
        ## Validating
        model.eval()
        
        t_min , t_max = args.time_interval
        n_points = args.test_points

        
        test_times=torch.sort(torch.rand(n_points),0)[0].to(device)*(t_max-t_min)+t_min
        #test_times=torch.linspace(t_min,t_max,n_points)
        
        #dummy_times = torch.cat([torch.Tensor([0.]).to(device),dummy_times])
        # print('times :',times)
        ###########################################################
        
        with torch.no_grad():
                
            model.eval()
            if Encoder is not None:
                Encoder.eval()
            if Decoder is not None:
                Decoder.eval()
                
            test_loss = 0.0
            loss_list = []
            #counter = 0  

            for j in tqdm(range(0,obs.shape[0],args.n_batch)):
                if args.n_batch==1:
                    if args.eqn_type == 'Burgers':
                        Dataset_all = Dynamics_Dataset(Data[j,:,:],times)
                    else:
                        Dataset_all = Dynamics_Dataset(Data[j,:,:,:],times)
                else:
                    if args.eqn_type == 'Burgers':
                        Dataset_all = Dynamics_Dataset(obs[j:j+args.n_batch,:,:],times,args.n_batch)
                    else:
                        Dataset_all = Dynamics_Dataset(obs[j:j+args.n_batch,:,:,:],times,args.n_batch)

                loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = args.n_batch)

                obs_test, ts_test, ids_test = Dataset_all.__getitem__(index_np)#next(iter(loader_test))

                ids_test, indices = torch.sort(torch.from_numpy(ids_test))
                # print('indices: ',indices)
                if args.n_batch==1:
                    if args.eqn_type == 'Burgers':
                        obs_test = obs_test[indices,:]
                    else:
                        if Encoder is None:
                            obs_test = obs_test[indices,:,:]
                            obs_test = obs_test[:,indices,:]
                else:
                    if args.eqn_type == 'Burgers':
                        obs_test = obs_test[:,indices,:]
                    else:
                        if Encoder is None:
                            obs_test = obs_test[:,indices,:,:]
                            obs_test = obs_test[:,:,indices,:]
                ts_test = ts_test[indices]
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)


                obs_test = obs_test.to(args.device)
                ts_test = ts_test.to(args.device)
                ids_test = ids_test.to(args.device)
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)
                # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                if args.n_batch ==1:
                    if args.eqn_type == 'Burgers':
                        c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_test[:,:1])
                        interpolation = NaturalCubicSpline(c_coeffs)
                        c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,args.time_points).unsqueeze(-1)
                    else:
                        c = lambda x: obs_test[:,:,:1].repeat(1,1,args.time_points).unsqueeze(-1).to(device)
                else:
                    if args.eqn_type == 'Burgers':
                        c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_test[:,:,:1])
                        interpolation = NaturalCubicSpline(c_coeffs)
                        c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,1,args.time_points).unsqueeze(-1)
                    else:
#                                 c= lambda x: \
#                                             Encoder(obs_test[:,:,:,:1].repeat(1,1,1,args.time_points)\
#                                             .permute(0,3,1,2)).unsqueeze(-1)\
#                                             .permute(0,2,3,1,4).contiguous().to(args.device)
                        c= lambda x: Encoder(obs_test[:,:,:,:1].permute(0,3,1,2))\
                                    .permute(0,2,3,1).unsqueeze(-2)\
                                    .contiguous().repeat(1,1,1,args.time_points,1).to(args.device)

                if args.eqn_type == 'Navier-Stokes':
#                             y_0 = Encoder(obs_test[:,:,:,:1].repeat(1,1,1,args.time_points)\
#                             .permute(0,3,1,2)).unsqueeze(-1)\
#                             .permute(0,2,3,1,4)[:,:,:,:1,:]
                    y_0 = Encoder(obs_test[:,:,:,:1].permute(0,3,1,2))\
                                    .permute(0,2,3,1).unsqueeze(-2)\
                                    .to(args.device)

                if args.ts_integration is not None:
                    times_integration = args.ts_integration
                else:
                    times_integration = torch.linspace(0,1,args.time_points)

                if args.support_test is False:
                    if args.n_batch==1:
                        if args.eqn_type == 'Burgers':
                            z_test = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_test[0].unsqueeze(1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                    spatial_domain_dim=1,
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()
                        else:
                            z_test = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_test[:,:,0].unsqueeze(-1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(device),
                                    spatial_domain_dim=2,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()

                    else:
                        if args.eqn_type == 'Burgers':
                            z_test = Integral_spatial_attention_solver_multbatch(
                                torch.linspace(0,1,args.time_points).to(device),
                                obs_test[:,0].unsqueeze(-1).to(args.device),
                                c=c,
                                sampling_points = args.time_points,
                                mask=mask,
                                Encoder = model,
                                max_iterations = args.max_iterations,
                                spatial_integration=True,
                                spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                                spatial_domain_dim=1,
                                #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                smoothing_factor=args.smoothing_factor,
                                use_support=False,
                                ).solve()
                        else:
                            z_test = Integral_spatial_attention_solver_multbatch(
                                    times_integration.to(args.device),
                                    y_0.to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(args.device),
                                    spatial_domain_dim=2,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()
                else:
                    z_test = Integral_spatial_attention_solver(
                            torch.linspace(0,1,args.time_points).to(device),
                            obs_[0].unsqueeze(1).to(args.device),
                            c=c,
                            sampling_points = args.time_points,
                            support_tensors=dummy_times.to(device),
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            spatial_integration=True,
                            spatial_domain= torch.linspace(0,1,args.n_points).to(device),
                            spatial_domain_dim=1,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            smoothing_factor=args.smoothing_factor,
                            output_support_tensors=True,
                            ).solve()




                if Decoder is not None:
                    z_test = Decoder(z_test)
                else:
                    z_test = z_test.view(args.n_batch,Data.shape[1],Data.shape[2],args.time_points)
                if args.initial_t is False:
                    obs_test = obs_test[...,1:]
                    z_test = z_test[...,1:]
                    
                mse_error = F.mse_loss(z_test, obs_test.detach())
                
                test_loss += mse_error.item()
                loss_list.append(mse_error.item())
                
#                 for in_batch_indx in range(args.n_batch):

#                     obs_print = to_np(obs_test[in_batch_indx,:,:,:])
#                     z_p = to_np(z_test[in_batch_indx,:,:,:])

#                     #plot_reconstruction(obs_print, z_p, None, path_to_save_plots, 'plot_epoch_', i, args)
#                     plot_reconstruction(obs_print, z_p, None, None, None, None, args)

#                     plt.close('all')
#                     del z_p, obs_print
                del z_test, obs_test
            
            print(loss_list)
            print("Average loss: ",test_loss*args.n_batch/obs.shape[0])
            

def Full_experiment_AttentionalIE_GeneratedFMRI(model, Data, dataloaders, time_seq, index_np, mask, times, args, extrapolation_points): # experiment_name, plot_freq=1):
    verbose=False
    
    #metadata for saving checkpoints
    if args.model=='nie': 
        str_model_name = "nie"
    elif args.model=='node': 
        str_model_name = "node"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)
    print('path_to_experiment: ',path_to_experiment)

    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        if verbose: print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        txt = [x for x in txt if not x.startswith('@$\t') and not x.startswith('eval_')]
        if len(txt) == 0:
            num_experiments=0
        else: 
            num_experiments = [int(i[3:]) for i in txt]
            num_experiments = np.array(num_experiments).max()
        #  # -- logger location
        # writer = SummaryWriter(os.path.join(path_to_experiment,'run'+str(num_experiments+1)))
        # print('writer.log_dir: ',writer.log_dir)
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)
            
        # with open(os.path.join(writer.log_dir,'commandline_args.txt'), 'w') as f:
        #     for key, value in args.__dict__.items(): 
        #         f.write('%s:%s\n' % (key, value))


    times = time_seq
    if verbose: print('times.shape: ',times.shape)
    
    All_parameters = model.parameters()
    
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        model, optimizer, scheduler, kernel, F_func, f_func = load_checkpoint(path, model, optimizer, scheduler, None, None,  None)


    
    if args.mode=='train':
        #lr_scheduler = LRScheduler(optimizer,patience = 50,min_lr=1e-5,factor=0.1)
        early_stopping = EarlyStopping(patience=1000,min_delta=0)

        # Loss_print = []
        # Val_Loss = []
        all_train_loss=[]
        all_val_loss=[]

        save_best_model = SaveBestModel()
        # save_last_model = SaveLastState()
        start = time.time()
        
        minibatch_train_counter=0
        minibatch_val_counter=0
        for epoch in range(args.epochs):
            
            model.train()
            
            start_i = time.time()
            print('Epoch:',epoch)
            counter=0
            train_loss = 0.0

            for obs_, ts_, ids_, frames_to_drop_train in tqdm(dataloaders['train']): 
        
                if verbose: 
                    print('[first] obs_.shape: ',obs_.shape)
                    print('[first] ids_.shape: ',ids_.shape)
                    print('[first] ts_.shape: ',ts_.shape)
                    print('[first] ts_: ',ts_)
                
                # obs_, ts_, ids_ = next(iter(train_loader))
                ids_, indices = torch.sort(ids_)
                # if verbose: print('indices: ',indices)
                obs_ = obs_[:,indices[0,:]]
                # obs_ = torch.cat([obs[j,0,:],obs_]) # To ensure the first point is always there.
                ts_ = ts_[:,indices[0,:]]

                if args.support_tensors is True or args.support_test is True:
                    if args.combine_points is True:
                        # dummy_times = torch.linspace(times[0],times[-1],args.sampling_points).float()
                        #Add in-between frames
                        dummy_times,_ = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(ts_.squeeze()[0], ts_.squeeze()[-1]))
                        dummy_times,_ = torch.sort(torch.cat((ts_.squeeze().cpu(),dummy_times))) # append the in-between and sort

                        #Check if there are duplicates and resample if there are
                        dup=np.array([0])
                        while dup.size != 0:
                            u, c = np.unique(dummy_times, return_counts=True)
                            dup = u[c > 1]
                            if dup.size != 0:
                                print('[sampling_full_frames] There are duplicated time coordinates: ',dup)
                                print('Resampling')

                                 #Add in-between frames
                                dummy_times,_ = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(ts_.squeeze()[0], ts_.squeeze()[-1]))
                                dummy_times,_ = torch.sort(torch.cat((ts_.squeeze().cpu(),dummy_times))) # append the in-between and sort
                        real_idx = np.argwhere(np.isin(dummy_times,ts_.squeeze().cpu(),invert=False)).flatten() #returns the indexes of the dummy frames. So dummy_idxs and 
                        if verbose: 
                            print('real_idx: ',real_idx)
                            print('dummy_times: ',dummy_times)
                    else:
                        # dummy_times = torch.linspace(times[0],times[-1],args.sampling_points).float()
                        #Add in-between frames
                        dummy_times,_ = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(ts_.squeeze()[0], ts_.squeeze()[-1]))
                        
                        #Check if there are duplicates and resample if there are
                        dup=np.array([0])
                        while dup.size != 0:
                            u, c = np.unique(dummy_times, return_counts=True)
                            dup = u[c > 1]
                            if dup.size != 0:
                                print('There are duplicated time coordinates: ',dup)
                                print('Resampling')

                                 #Add in-between frames
                                dummy_times,_ = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(ts_.squeeze()[0], ts_.squeeze()[-1]))

                    if verbose: 
                    #     print('dummy_times.type(): ',dummy_times.type())
                        print('dummy_times.shape: ',dummy_times.shape)
                
                obs_ = obs_.to(args.device)
                ts_ = ts_.to(args.device)
                ids_ = ids_.to(args.device)

                del indices, ids_

                if args.random_sample_n_points is not None:
                    idx_downsampled_points = np.random.choice(np.arange(1,len(ts_[0,:])), args.random_sample_n_points, replace=False)
                    idx_downsampled_points = np.sort(np.insert(idx_downsampled_points, 0, 0))
                    # # print('idx_downsampled_points.shape: ',idx_downsampled_points.shape)
                    # idx_downsampled_points = np.arange(0,21,step=5)-1
                    # idx_downsampled_points[0]=0
                    obs_=obs_[:,idx_downsampled_points,:]
                    ts_=ts_[:,idx_downsampled_points]
                
                if args.perturbation_to_obs:
                    # print('adding perturbation: ')
                    perturb = torch.normal(mean=torch.zeros_like(obs_).to(args.device),
                                              std=args.std_noise)#args.perturbation_to_obs0*obs_[:3,:].std(dim=0))
                    # print('perturb.shape: ',perturb.shape)
                else: perturb=None

                # Drop frames if specified
                if args.randomly_drop_n_last_frames is not None or args.drop_n_last_frames is not None:
                    frames_to_drop = frames_to_drop_train.numpy()[0]
                    if verbose: print('frames_to_drop: ',frames_to_drop)
                    obs_cropped = obs_[:,:frames_to_drop,:]
                    ts_cropped = ts_[:,:frames_to_drop]
                    if verbose: print('obs_cropped.shape: {}, ts_cropped.shape: {}'.format(obs_cropped.shape, ts_cropped.shape))
                else: frames_to_drop=None

                if args.support_tensors is True or args.support_test is True:
                    if verbose: print('[in epxeriments.py] dummy_times[:10]: ',dummy_times[:10])

                model = model.float()
                
                if args.integral_c is not None: # To pass a C
                    if args.integral_c == 'cte_2nd_half': # In this case, all points of the first half are used. So no need for downsample
                        interpolation = fun_interpolation(obs_.to(args.device),ts_.squeeze().to(args.device),verbose=False, given_points = args.num_points_for_c) #Original 
                        c = lambda x: interpolation.cte_2nd_half(x, noise=perturb, c_scaling_factor=args.c_scaling_factor).to(args.device) #I am giving x, but it is not actually used
                    else:
                        downsample_idx = np.sort(np.random.choice(ts_.shape[1], args.num_points_for_c, replace=False))
                        downsample_idx = np.insert(downsample_idx, 0, 0)
                        downsample_idx = np.insert(downsample_idx, len(downsample_idx), ts_.shape[1]-1)
                        if verbose: print('downsample_idx: ',downsample_idx)
                        dup=np.array([0])
                        while dup.size != 0:
                            u, c = np.unique(downsample_idx, return_counts=True)
                            dup = u[c > 1]
                            if dup.size != 0:
                                downsample_idx = np.sort(np.random.choice(ts_.shape[1], args.num_points_for_c, replace=False))
                                downsample_idx = np.insert(downsample_idx, 0, 0)
                                downsample_idx = np.insert(downsample_idx, len(downsample_idx), ts_.shape[1]-1)

                        # if verbose: print('downsample_idx: ',downsample_idx)

                        t_downsample = ts_[0, downsample_idx]
                        # if verbose:
                        #     print('ts_: ',ts_)
                        #     print('downsample_idx: ',downsample_idx)
                        #     print('t_downsample: ',t_downsample)
                        obs_downsampled = obs_[:,downsample_idx,:]
                        if verbose:
                            print('ts_: ',ts_)
                            print('downsample_idx: ',downsample_idx)
                            print('t_downsample.shape: ',t_downsample.shape)
                            print('t_downsample: ',t_downsample)
                            print('obs_downsampled.shape: ',obs_downsampled.shape)

                        # interpolation = fun_interpolation(obs_downsampled.squeeze().to(args.device),t_downsample.squeeze().to(args.device),verbose=True) #Original 
                        interpolation = fun_interpolation(obs_downsampled.to(args.device),t_downsample.squeeze().to(args.device),verbose=False) #Original 
                        if args.integral_c == 'spline': # To pass a C as cubic spline
                            c = lambda x: interpolation.spline_interpolation(x).squeeze()
                        elif args.integral_c == 'linear': # 
                            c = lambda x: interpolation.linear_interpolation(x).to(args.device)
                        elif args.integral_c == 'step': # 
                            c = lambda x: interpolation.step_interpolation(x).to(args.device)
                        
                elif args.integral_c is None:
                    c=None
                
                if args.support_tensors is False:
                    if verbose: print('using Integral_attention_solver_multbatch')
                    z_ = Integral_attention_solver_multbatch(
                            # ts_.to(device),
                            ts_[0].squeeze().to(device), # Because it has to be a 1D vector
                            obs_[:,0,:].unsqueeze(1).to(args.device), # This should should the first point for each curve in the batch. For the multbatch [num_batch, 1 time point, num_dim]. For single batch [1 time point,num_dim]
                            c=c,
                            sampling_points = ts_[0].squeeze().size(0),
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            smoothing_factor=args.smoothing_factor,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            use_support=args.use_support,
                            ).solve()
                else:
                    z_ = Integral_attention_solver_multbatch(
                            # ts_.to(device),
                            # obs_[0].unsqueeze(0).to(args.device),
                            ts_[0].squeeze().to(device), # Because it has to be a 1D vector
                            obs_[:,0,:].unsqueeze(1).to(args.device), # This should should the first point for each curve in the batch. For the multbatch [num_batch, 1 time point, num_dim]. For single batch [1 time point,num_dim]
                            c=c,
                            # sampling_points = dummy_times.size(0),
                            sampling_points = dummy_times.squeeze().size(0),
                            support_tensors=dummy_times.squeeze().to(args.device),
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            smoothing_factor=args.smoothing_factor,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            output_support_tensors=args.use_support,
                            ).solve()
                    if args.combine_points is True:
                        z_ = z_[:,real_idx,:]
            
                
                # #loss_ts_ = get_times.select_times(ts_)[1]
                # if args.randomly_drop_n_last_frames is not None:
                #     frames_to_drop = frames_to_drop_train.numpy()[0]
                # else: frames_to_drop=None
                
                if len(z_.shape)<3: #because if it is single curve per batch, this won't return the batch=1 dimension.
                    z_ = z_[None,:]
                    if args.combine_points is True:
                            z_clone =  z_clone[None,:] 
                if verbose: 
                    print('[before loss] z_: ',z_)
                    print('[before loss] z_.shape: ',z_.shape)
                    
                if args.compute_loss_on_unseen_points:
                    all_points = np.arange(ts_.shape[1])
                    unseen_points = np.argwhere(np.isin(all_points,downsample_idx,invert=True)).flatten() #returns the indexes of the dummy frames. So dummy_idxs and 
                    all_points = all_points[unseen_points]
                    if verbose: 
                        print('\nselecting unseen points: ')
                        print('downsample_idx: ',downsample_idx)
                        print('np.arange(ts_.shape[1]): ',np.arange(ts_.shape[1]))
                        print('unseen_points: ',unseen_points)
                        print('all_points: ',all_points)
                    z_unseen = z_[:,unseen_points,:]
                    obs_unseen = obs_[:,unseen_points,:]
                    loss = F.mse_loss(z_unseen, obs_unseen.detach()) #Original 
                else: 
                    loss = F.mse_loss(z_, obs_.detach()) #Original 
                # print('z_[:,:].to(args.device): ',z_[:,:].to(args.device))
                # print('obs_.to(args.device).detach()[:,:]: ',obs_.to(args.device).detach()[:,:])
                # loss = F.mse_loss(z_[:,:].to(args.device), obs_.to(args.device).detach()[:,:]) #Original 


                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # loss.backward()
                optimizer.step()

                # n_iter += 1
                counter += 1
                minibatch_train_counter+=1
                train_loss += loss.item()
                del loss
                
                # if args.log_per_minibatch and minibatch_train_counter%args.num_minibatches==0:
                #     tmp_loss = train_loss / counter
                #     writer.add_scalar('Minibatch/train_loss', tmp_loss, global_step=minibatch_train_counter)

            train_loss /= counter
            all_train_loss.append(train_loss)
            del train_loss
            
            if epoch>args.warm_up and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
            # if args.validation_split==0:
            #     scheduler.step(train_loss)
            
            if epoch % args.plot_freq == 0:
                model.eval()
                with torch.no_grad():
                    z_ = Integral_attention_solver_multbatch(
                            # ts_.to(device),
                            ts_[0].squeeze().to(device), # Because it has to be a 1D vector
                            obs_[:,0,:].unsqueeze(1).to(args.device), # This should should the first point for each curve in the batch. For the multbatch [num_batch, 1 time point, num_dim]. For single batch [1 time point,num_dim]
                            c=c,
                            sampling_points = ts_[0].squeeze().size(0),
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            smoothing_factor=args.smoothing_factor,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            use_support=args.use_support,
                            ).solve()

                    obs_to_print = to_np(obs_)
                    time_to_print = to_np(ts_)
                    z_to_print = to_np(z_)
                    if args.support_tensors is True or args.support_test is True:
                        z_all_to_print = to_np(z_clone)
                        dummy_times_to_print = to_np(dummy_times)
                        if verbose: print('dummy_times_to_print.shape: ',dummy_times_to_print.shape)
                    else:
                        dummy_times_to_print = to_np(ts_)[0,:]
                        z_all_to_print = to_np(z_)
                    # if len(z_to_print.shape)<3: #because if it is single curve per batch, this won't return the batch=1 dimension.
                    #     z_to_print = z_to_print[None,:]
                    # #     time_to_print = time_to_print[None,:]
                        
                    if obs_to_print.shape[2]<args.num_dim_plot: args.num_dim_plot=obs_to_print.shape[2]

                    # plot_dim_vs_time(obs_to_print[0,:], time_to_print[0,:], z_to_print[0,:], frames_to_drop, path_to_save_plots, name='plot_train_ndim_epoch', epoch=epoch, args=args)
                    plot_dim_vs_time(obs_to_print[0,:], time_to_print[0,:], z_to_print[0,:], dummy_times_to_print, z_all_to_print[0,:], frames_to_drop, path_to_save_plots, name='plot_train_ndim_epoch', epoch=epoch, args=args)
                
                    
                    if args.integral_c is not None: # To pass a C# Now print 'c' 
                        if args.integral_c == 'cte_2nd_half':
                            obs_downsampled = obs_.detach().clone()
                            t_downsample = ts_[0].detach().clone()
                        obs_to_print = to_np(obs_downsampled)
                        time_to_print = to_np(t_downsample)
                        z_real_to_print = to_np(c(t_downsample))
                        z_all_to_print = to_np(c(ts_[0,:].to(args.device)))
                        if len(z_real_to_print.shape)<3: #because if it is single curve per batch, this won't return the batch=1 dimension.
                            z_real_to_print = z_real_to_print[None,:]
                        # z_all_to_print = z_all_to_print[None,:]

                        if verbose: 
                            print('Plotting c')
                            print('obs_to_print.shape: ',obs_to_print.shape)
                            print('time_to_print.shape: ',time_to_print.shape)
                            print('z_real_to_print.shape: ',z_real_to_print.shape)
                            print('z_all_to_print.shape: ',z_all_to_print.shape)

                        plot_dim_vs_time(obs_to_print[0,:], time_to_print, z_real_to_print[0,:], dummy_times_to_print, z_all_to_print[0,:], frames_to_drop, path_to_save_plots, name='plot_train_c_ndim_epoch', epoch=epoch, args=args)
                    
                    del obs_to_print, time_to_print, z_to_print
            del obs_, ts_, z_
            


            ## Validating
            model.eval()
            with torch.no_grad():

                #Only do this if there is a validation dataset
                
                val_loss = 0.0
                counter = 0
                if args.validation_split>0 and len(dataloaders['val'])>0: # If there are validation samples
                    for obs_val, ts_val, ids_val, frames_to_drop_val in tqdm(dataloaders['val']):

                        ids_val, indices = torch.sort(ids_val)
                        # print('indices: ',indices)
                        obs_val = obs_val[:,indices[0,:]]
                        ts_val = ts_val[:, indices[0,:]]
                
                        obs_val = obs_val.to(args.device)
                        ts_val = ts_val.to(args.device)
                        ids_val = ids_val.to(args.device)
                        del indices, ids_val

                        if args.random_sample_n_points is not None:
                            idx_downsampled_points = np.random.choice(np.arange(1,len(ts_val[0,:])), args.random_sample_n_points, replace=False)
                            idx_downsampled_points = np.sort(np.insert(idx_downsampled_points, 0, 0))
                            # # print('idx_downsampled_points.shape: ',idx_downsampled_points.shape)
                            # idx_downsampled_points = np.arange(0,21,step=5)-1
                            # idx_downsampled_points[0]=0
                            obs_val=obs_val[:,idx_downsampled_points,:]
                            ts_val=ts_val[:,idx_downsampled_points]

                        # Drop frames if specified
                        if args.randomly_drop_n_last_frames is not None or args.drop_n_last_frames is not None:
                            frames_to_drop = frames_to_drop_val.numpy()[0]
                            if verbose: print('frames_to_drop: ',frames_to_drop)
                            obs_val_cropped = obs_val[:,:frames_to_drop,:]
                            ts_val_cropped = ts_val[:,:frames_to_drop]
                            if verbose: print('obs_cropped.shape: {}, ts_cropped.shape: {}'.format(obs_cropped.shape, ts_cropped.shape))
                        else: frames_to_drop=None
                        

                        if args.integral_c is not None: # To pass a C
                            if args.integral_c == 'cte_2nd_half': # In this case, all points of the first half are used. So no need for downsample
                                interpolation = fun_interpolation(obs_val.to(args.device),ts_val.squeeze().to(args.device),verbose=False,  given_points = args.num_points_for_c) #Original 
                                c = lambda x: interpolation.cte_2nd_half(x, c_scaling_factor=args.c_scaling_factor).to(args.device) #I am giving x, but it is not actually used
                            else:
                                downsample_idx = np.sort(np.random.choice(ts_val.shape[1], args.num_points_for_c, replace=False))
                                downsample_idx = np.insert(downsample_idx, 0, 0)
                                downsample_idx = np.insert(downsample_idx, len(downsample_idx), ts_val.shape[1]-1)
                                if verbose: print('downsample_idx: ',downsample_idx)
                                dup=np.array([0])
                                while dup.size != 0:
                                    u, c = np.unique(downsample_idx, return_counts=True)
                                    dup = u[c > 1]
                                    if dup.size != 0:
                                        downsample_idx = np.sort(np.random.choice(ts_val.shape[1], args.num_points_for_c, replace=False))
                                        downsample_idx = np.insert(downsample_idx, 0, 0)
                                        downsample_idx = np.insert(downsample_idx, len(downsample_idx), ts_val.shape[1]-1)

                                # if verbose: print('downsample_idx: ',downsample_idx)

                                t_downsample = ts_val[0, downsample_idx]
                                # if verbose:
                                #     print('ts_: ',ts_)
                                #     print('downsample_idx: ',downsample_idx)
                                #     print('t_downsample: ',t_downsample)
                                obs_downsampled = obs_val[:,downsample_idx,:]
                                if verbose:
                                    print('ts_val.shape: ',ts_val.shape)
                                    print('downsample_idx: ',downsample_idx)
                                    print('t_downsample.shape: ',t_downsample.shape)
                                    print('t_downsample: ',t_downsample)
                                    print('obs_downsampled.shape: ',obs_downsampled.shape)

                                # interpolation = fun_interpolation(obs_downsampled.squeeze().to(args.device),t_downsample.squeeze().to(args.device),verbose=True) #Original 
                                interpolation = fun_interpolation(obs_downsampled.to(args.device),t_downsample.squeeze().to(args.device),verbose=False) #Original 
                                if args.integral_c == 'spline': # To pass a C as cubic spline
                                    c = lambda x: interpolation.spline_interpolation(x).squeeze()
                                elif args.integral_c == 'linear': # 
                                    c = lambda x: interpolation.linear_interpolation(x).to(args.device)
                                elif args.integral_c == 'step': # 
                                    c = lambda x: interpolation.step_interpolation(x).to(args.device)

                        elif args.integral_c is None:
                            c=None
                        
                        
                        
                        
                        if verbose: print('[in val] using Integral_attention_solver_multbatch')
                        if args.support_tensors is False:
                            z_val = Integral_attention_solver_multbatch(
                                    # ts_val.to(device),
                                    # obs_val[0].unsqueeze(0).to(args.device),
                                    # sampling_points = ts_val.size(0),
                                    ts_val[0].squeeze().to(device), # Because it has to be a 1D vector
                                    obs_val[:,0,:].unsqueeze(1).to(args.device), # This should should the first point for each curve in the batch. For the multbatch [num_batch, 1 time point, num_dim]. For single batch [1 time point,num_dim]
                                    c=c,
                                    sampling_points = ts_val[0].squeeze().size(0),
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    smoothing_factor=args.smoothing_factor,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    use_support=False,
                                    ).solve()
                        else:
                            z_val = Integral_attention_solver_multbatch(
                                    # ts_val.to(device),
                                    # obs_[0].unsqueeze(0).to(args.device),
                                    # sampling_points = dummy_times.size(0),
                                    # support_tensors=dummy_times.to(device),
                                    ts_val[0].squeeze().to(device), # Because it has to be a 1D vector
                                    obs_val[:,0,:].unsqueeze(1).to(args.device), # This should should the first point for each curve in the batch. For the multbatch [num_batch, 1 time point, num_dim]. For single batch [1 time point,num_dim]
                                    c=c,
                                    # sampling_points = dummy_times.size(0),
                                    sampling_points = dummy_times.squeeze().size(0),
                                    support_tensors=dummy_times.squeeze().to(args.device),
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    smoothing_factor=args.smoothing_factor,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    output_support_tensors=args.output_support_tensors
                                    ).solve()

                            if args.combine_points is True:
                                z_val = z_val[:,real_idx,:]


                        #validation_ts_ = get_times.select_times(ts_val)[1]
                        if len(z_val.shape)<3: #because if it is single curve per batch, this won't return the batch=1 dimension.
                            z_val = z_val[None,:]
                            if args.combine_points is True:
                                z_clone =  z_clone[None,:] 
                                
                        if args.compute_loss_on_unseen_points:
                            all_points = np.arange(ts_val.shape[1])
                            unseen_points = np.argwhere(np.isin(all_points,downsample_idx,invert=True)).flatten() #returns the indexes of the dummy frames. So dummy_idxs and 
                            all_points = all_points[unseen_points]
                            if verbose: 
                                print('\nselecting unseen points: ')
                                print('downsample_idx: ',downsample_idx)
                                print('np.arange(ts_.shape[1]): ',np.arange(ts_val.shape[1]))
                                print('unseen_points: ',unseen_points)
                                print('all_points: ',all_points)
                            z_unseen = z_val[:,unseen_points,:]
                            obs_unseen = obs_val[:,unseen_points,:]
                            loss_validation = F.mse_loss(z_unseen, obs_unseen.detach()) #Original 
                        else:
                            loss_validation = F.mse_loss(z_val, obs_val.detach())
                        # Val_Loss.append(to_np(loss_validation))


                        counter += 1
                        minibatch_val_counter+=1
                        val_loss += loss_validation.item()
                        del loss_validation
                        
                        # if args.log_per_minibatch and minibatch_val_counter%args.num_minibatches==0:
                        #     tmp_loss = val_loss / counter
                        #     writer.add_scalar('Minibatch/val_loss', tmp_loss, global_step=minibatch_val_counter)

                            
                    else: counter += 1

                    val_loss /= counter
                    all_val_loss.append(val_loss)

                    #LRScheduler(loss_validation)
                    if args.lr_scheduler == 'ReduceLROnPlateau':
                        scheduler.step(val_loss)
                    del val_loss

                    if epoch % args.plot_freq == 0:
                        # obs_val, ts_val, ids_val = obs_val.squeeze(), ts_val.squeeze(), ids_val.squeeze()
                        obs_to_print = to_np(obs_val)
                        time_to_print = to_np(ts_val)
                        z_to_print = to_np(z_val)
                        z_all_to_print = to_np(z_val)
                        if args.support_tensors is True or args.support_test is True:
                            dummy_times_to_print = to_np(dummy_times)
                            if verbose: print('dummy_times_to_print.shape: ',dummy_times_to_print.shape)
                        else:
                            # dummy_times_to_print = to_np(ts_)[0,:]
                            # z_all_to_print = to_np(z_)
                            dummy_times = torch.linspace(0,1,1000)
                            dummy_times_to_print = to_np(dummy_times)
                            if verbose: print('dummy_times.shape: ',dummy_times.shape)

                            if args.integral_c != 'cte_2nd_half' and args.integral_c is not None:
                                z_all_to_print = Integral_attention_solver_multbatch(
                                    # ts_val.to(device),
                                    # obs_val[0].unsqueeze(0).to(args.device),
                                    # sampling_points = ts_val.size(0),
                                    dummy_times.squeeze().to(device), # Because it has to be a 1D vector
                                    obs_val[:,0,:].unsqueeze(1).to(args.device), # This should should the first point for each curve in the batch. For the multbatch [num_batch, 1 time point, num_dim]. For single batch [1 time point,num_dim]
                                    c=c,
                                    sampling_points = dummy_times.squeeze().size(0),
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    smoothing_factor=args.smoothing_factor,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    use_support=False,
                                    ).solve()
                                z_all_to_print = to_np(z_all_to_print)
                            elif args.integral_c == 'cte_2nd_half':
                                dummy_times_to_print = time_to_print[0,:]
                        # if len(z_to_print.shape)<3: #because if it is single curve per batch, this won't return the batch=1 dimension.
                        #     print('[before] z_to_print.shape: ',z_to_print.shape)
                        #     z_to_print = z_to_print[None,:]

                        # frames_to_drop=None
                        # plot_dim_vs_time(obs_to_print[0,:], time_to_print[0,:], z_to_print[0,:], frames_to_drop, path_to_save_plots, name='plot_val_ndim_epoch', epoch=epoch, args=args)
                        plot_dim_vs_time(obs_to_print[0,:], time_to_print[0,:], z_to_print[0,:], dummy_times_to_print, z_all_to_print[0,:], frames_to_drop, path_to_save_plots, name='plot_val_ndim_epoch', epoch=epoch, args=args)
                            
                        if args.integral_c is not None: # To pass a C# Now print 'c' 
                            if args.integral_c == 'cte_2nd_half':
                                obs_downsampled = obs_val.detach().clone()
                                t_downsample = ts_val[0].detach().clone()
                            obs_to_print = to_np(obs_downsampled)
                            time_to_print = to_np(t_downsample)
                            z_real_to_print = to_np(c(t_downsample))
                            # z_all_to_print = to_np(c(ts_[0,:].to(args.device)))
                            z_all_to_print = to_np(c(dummy_times.to(args.device)))
                            if len(z_real_to_print.shape)<3: #because if it is single curve per batch, this won't return the batch=1 dimension.
                                z_real_to_print = z_real_to_print[None,:]
                            # z_all_to_print = z_all_to_print[None,:]

                            if verbose: 
                                print('Plotting c')
                                print('obs_to_print.shape: ',obs_to_print.shape)
                                print('time_to_print.shape: ',time_to_print.shape)
                                print('z_real_to_print.shape: ',z_real_to_print.shape)
                                print('z_all_to_print.shape: ',z_all_to_print.shape)

                            plot_dim_vs_time(obs_to_print[0,:], time_to_print, z_real_to_print[0,:], dummy_times_to_print, z_all_to_print[0,:], frames_to_drop, path_to_save_plots, name='plot_val_c_ndim_epoch', epoch=epoch, args=args)
                    

                        del z_to_print, time_to_print, obs_to_print
                    del obs_val, ts_val, z_val
                

            # writer.add_scalar('train_loss', all_train_loss[-1], global_step=epoch)
            # if len(all_val_loss)>0:
            #     writer.add_scalar('val_loss', all_val_loss[-1], global_step=epoch)
            # if args.lr_scheduler == 'ReduceLROnPlateau':
            #     writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch)
            # elif args.lr_scheduler == 'CosineAnnealingLR':
            #     writer.add_scalar('Epoch/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)

            end_i = time.time()

            
            model_state = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            if args.validation_split>0:
                save_best_model(path_to_save_models, all_val_loss[-1], epoch, model_state, model, None, None, None)
            else: 
                save_best_model(path_to_save_models, all_train_loss[-1], epoch, model_state, model, None, None, None)
            # save_last_model(path_to_save_models, all_train_loss[-1], epoch, model_state, None, None, None, None)

            if len(all_val_loss)>0:
                early_stopping(all_val_loss[-1])
            else: 
                early_stopping(all_train_loss[-1])
                
            if early_stopping.early_stop:
                break

        end = time.time()
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')
        verbose=False

        # # Create a 'eval' folder to save things
        # if verbose: print('path_to_experiment: ',path_to_experiment)
        
        path_to_save_plots = os.path.join(path_to_experiment,'eval_'+args.resume_from_checkpoint)
        
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)

        if verbose: print('saving variables to: ',path_to_save_plots)
        
        with open(os.path.join(path_to_save_plots,'commandline_args.txt'), 'w') as f:
            for key, value in args.__dict__.items(): 
                f.write('%s:%s\n' % (key, value))
        
        ## Validating
        model.eval()
        with torch.no_grad():
            # splitting_size = int(args.training_split*Data.size(0))
            all_r2_scores = []
            all_mse = []
            batch_idx = 0
            for obs_test, ts_test, ids_test, frames_to_drop_test in tqdm(dataloaders['val']):
                # Dataset_all = Test_Dynamics_Dataset(Data[j,:,:],times)
                # loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np)))
                
                ids_test, indices = torch.sort(ids_test)
                # print('indices: ',indices)
                obs_test = obs_test[:,indices[0,:]]
                ts_test = ts_test[:, indices[0,:]]

                obs_test = obs_test.to(args.device)
                ts_test = ts_test.to(args.device)
                # ids_val = ids_val.to(args.device)
                del indices, ids_test

                # Drop frames if specified
                if args.randomly_drop_n_last_frames is not None or args.drop_n_last_frames is not None:
                    frames_to_drop = frames_to_drop_test.numpy()[0]
                    if verbose: print('frames_to_drop: ',frames_to_drop)
                    # obs_val_cropped = obs_test[:,:frames_to_drop,:]
                    # ts_val_cropped = ts_test[:,:frames_to_drop]
                    # if verbose: print('obs_cropped.shape: {}, ts_cropped.shape: {}'.format(obs_cropped.shape, ts_cropped.shape))
                else: frames_to_drop=None
                
                if verbose: 
                    print('ts_test: ',ts_test)
                    print('times: ',times)
                
                if args.support_tensors is True or args.support_test is True:
                    if args.combine_points is True:
                        sampled_tensors,real_idx = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(0, 1).float())
                        if verbose: 
                            print('real_idx: ',real_idx)
                            print('sampled_tensors: ',sampled_tensors)
                            print('sampled_tensors.type(): ',sampled_tensors.type())
                        #Check if there are duplicates and resample if there are
                        sampled_tensors = torch.cat([times,sampled_tensors])
                        dup=np.array([0])
                        while dup.size != 0:
                            u, c = np.unique(sampled_tensors, return_counts=True)
                            dup = u[c > 1]
                            if dup.size != 0:
                                sampled_tensors,real_idx = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(0, 1).float())
                                sampled_tensors  = torch.cat([times,sampled_tensors])
                        dummy_times=sampled_tensors.float()
                        real_idx=real_idx[:times.size(0)]
                    else:
                        # dummy_times = torch.linspace(times[0],times[-1],args.sampling_points).float()
                        #Add in-between frames
                        dummy_times,_ = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(ts_test.squeeze()[0], ts_test.squeeze()[-1]))
                        dummy_times,_ = torch.sort(torch.cat((ts_test.squeeze().cpu(),dummy_times))) # append the in-between and sort

                        #Check if there are duplicates and resample if there are
                        dup=np.array([0])
                        while dup.size != 0:
                            u, c = np.unique(dummy_times, return_counts=True)
                            dup = u[c > 1]
                            if dup.size != 0:
                                print('[sampling_full_frames] There are duplicated time coordinates: ',dup)
                                print('Resampling')

                                 #Add in-between frames
                                dummy_times,_ = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(ts_test.squeeze()[0], ts_test.squeeze()[-1]))
                                dummy_times,_ = torch.sort(torch.cat((ts_test.squeeze().cpu(),dummy_times))) # append the in-between and sort

                    if verbose: 
                        print('dummy_times.type(): ',dummy_times.type())
                        print('dummy_times.shape: ',dummy_times.shape)
                
                if args.integral_c is not None: # To pass a C
                    if args.integral_c == 'cte_2nd_half': # In this case, all points of the first half are used. So no need for downsample
                        interpolation = fun_interpolation(obs_test.to(args.device),ts_test.squeeze().to(args.device),verbose=False,  given_points = args.num_points_for_c) #Original 
                        c = lambda x: interpolation.cte_2nd_half(x, c_scaling_factor=args.c_scaling_factor).to(args.device) #I am giving x, but it is not actually used
                    else:
                        #Downsample to 1/3 of the points, randomly 
                        downsample_idx = np.sort(np.random.choice(ts_test.shape[1], args.num_points_for_c, replace=False))
                        downsample_idx = np.insert(downsample_idx, 0, 0)
                        downsample_idx = np.insert(downsample_idx, len(downsample_idx), ts_test.shape[1]-1)
                        if verbose: print('downsample_idx: ',downsample_idx)
                        dup=np.array([0])
                        while dup.size != 0:
                            u, c = np.unique(downsample_idx, return_counts=True)
                            dup = u[c > 1]
                            if dup.size != 0:
                                downsample_idx = np.sort(np.random.choice(ts_test.shape[1], args.num_points_for_c, replace=False))
                                downsample_idx = np.insert(downsample_idx, 0, 0)
                                downsample_idx = np.insert(downsample_idx, len(downsample_idx), ts_test.shape[1]-1)

                        # if verbose: print('downsample_idx: ',downsample_idx)

                        t_downsample = ts_test[0, downsample_idx]
                        # if verbose:
                        #     print('ts_: ',ts_)
                        #     print('downsample_idx: ',downsample_idx)
                        #     print('t_downsample: ',t_downsample)
                        obs_downsampled = obs_test[:,downsample_idx,:]
                        if verbose:
                            print('ts_test: ',ts_test)
                            print('downsample_idx: ',downsample_idx)
                            print('t_downsample.shape: ',t_downsample.shape)
                            print('t_downsample: ',t_downsample)
                            print('obs_downsampled.shape: ',obs_downsampled.shape)

                        # interpolation = fun_interpolation(obs_downsampled.squeeze().to(args.device),t_downsample.squeeze().to(args.device),verbose=True) #Original 
                        interpolation = fun_interpolation(obs_downsampled.to(args.device),t_downsample.squeeze().to(args.device),verbose=False) #Original 
                        if args.integral_c == 'spline': # To pass a C as cubic spline
                            c = lambda x: interpolation.spline_interpolation(x).squeeze()
                        elif args.integral_c == 'linear': # 
                            c = lambda x: interpolation.linear_interpolation(x).to(args.device)
                        elif args.integral_c == 'step': # 
                            c = lambda x: interpolation.step_interpolation(x).to(args.device)
                        
                elif args.integral_c is None:
                    c=None

                if verbose: print('[in val] using Integral_attention_solver_multbatch')
                if args.support_tensors is False:
                    z_test = Integral_attention_solver_multbatch(
                            # ts_val.to(device),
                            # obs_val[0].unsqueeze(0).to(args.device),
                            # sampling_points = ts_val.size(0),
                            ts_test[0].squeeze().to(device), # Because it has to be a 1D vector
                            obs_test[:,0,:].unsqueeze(1).to(args.device), # This should should the first point for each curve in the batch. For the multbatch [num_batch, 1 time point, num_dim]. For single batch [1 time point,num_dim]
                            c=c,
                            sampling_points = ts_test[0].squeeze().size(0),
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            smoothing_factor=args.smoothing_factor,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            use_support=False,
                            ).solve()
                else:
                    z_test = Integral_attention_solver_multbatch(
                            # ts_val.to(device),
                            # obs_[0].unsqueeze(0).to(args.device),
                            # sampling_points = dummy_times.size(0),
                            # support_tensors=dummy_times.to(device),
                            ts_test[0].squeeze().to(device), # Because it has to be a 1D vector
                            obs_test[:,0,:].unsqueeze(1).to(args.device), # This should should the first point for each curve in the batch. For the multbatch [num_batch, 1 time point, num_dim]. For single batch [1 time point,num_dim]
                            c=c,
                            # sampling_points = dummy_times.size(0),
                            sampling_points = dummy_times.squeeze().size(0),
                            support_tensors=dummy_times.squeeze().to(args.device),
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            smoothing_factor=args.smoothing_factor,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            output_support_tensors=args.output_support_tensors
                            ).solve()

                if args.combine_points is True:
                    z_test = z_test[:,real_idx,:]


                        # if args.combine_points is True:
                        #     z_val = z_val[real_idx,:]
                
                # z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                if len(z_test.shape)<3: #because if it is single curve per batch, this won't return the batch=1 dimension.
                    z_test = z_test[None,:]
                    
                obs_to_print = to_np(obs_test)
                time_to_print = to_np(ts_test)
                if args.support_tensors is True or args.support_test is True:
                    dummy_times_to_print = to_np(dummy_times)
                    if verbose: print('dummy_times_to_print.shape: ',dummy_times_to_print.shape)
                else: 
                    dummy_times = ts_test[0,:].clone().cpu()
                    # dummy_times = torch.linspace(0,1,1000)
                    dummy_times_to_print = to_np(dummy_times)
                    if verbose: print('dummy_times_to_print.shape: ',dummy_times_to_print.shape)
                    
                real_idx = np.argwhere(np.isin(dummy_times,ts_test[0,:].cpu(),invert=False)).flatten() #returns the indexes of the dummy frames. So dummy_idxs and 
                if verbose: 
                    print('real_idx: ',real_idx)
                    print('time_to_print: ',time_to_print)
                    print('dummy_times: ',dummy_times)
                z_real_to_print = to_np(z_test[:,real_idx,:])
                z_all_to_print = to_np(z_test)
                
                if verbose: 
                    print('real_idx: ',real_idx)
                    print('obs_to_print.shape: ',obs_to_print.shape)
                    print('time_to_print.shape: ',time_to_print.shape)
                    print('z_real_to_print.shape: ',z_real_to_print.shape)
                    print('z_all_to_print.shape: ',z_all_to_print.shape)
                # if len(z_to_print.shape)<3: #because if it is single curve per batch, this won't return the batch=1 dimension.
                #     print('[before] z_to_print.shape: ',z_to_print.shape)
                #     z_to_print = z_to_print[None,:]

                # frames_to_drop=None
                # path_to_save_plots= None
                epoch=None
                if obs_to_print.shape[2]<args.num_dim_plot: args.num_dim_plot=obs_to_print.shape[2]

                plot_dim_vs_time(obs_to_print[0,:], time_to_print[0,:], z_real_to_print[0,:], dummy_times_to_print, z_all_to_print[0,:], frames_to_drop, path_to_save_plots, name='plot_test_ndim_epoch', epoch=batch_idx, args=args)

                #make dict with variables to be saved
                outputs_to_save = {
                        'batch_idx': batch_idx ,
                        'obs_to_print': obs_to_print[0,:],
                        'time_to_print' : time_to_print[0,:],
                        'dummy_times_to_print' : dummy_times_to_print,
                        'z_all_to_print' : z_all_to_print[0,:],
                        'frames_to_drop' : frames_to_drop,
                }
                torch.save(outputs_to_save, os.path.join(path_to_save_plots,'output_batch_' + str(batch_idx) + '.pt'))
                
                if args.integral_c is not None: # To pass a C# Now print 'c' 
                    if args.integral_c == 'cte_2nd_half' or args.integral_c == 'cte_2nd_half_shifted':
                        obs_downsampled = obs_test.detach().clone()
                        t_downsample = ts_test[0].detach().clone()
                    obs_to_print = to_np(obs_downsampled)
                    time_to_print = to_np(t_downsample)
                    z_real_to_print = to_np(c(t_downsample))
                    
                    # dummy_times = torch.linspace(0,1,1000)
                    # dummy_times_to_print = to_np(dummy_times)
                    
                    z_all_to_print = to_np(c(dummy_times.to(args.device)))
                    if len(z_real_to_print.shape)<3: #because if it is single curve per batch, this won't return the batch=1 dimension.
                        z_real_to_print = z_real_to_print[None,:]
                        z_all_to_print = z_all_to_print[None,:]

                    if verbose: 
                        print('Plotting c')
                        print('obs_to_print.shape: ',obs_to_print.shape)
                        print('time_to_print.shape: ',time_to_print.shape)
                        print('z_real_to_print.shape: ',z_real_to_print.shape)
                        print('z_all_to_print.shape: ',z_all_to_print.shape)

                    plot_dim_vs_time(obs_to_print[0,:], time_to_print, z_real_to_print[0,:], dummy_times_to_print, z_all_to_print[0,:], frames_to_drop, path_to_save_plots, name='plot_c_ndim_batch', epoch=batch_idx, args=args)
                batch_idx+=1                
                                                                 
def Full_experiment_AttentionalIE_PDE_Brain(model, Encoder, Decoder, Data, time_seq, index_np, mask, times, args, extrapolation_points): # experiment_name, plot_freq=1):
    # scaling_factor=1
    
    
    #metadata for saving checkpoints
    if args.model=='nie': 
        str_model_name = "nie"
    elif args.model=='node': 
        str_model_name = "node"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)

    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else: 
            num_experiments = [int(i[3:]) for i in txt]
            num_experiments = np.array(num_experiments).max()
         # -- logger location
        #writer = SummaryWriter(os.path.join(path_to_experiment,'run'+str(num_experiments+1)))
        #print('writer.log_dir: ',writer.log_dir)
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)
            
        #with open(os.path.join(writer.log_dir,'commandline_args.txt'), 'w') as f:
        #    for key, value in args.__dict__.items(): 
        #        f.write('%s:%s\n' % (key, value))



    obs = Data
    times = time_seq
    
    
    All_parameters = model.parameters()
    
    if Encoder is not None:
        All_parameters = list(All_parameters)+list(Encoder.parameters())
    if Decoder is not None:
        All_parameters = list(All_parameters)+list(Decoder.parameters())
     
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0,last_epoch=-1)# Emanuele's version
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,last_epoch=-1) #My first version
    #scheduler = LRScheduler(optimizer,patience = 20,min_lr=1e-12,factor=0.1)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0,last_epoch=-1)
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        
        if Encoder is None or Decoder is None:
            model, optimizer, scheduler, pos_enc, pos_dec, f_func = load_checkpoint(path, model, optimizer, scheduler, None, None,  None)
        else:
            G_NN, optimizer, scheduler, model, Encoder, Decoder = load_checkpoint(path, None, optimizer, scheduler, model, Encoder, Decoder)

    
    if args.eqn_type=='Navier-Stokes':
        spatial_domain_xy = torch.meshgrid([torch.linspace(0,1,args.n_points) for i in range(2)])
        
        x_space = spatial_domain_xy[0].flatten().unsqueeze(-1)
        y_space = spatial_domain_xy[1].flatten().unsqueeze(-1)
        
        spatial_domain = torch.cat([x_space,y_space],-1)
    
    
    if args.mode=='train':
        #lr_scheduler = LRScheduler(optimizer,patience = 50,min_lr=1e-5,factor=0.1)
        early_stopping = EarlyStopping(patience=1000,min_delta=0)

        # Loss_print = []
        # Val_Loss = []
        all_train_loss=[]
        all_val_loss=[]
        
            
        Data_splitting_indices = Train_val_split(np.copy(index_np),0)
        Train_Data_indices = Data_splitting_indices.train_IDs()
        Val_Data_indices = Data_splitting_indices.val_IDs()
        print('\nlen(Train_Data_indices): ',len(Train_Data_indices))
        print('Train_Data_indices: ',Train_Data_indices)
        print('\nlen(Val_Data_indices): ',len(Val_Data_indices))
        print('Val_Data_indices: ',Val_Data_indices)
        
        # Train Neural IDE
        get_times = Select_times_function(times,extrapolation_points)

        save_best_model = SaveBestModel()
        start = time.time()
        
        split_size = int(args.training_split*obs.size(0))
        
        if args.eqn_type == 'Burgers':
            obs_train = obs[:obs.size(0)-split_size,:,:]
        else:
            obs_train = obs[:obs.size(0)-split_size,:,:,:]
            
        for i in range(args.epochs):
            
            if args.support_tensors is True or args.support_test is True:
                if args.combine_points is True:
                    sampled_tensors,real_idx = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(0, 1))
                    temp_sampled_tensors = sampled_tensors
                    sampled_tensors = sampled_tensors.to(device)
                    #Check if there are duplicates and resample if there are
                    sampled_tensors = torch.cat([times,sampled_tensors])
                    dup=np.array([0])
                    while dup.size != 0:
                        u, c = np.unique(temp_sampled_tensors, return_counts=True)
                        dup = u[c > 1]
                        if dup.size != 0:
                            sampled_tensors,real_idx = torch.sort(torch.FloatTensor(args.sampling_points).uniform_(0, 1))
                            sampled_tensors = sampled_tensors.to(device)
                            sampled_tensors = torch.cat([times,sampled_tensors])
                    dummy_times=sampled_tensors
                    real_idx=real_idx[:times.size(0)]
                if args.combine_points is False:
                        dummy_times = torch.linspace(times[0],times[-1],args.sampling_points)
            
            model.train()
            if Encoder is not None:
                Encoder.train()
            if Decoder is not None:
                Decoder.train()
            
            start_i = time.time()
            print('Epoch:',i)
            # GPUtil.showUtilization()
            counter=0
            train_loss = 0.0
            
            if args.n_batch>1:
                obs_shuffle = obs_train[torch.randperm(obs_train.size(0)),:,:,:]
                
            for j in tqdm(range(0,obs.size(0)-split_size,args.n_batch)):
                
                if args.n_batch==1:
                    Dataset_train = Dynamics_Dataset(obs_train[j,:,:,:],times)
                else:
                    Dataset_train = Dynamics_Dataset(obs_shuffle[j:j+args.n_batch,:,:,:],times,args.n_batch)
                #Dataset_val = Dynamics_Dataset(obs[j-split_size,:,:],times)
                # loader = torch.utils.data.DataLoader(Dataset, batch_size = batch_size)
                # Dataset_val = Val_Dynamics_Dataset(Data,Val_Data_indices,times)

                # For the sampler
                train_sampler = SubsetRandomSampler(Train_Data_indices)
                #valid_sampler = SubsetRandomSampler(Val_Data_indices)

                # loader_val = torch.utils.data.DataLoader(Dataset, batch_size = args.batch_size)

                dataloaders = {'train': torch.utils.data.DataLoader(Dataset_train, sampler=train_sampler,\
                                                                    batch_size = args.n_batch, drop_last=True),
                              }

                train_loader = dataloaders['train']
                #val_loader = dataloaders['val']
                #loader_test = dataloaders['test']

            #for obs_, ts_, ids_ in tqdm(train_loader): 
                obs_, ts_, ids_ = Dataset_train.__getitem__(index_np)#next(iter(train_loader))
                
                obs_ = obs_.to(args.device)
                ts_ = ts_.to(args.device)
                ids_ = torch.from_numpy(ids_).to(args.device)
                # obs_, ts_, ids_ = next(iter(loader))

                ids_, indices = torch.sort(ids_)
                ts_ = ts_[indices]
                ts_ = torch.cat([times[:1],ts_])
                if args.n_batch==1:
                    if Encoder is None:
                        obs_ = obs_[indices,:,:]
                        obs_ = obs_[:,indices,:]
                else:
                    if Encoder is None:
                        obs_ = obs_[:,indices,:,:]
                        obs_ = obs_[:,:,indices,:]
                            
                    
                if args.perturbation_to_obs0 is not None:
                       perturb = torch.normal(mean=torch.zeros(obs_.shape[1]).to(args.device),
                                              std=args.std_noise)#args.perturbation_to_obs0*obs_[:3,:].std(dim=0))
                else:
                    perturb = torch.zeros_like(obs_[0]).to(args.device)
                # print('obs_[:5]: ',obs_[:5])
                # print('ids_[:5]: ',ids_[:5])
                # print('ts_[:5]: ',ts_[:5])

                # print('obs_: ',obs_)
                # print('ids_: ',ids_)
                # print('ts_: ',ts_)

                # obs_, ts_ = obs_.squeeze(1), ts_.squeeze(1)
                if args.n_batch==1:
                    c= lambda x: obs_[:,:,:1].repeat(1,1,args.time_points).unsqueeze(-1).to(device)
                else:
                    if Encoder is not None:
                        c= lambda x: \
                            torch.nn.functional.interpolate(Encoder(obs_[...,:args.initial_frames_num,:]\
                           .requires_grad_(True))\
                           .permute(0,4,1,2,3).float(),
                            size=[args.shapes[1],args.shapes[2],obs_.shape[3]],
                            mode='trilinear')\
                            .permute(0,2,3,4,1)
                    else:
                        c= None
                        
                
                #if args.patches is True:
                if Encoder is not None:
                    y_0 = Encoder(obs_[...,:args.initial_frames_num])[...,:1,:]
                else:
                    y_0 = obs_[:,:,:,:1]
                
                if Encoder is not None or Decoder is not None:
                    accumulation = True
                else:
                    accumulation = False
                
                
                if args.n_batch==1:
                    z_ = Integral_spatial_attention_solver(
                            torch.linspace(0,1,args.time_points).to(device),
                            obs_[:,:,0].unsqueeze(-1).to(args.device),
                            c=c,
                            sampling_points = args.time_points,
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            spatial_integration=True,
                            spatial_domain= spatial_domain.to(device),
                            spatial_domain_dim=2,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            smoothing_factor=args.smoothing_factor,
                            use_support=False,
                            ).solve()
                else:
                    z_ = Integral_spatial_attention_solver_multbatch(
                            torch.linspace(0,1,args.time_points).to(device),
                            y_0.to(args.device),
                            c=c,
                            sampling_points = args.time_points,
                            mask=mask,
                            Encoder = model,
                            max_iterations = args.max_iterations,
                            spatial_integration=True,
                            spatial_domain= spatial_domain.to(device),
                            spatial_domain_dim=2,
                            #lower_bound = lambda x: torch.Tensor([0]).to(device),
                            #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                            smoothing_factor=args.smoothing_factor,
                            use_support=False,
                            accumulate_grads=accumulation
                            ).solve()
                
                
                if args.n_batch==1:
                    z_ = z_.view(args.n_points,args.n_points,args.time_points)
                else:
                    z_ = z_.view(args.n_batch,args.n_points,args.n_points,args.time_points,args.dim)

                if Decoder is not None:
                    z_ = z_.squeeze(-1).permute(0,3,1,2)
                    z_ = Decoder(z_.requires_grad_(True)).permute(0,2,3,1)
                else:
                    z_ = z_.view(args.n_batch,Data.shape[1],Data.shape[2],args.time_points)
                if args.initial_t is False:
                    obs_ = obs_[:,:,:,1:]
                     
                #loss_ts_ = get_times.select_times(ts_)[1]
                loss = F.mse_loss(z_, obs_.detach()) #Original 
                # print('z_[:,:].to(args.device): ',z_[:,:].to(args.device))
                # print('obs_.to(args.device).detach()[:,:]: ',obs_.to(args.device).detach()[:,:])
                # loss = F.mse_loss(z_[:,:].to(args.device), obs_.to(args.device).detach()[:,:]) #Original 

                
                # ###############################
                # Loss_print.append(to_np(loss))
                # ###############################

                optimizer.zero_grad()
                loss.backward()#(retain_graph=True)
                optimizer.step()

                # n_iter += 1
                counter += 1
                train_loss += loss.item()
                
            if i>15 and args.lr_scheduler == 'CosineAnnealingLR':
                scheduler.step()
                
                
            train_loss /= counter
            all_train_loss.append(train_loss)
            if  split_size==0 and args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
                   
            del train_loss, loss, obs_, ts_, z_, ids_

            ## Validating
                
            model.eval()
            if Encoder is not None:
                Encoder.eval()
            if Decoder is not None:
                Decoder.eval()
                
            with torch.no_grad():

                    #Only do this if there is a validation dataset
                
                val_loss = 0.0
                counter = 0
                if split_size>0:
                    # for images, _, _, _, _ in tqdm(val_loader):   # frames, timevals, angular_velocity, mass_height, mass_xpos
                    for j in tqdm(range(obs.size(0)-split_size,obs.size(0),args.n_batch)):
                        
                        valid_sampler = SubsetRandomSampler(Train_Data_indices)
                        if args.n_batch==1:
                            Dataset_val = Dynamics_Dataset(obs[j,:,:,:],times)
                        else:
                            Dataset_val = Dynamics_Dataset(obs[j:j+args.n_batch,:,:,:],times,args.n_batch)
                        
                        val_loader = torch.utils.data.DataLoader(Dataset_val, sampler=valid_sampler,\
                                                                 batch_size = args.n_batch, drop_last=True)
                    
                    #for obs_val, ts_val, ids_val in tqdm(val_loader):
                        obs_val, ts_val, ids_val = Dataset_val.__getitem__(index_np)#next(iter(val_loader))
                        obs_val = obs_val.to(args.device)
                        ts_val = ts_val.to(args.device)
                        
                        ids_val = torch.from_numpy(ids_val).to(args.device)

                        ids_val, indices = torch.sort(ids_val)
                        # print('indices: ',indices)
                        if args.n_batch ==1:
                            if Encoder is None:
                                obs_val = obs_val[indices,:,:]
                                obs_val = obs_val[:,indices,:]
                        else:
                            if Encoder is None:
                                obs_val = obs_val[:,indices,:,:]
                                obs_val = obs_val[:,:,indices,:]

                        ts_val = ts_val[indices]
                                             

                        #Concatenate the first point of the train minibatch
                        # obs_[0],ts_
                        # print('\n In validation mode...')
                        # print('obs_[:5]: ',obs_[:5])
                        # print('ids_[:5]: ',ids_[:5])
                        # print('ts_[:5]: ',ts_[:5])
                        # print('ts_[0]:',ts_[0])

                        ## Below is to add initial data point to val
                        #obs_val = torch.cat((obs_[0][None,:],obs_val))
                        #ts_val = torch.hstack((ts_[0],ts_val))
                        #ids_val = torch.hstack((ids_[0],ids_val))

                        # obs_val, ts_val, ids_val = next(iter(loader_val))
                        # print('obs_val.shape: ',obs_val.shape)
                        # print('ids_val: ',ids_val)
                        # print('ts_val: ',ts_val)

                        # obs_val, ts_val = obs_val.squeeze(1), ts_val.squeeze(1)
                        if args.n_batch==1:
                            c= lambda x: obs_val[:,:,:1].repeat(1,1,args.time_points).unsqueeze(-1).to(device)
                        else:
                            if Encoder is not None:
                                c= lambda x: \
                                    torch.nn.functional.interpolate(Encoder(obs_val[...,:args.initial_frames_num,:])\
                                   .permute(0,4,1,2,3).float(),
                                    size=[args.shapes[1],args.shapes[2],obs_val.shape[3]],
                                    mode='trilinear')\
                                    .permute(0,2,3,4,1)
                            else:
                                c=None
                        
                        if Encoder is not None:
                            y_0 = Encoder(obs_val[...,:args.initial_frames_num,:])[...,:1,:]
                        else:
                            y_0 = obs_val[:,:,:,:1]
                            
                    
                        if args.n_batch==1:
                            z_val = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_val[:,:,0].unsqueeze(-1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(device),
                                    spatial_domain_dim=2,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()
                        else:
                            z_val = Integral_spatial_attention_solver_multbatch(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    y_0.to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(device),
                                    spatial_domain_dim=2,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=False
                                    ).solve()
                          
                        if args.n_batch==1:
                            z_val = z_val.view(args.n_points,args.n_points,args.time_points)
                        else:
                            z_val = z_val.view(args.n_batch,args.n_points,args.n_points,args.time_points,args.dim)

                        if Decoder is not None:
                            z_val = z_val.squeeze(-1).permute(0,3,1,2)
                            z_val = Decoder(z_val).permute(0,2,3,1)
                        else:
                            z_val = z_val.view(args.n_batch,Data.shape[1],Data.shape[2],args.time_points)
                        if args.initial_t is False:
                            obs_val = obs_val[:,:,:,1:]
                            
                        #validation_ts_ = get_times.select_times(ts_val)[1]
                        loss_validation = F.mse_loss(z_val, obs_val.detach())
                        # Val_Loss.append(to_np(loss_validation))
                        
                        del obs_val, ts_val, z_val, ids_val

                        counter += 1
                        val_loss += loss_validation.item()
                        
                        del loss_validation

                        #LRScheduler(loss_validation)
                        if args.lr_scheduler == 'ReduceLROnPlateau':
                            scheduler.step(val_loss)
                
                
                else: counter += 1

                val_loss /= counter
                all_val_loss.append(val_loss)
                
                del val_loss

            #writer.add_scalar('train_loss', all_train_loss[-1], global_step=i)
            #if len(all_val_loss)>0:
            #    writer.add_scalar('val_loss', all_val_loss[-1], global_step=i)
            #if args.lr_scheduler == 'ReduceLROnPlateau':
            #    writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], global_step=i)
            #elif args.lr_scheduler == 'CosineAnnealingLR':
            #    writer.add_scalar('Epoch/learning_rate', scheduler.get_last_lr()[0], global_step=i)

            
            with torch.no_grad():
                
                model.eval()
                if Encoder is not None:
                    Encoder.eval()
                if Decoder is not None:
                    Decoder.eval()
                
                if i % args.plot_freq == 0:
                    if obs.size(2)>2:
                        pca_proj = PCA(n_components=2)
                    
                    plt.figure(0, figsize=(8,8),facecolor='w')
                    # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
                    # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
                        
                    plt.plot(np.log10(all_train_loss),label='Train loss')
                    if split_size>0:
                        plt.plot(np.log10(all_val_loss),label='Val loss')
                    plt.xlabel("Epoch")
                    plt.ylabel("MSE Loss")
                    # timestr = time.strftime("%Y%m%d-%H%M%S")
                    #plt.show()
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))

                    for j in tqdm(range(0,obs.size(0),args.n_batch)):
                        if args.n_batch==1:
                            if args.eqn_type == 'Burgers':
                                Dataset_all = Dynamics_Dataset(Data[j,:,:],times)
                            else:
                                Dataset_all = Dynamics_Dataset(Data[j,:,:,:],times)
                        else:
                            if args.eqn_type == 'Burgers':
                                Dataset_all = Dynamics_Dataset(obs[j:j+args.n_batch,:,:],times,args.n_batch)
                            else:
                                Dataset_all = Dynamics_Dataset(obs[j:j+args.n_batch,:,:,:],times,args.n_batch)
                                
                        loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = args.n_batch)

                        obs_test, ts_test, ids_test = Dataset_all.__getitem__(index_np)#next(iter(loader_test))

                        ids_test, indices = torch.sort(torch.from_numpy(ids_test))
                        # print('indices: ',indices)
                        if args.n_batch==1:
                            if Encoder is None:
                                obs_test = obs_test[indices,:,:]
                                obs_test = obs_test[:,indices,:]
                        else:
                            if Encoder is None:
                                obs_test = obs_test[:,indices,:,:]
                                obs_test = obs_test[:,:,indices,:]
                        ts_test = ts_test[indices]
                        # print('obs_test.shape: ',obs_test.shape)
                        # print('ids_test: ',ids_test)
                        # print('ts_test: ',ts_test)


                        obs_test = obs_test.to(args.device)
                        ts_test = ts_test.to(args.device)
                        ids_test = ids_test.to(args.device)
                        # print('obs_test.shape: ',obs_test.shape)
                        # print('ids_test: ',ids_test)
                        # print('ts_test: ',ts_test)
                        # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                        if args.n_batch ==1:
                            c = lambda x: obs_test[:,:,:1].repeat(1,1,args.time_points).unsqueeze(-1).to(device)
                        else:
                            if Encoder is not None:
                                c= lambda x: \
                                    torch.nn.functional.interpolate(Encoder(obs_test[...,:args.initial_frames_num,:]\
                                   .requires_grad_(True))\
                                   .permute(0,4,1,2,3).float(),
                                    size=[args.shapes[1],args.shapes[2],obs.shape[3]],
                                    mode='trilinear')\
                                    .permute(0,2,3,4,1)
                            else:
                                c= None
                                
                        if Encoder is not None:
                            y_0 = Encoder(obs_test[:,:,:,:args.initial_frames_num,:])[...,:1,:]
                        else:
                            y_0 = obs_test[:,:,:,:1]
                                  
                        if args.n_batch==1:
                            z_test = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_test[:,:,0].unsqueeze(-1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(device),
                                    spatial_domain_dim=2,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()
                        else:
                            z_test = Integral_spatial_attention_solver_multbatch(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    y_0.to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= spatial_domain.to(device),
                                    spatial_domain_dim=2,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    accumulate_grads=False
                                    ).solve()
                          



                        plt.figure(1,facecolor='w')

                        z_p = z_test
                        if args.n_batch >1:
                            z_p = z_test[0,:,:]
                        z_p = to_np(z_p)

                        if args.n_batch >1:
                            obs_print = to_np(obs_test[0,:,:])
                        else:
                            obs_print = to_np(obs_test[:,:])                  

                            
                        if Decoder is not None:
                            z_test = z_test.squeeze(-1).permute(0,3,1,2)
                            z_test = Decoder(z_test).permute(0,2,3,1)
                        else:
                            z_test = z_test.view(args.n_batch,Data.shape[1],Data.shape[2],args.time_points)
                        if args.initial_t is False:
                            obs_test = obs_test[:,:,:,1:]

                        z_p = z_test
                        if args.n_batch >1:
                            z_p = z_test[0,:,:,:]
                        z_p = to_np(z_p)

                        if args.n_batch >1:
                            obs_print = to_np(obs_test[0,:,:,:])
                        else:
                            obs_print = to_np(obs_test[:,:,:])

                        plot_reconstruction(obs_print, z_p, None, path_to_save_plots, 'plot_epoch_', i, args)


                        plt.close('all')

            end_i = time.time()
            # print(f"Epoch time: {(end_i-start_i)/60:.3f} seconds")

            
            model_state = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                }


            if split_size>0:
                if Encoder is None:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, model, None, None, None)
                else:
                    save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, None, model, Encoder, Decoder)
            else: 
                if Encoder is None:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, model, None, None, None)
                else:
                    save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, None, model, Encoder, Decoder)

            #lr_scheduler(loss_validation)

            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break

        if args.support_tensors is True or args.support_test is True:
                del dummy_times
                
        end = time.time()
        # print(f"Training time: {(end-start)/60:.3f} minutes")
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),Loss_print)
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),Val_Loss)
        # # plt.savefig('trained.png')
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'trained'+timestr))
        # # plt.show()
        # plt.figure()
        # plt.plot(np.linspace(0,len(Loss_print),len(Loss_print)),np.log10(Loss_print))
        # plt.plot(np.linspace(0,len(Val_Loss),len(Val_Loss)),np.log10(Val_Loss))
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(os.path.join(path_to_save_plots,'final_losses'+timestr))
        # # plt.show()
    elif args.mode=='evaluate':
        print('Running in evaluation mode')
        ## Validating
        model.eval()
        
        t_min , t_max = args.time_interval
        n_points = args.test_points

        
        test_times=torch.sort(torch.rand(n_points),0)[0].to(device)*(t_max-t_min)+t_min
        #test_times=torch.linspace(t_min,t_max,n_points)
        
        #dummy_times = torch.cat([torch.Tensor([0.]).to(device),dummy_times])
        # print('times :',times)
        ###########################################################
        
        with torch.no_grad():
            splitting_size = int(args.training_split*Data.size(0))
            all_r2_scores = []
            all_mse = []
            for j in tqdm(range(Data.size(0)-splitting_size)):
                Dataset_all = Test_Dynamics_Dataset(Data[j,:,:],times)
                loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np)))

                obs_test, ts_test, ids_test = next(iter(loader_test))
                ids_test, indices = torch.sort(ids_test)
                # print('indices: ',indices)
                obs_test = obs_test[indices,:]
                ts_test = ts_test[indices]
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)


                obs_test = obs_test.to(args.device)
                
                ts_test = ts_test.to(args.device)
                ids_test = ids_test.to(args.device)
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)
                # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                
                
                c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_test[:,:1])
                interpolation = NaturalCubicSpline(c_coeffs)
                c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,args.time_points).unsqueeze(-1)
                
                z_test = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_test[0].unsqueeze(1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.linspace(0,1,args.test_points).to(device),
                                    spatial_domain_dim=1,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()
                
                # z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_test)
                obs_print = to_np(obs_test)
                print(z_test.shape)
                new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                plt.figure(figsize=(8,8),facecolor='w')
                z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_p)

#                 plt.figure(j, facecolor='w')
#                 plt.plot(z_p[:,0],z_p[:,1],c='r', label='model')
#                 plt.scatter(z_p[:,0],z_p[:,1],s=10,c='red', label='model')
                
                obs_print = to_np(obs_test[:,:])
                # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                #plt.scatter(obs_print[:,0],obs_print[:,1],label='Data',c='blue', alpha=0.5)
                #plt.xlabel("dim 0")
                #plt.ylabel("dim 1")
                ##plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                #plt.legend()
                
                plt.figure(j, facecolor='w')
                plt.plot(to_np(torch.linspace(0,1,args.test_points)),z_p[:,0],c='green',label='t0')
                plt.scatter(to_np(times),obs_print[:,0],label='Data_t0',c='red', alpha=0.5)
                obs_print = to_np(obs_test[:,:])
                # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                #plt.xlabel("dim 0")
                #plt.ylabel("dim 1")
                #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                plt.legend()
                
                plt.figure(j, facecolor='w')
                plt.plot(to_np(torch.linspace(0,1,args.test_points)),z_p[:,1],c='orange',label='t1')
                plt.scatter(to_np(times),obs_print[:,1],label='Data_t1',c='blue', alpha=0.5)
                obs_print = to_np(obs_test[:,:])
                # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                #plt.xlabel("dim 0")
                #plt.ylabel("dim 1")
                #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                plt.legend()
                
                if args.test_points == obs_test.size(0):
                    _, _, r_value, _, _ = scipy.stats.linregress(z_p[:,:].flatten(),obs_print[:,:].flatten())
                    mse_value = mean_squared_error(z_p[:,:].flatten(),obs_print[:,:].flatten())

                    print('R2:',r_value)
                    print('MSE:',mse_value)

                    all_r2_scores.append(r_value)
                    all_mse.append(mse_value)
            
            if args.test_points == obs_test.size(0):
                print("Average R2:",sum(all_r2_scores)/len(all_r2_scores))
                print("Average MSE:",sum(all_mse)/len(all_mse))
                
            for j in tqdm(range(Data.size(0)-splitting_size,Data.size(0))):
                Dataset_all = Test_Dynamics_Dataset(Data[j,:,:],times)
                loader_test = torch.utils.data.DataLoader(Dataset_all, batch_size = len(np.copy(index_np)))

                obs_test, ts_test, ids_test = next(iter(loader_test))
                ids_test, indices = torch.sort(ids_test)
                # print('indices: ',indices)
                obs_test = obs_test[indices,:]
                ts_test = ts_test[indices]
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)


                obs_test = obs_test.to(args.device)
                ts_test = ts_test.to(args.device)
                ids_test = ids_test.to(args.device)
                # print('obs_test.shape: ',obs_test.shape)
                # print('ids_test: ',ids_test)
                # print('ts_test: ',ts_test)
                # obs_test, ts_test = obs_test.squeeze(1), ts_test.squeeze(1)
                c_coeffs = natural_cubic_spline_coeffs(torch.linspace(0,1,args.n_points).to(device), obs_test[:,:1])
                interpolation = NaturalCubicSpline(c_coeffs)
                c = lambda x: interpolation.evaluate(x[:,0]).repeat(1,args.time_points).unsqueeze(-1)
                    
                z_test = Integral_spatial_attention_solver(
                                    torch.linspace(0,1,args.time_points).to(device),
                                    obs_test[0].unsqueeze(1).to(args.device),
                                    c=c,
                                    sampling_points = args.time_points,
                                    mask=mask,
                                    Encoder = model,
                                    max_iterations = args.max_iterations,
                                    spatial_integration=True,
                                    spatial_domain= torch.linspace(0,1,args.test_points).to(device),
                                    spatial_domain_dim=1,
                                    #lower_bound = lambda x: torch.Tensor([0]).to(device),
                                    #upper_bound = lambda x: x,#torch.Tensor([1]).to(device),
                                    smoothing_factor=args.smoothing_factor,
                                    use_support=False,
                                    ).solve()

                # z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_test)
                obs_print = to_np(obs_test)
                
                new_times = to_np(ts_test)#torch.linspace(0,1,ts_.size(0))

                z_p = z_test#model(obs[0],new_times, return_whole_sequence=True)
                z_p = to_np(z_p)
                
                obs_print = to_np(obs_test[:,:])
                
                plt.figure(j, facecolor='w')
                plt.plot(to_np(torch.linspace(0,1,args.test_points)),z_p[:,0],c='green',label='t0')
                plt.scatter(to_np(times),obs_print[:,0],label='Data_t0',c='red', alpha=0.5)
                obs_print = to_np(obs_test[:,:])
                # plt.scatter(obs_print[:extrapolation_points,0]*scaling_factor,obs_print[:extrapolation_points,1]*scaling_factor,label='Data',c='blue')
                #plt.xlabel("dim 0")
                #plt.ylabel("dim 1")
                #plt.scatter(obs_print[extrapolation_points:,0,0],obs_print[extrapolation_points:,0,1],label='Data extr',c='red')
                plt.legend()
                
                plt.figure(j, facecolor='w')
                plt.plot(to_np(torch.linspace(0,1,args.test_points)),z_p[:,1],c='orange',label='t1')
                plt.scatter(to_np(times),obs_print[:,1],label='Data_t1',c='blue', alpha=0.5)
                obs_print = to_np(obs_test[:,:])

                if args.test_points == obs_test.size(0):      
                    print(scipy.stats.linregress(z_p[:,:].flatten(),obs_print[:,:].flatten()))
                    print(mean_squared_error(z_p[:,:].flatten(),obs_print[:,:].flatten()))
                
                '''
                # Plot the last 20 frames  
                data_to_plot = obs_print[:,:]#*args.scaling_factor #Get the first 10 samples for a test 
                predicted_to_plot = z_p[:,:]*args.scaling_factor
                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                predicted_to_plot = predicted_to_plot.reshape(predicted_to_plot.shape[0],184, 208) # Add the original frame dimesion as input
                data_to_plot = data_to_plot.reshape(data_to_plot.shape[0],184, 208)

                fig,ax = plt.subplots(4,10, figsize=(15,5), facecolor='w')
                c=0
                for idx_row in range (2): 
                    for idx_col in range(10):
                        ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                        ax[2*idx_row,idx_col].axis('off')
                        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                        ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                        ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                        ax[2*idx_row+1,idx_col].axis('off')
                        c+=1
                fig.tight_layout()

                #Plot the R2 and MSE loss between the original data and the predicted overtime. 
                data_to_plot = obs_print[:,:]*args.scaling_factor #Get the first 10 samples for a test 
                predicted_to_plot = z_p[:,:]*args.scaling_factor
                data_to_plot = args.fitted_pca.inverse_transform(data_to_plot)
                predicted_to_plot = args.fitted_pca.inverse_transform(predicted_to_plot)

                all_r2_scores = []
                all_mse_scores = []

                for idx_frames in range(len(data_to_plot)):
                    _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                    all_r2_scores.append(r_value)
                    # print('data_to_plot[idx_frames,:].flatten().shape: ',data_to_plot[idx_frames,:].flatten().shape)
                    # print('predicted_to_plot[idx_frames,:].flatten().shape: ',predicted_to_plot[idx_frames,:].flatten().shape)
                    tmp_mse_loss = mean_squared_error(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
                    all_mse_scores.append(tmp_mse_loss)

                fig,ax = plt.subplots(2,1, figsize=(15,5), sharex=True, facecolor='w')
                ax[0].plot(np.arange(len(all_r2_scores)),all_r2_scores)
                ax[1].plot(np.arange(len(all_mse_scores)),all_mse_scores)
                ax[1].set_xlabel("Frames")
                ax[0].set_ylabel("R2")
                ax[1].set_ylabel("MSE")
                fig.tight_layout()

                print('R2: ',all_r2_scores)
                print('MSE: ',all_mse_scores)
                '''
       
