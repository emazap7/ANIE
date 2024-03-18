import os
import shutil
import torch
import numpy as np
import pickle
import scipy
# import yaml


# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image
from torch.utils.data import Dataset
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from IE_source.integrators import MonteCarlo
mc = MonteCarlo()

from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
# from torchinterp1d import Interp1d

if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_dict_template():
    return {"observed_data": None,
            "observed_tp": None,
            "data_to_predict": None,
            "tp_to_predict": None,
            "observed_mask": None,
            "mask_predicted_data": None,
            "labels": None
            }

def normalize_data(data):
    reshaped = data.reshape(-1, data.size(-1))

    att_min = torch.min(reshaped, 0)[0]
    att_max = torch.max(reshaped, 0)[0]

    # we don't want to divide by zero
    att_max[ att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    return data_norm, att_min, att_max

def display_video(frames, framerate, filename=None):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    
    if filename is not None: anim.save(filename)
    return HTML(anim.to_html5_video())

    
def get_system_definition(name, mode='rb'):
    with open(name, mode=mode) as f:
        return f.read()
    
    
class Train_val_split:
    def __init__(self, IDs,val_size_fraction):
        
        
        IDs = np.random.permutation(IDs)
        # print('IDs: ',IDs)
        self.IDs = IDs
        self.val_size = int(val_size_fraction*len(IDs))
    
    def train_IDs(self):
        train = sorted(self.IDs[:len(self.IDs)-self.val_size])
        # print('len(train): ',len(train))
        # print('train: ',train)
        return train
    
    def val_IDs(self):
        val = sorted(self.IDs[len(self.IDs)-self.val_size:])
        # print('len(val): ',len(val))
        # print('val: ',val)
        return val
    
class Train_val_split2:
    '''
    In this class, the indices are split continuously and 50% bigger than the batch_size to allow random segments
    '''
    def __init__(self, IDs,val_size_fraction, segment_len,segment_window_factor):
        
        #Split the data into len(data)/(1.5*batch_size)
        segment_size = len(IDs)/((1+segment_window_factor)*segment_len)
        bins = len(IDs)/np.ceil(segment_size)
        print(bins)

        bins = np.arange(0,len(IDs),step=np.ceil(bins),dtype=np.dtype(np.int16))[:-1] #Discard the last ID because it might lead to smaller sequence
        print("bins: ",bins)

        IDs = np.random.permutation(bins) 
        print('IDs:', IDs)

        val_size = int(val_size_fraction*len(IDs))
        

        # train_ids = sorted(IDs[:len(IDs)-val_size])
        # print('len(train_ids): ',len(train_ids))
        # print('train_ids: ',train_ids)
        # return train

        # val_ids = sorted(IDs[len(IDs)-val_size:])
        # print('len(val_ids): ',len(val_ids))
        # print('val_ids: ',val_ids)

        
        # IDs = np.random.permutation(IDs)
        # print('IDs: ',IDs)
        self.IDs = IDs
        self.val_size = int(np.ceil(val_size_fraction*len(IDs)))
    
    def train_IDs(self):
        train = sorted(self.IDs[:len(self.IDs)-self.val_size])
        # print('len(train): ',len(train))
        # print('train: ',train)
        return train
    
    def val_IDs(self):
        val = sorted(self.IDs[len(self.IDs)-self.val_size:])
        # print('len(val): ',len(val))
        # print('val: ',val)
        return val
    
class Train_val_split3:
    '''
    In this class, each frame is a new curve
    '''
    def __init__(self, IDs,val_size_fraction, segment_len,segment_window_factor):
        verbose=True
        #Split the data into len(data)/(1.5*batch_size)
        # segment_size = len(IDs)/((1+segment_window_factor)*segment_len)
        # if verbose: print('segment_size: ',segment_size)
        # bins = len(IDs)/np.ceil(segment_size)
        # bins = len(IDs)
        # print(bins)

        # bins = np.arange(0,len(IDs),step=np.ceil(bins),dtype=np.dtype(np.int16))[:-1] #Discard the last ID because it might lead to smaller sequence
        bins = np.arange(len(IDs)-segment_len,dtype=np.dtype(np.int16))[:-1]
        if verbose: print("bins: ",bins)

        IDs = np.random.permutation(bins) 
        if verbose: print('IDs:', IDs)

        val_size = int(val_size_fraction*len(IDs))
        
        self.IDs = IDs
        self.val_size = int(np.ceil(val_size_fraction*len(IDs)))
    
    def train_IDs(self):
        train = sorted(self.IDs[:len(self.IDs)-self.val_size])
        # print('len(train): ',len(train))
        # print('train: ',train)
        return train
    
    def val_IDs(self):
        val = sorted(self.IDs[len(self.IDs)-self.val_size:])
        # print('len(val): ',len(val))
        # print('val: ',val)
        return val

# class MyDataset(Dataset):
#     def __init__(self,frames, timevals, angular_velocity, mass_height, mass_xpos, compression_factor, mode):

#         if compression_factor != 1:
#             #print(data.shape)
#             data = zoom(data, (1,1,compression_factor, compression_factor))
#             #print(data.shape)

#         # self.frames = torch.from_numpy(frames).float()
#         self.frames = frames.astype(np.uint8)
#         self.timevals = torch.from_numpy(timevals).float()
#         self.angular_velocity = torch.from_numpy(angular_velocity).float()
#         self.mass_height = torch.from_numpy(mass_height).float()
#         self.mass_xpos = torch.from_numpy(mass_xpos).float()
#         size, s=300, 1
#         color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

#         #Define here what type of augmentation you want to use, if any
#         self.transform =  transforms.Compose([#transforms.RandomResizedCrop(size=size, scale=(0.5, 1.0)),
#                                               # transforms.RandomHorizontalFlip(),
#                                               # transforms.RandomApply([color_jitter], p=0.8),
#                                               # transforms.RandomGrayscale(p=0.2),
#                                               # GaussianBlur(kernel_size=int(0.1 * size)),
#                                               transforms.ToTensor()])
#                                               # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         self.mode=mode


#     def __getitem__(self, index):
#         print('index: ',index)
#         x = Image.fromarray(self.frames[index]) #Sort of works, but the loss doesn't go down

#         t = self.timevals[index]
#         y1 = self.angular_velocity[index]
#         y2 = self.mass_height[index]
#         y3 = self.mass_xpos[index]

#         if self.mode=='train':
#             if self.transform is not None:
#                 xx = [self.transform(x) for i in range(2)]
#         else:
#             xx = [transforms.ToTensor()(x)]

#         return xx, t, y1,y2,y3

#     def __len__(self):
#         return len(self.frames)

        
# class Dynamics_Dataset(torch.utils.data.Dataset):
class Dynamics_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, Data, times):
        'Initialization'
        self.times = times.float()
        self.Data = Data.float()
        # self.batch_size = batch_size

    def __getitem__(self, index):
        # print('index: ',index)
        # print('self.list_IDs.shape: ',len(self.list_IDs))
        # print('self.Data: ',self.Data)
        # print('self.times: ', self.times)
        ID = index #self.list_IDs[index]
        obs = self.Data[ID]
        t = self.times

        return obs, t, ID, ID #Just adding one more output so I can drop later 
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.times)

class Dynamics_Dataset3(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, Data, times, frames_to_drop, segment_len):
        'Initialization'
        self.times = times.float()
        self.Data = Data.float()
        self.frames_to_drop = frames_to_drop
        self.segment_len=segment_len
        # self.batch_size = batch_size

    def __getitem__(self, index):
        # print('index: ',index)
        # print('self.list_IDs.shape: ',len(self.list_IDs))
        # print('self.Data: ',self.Data)
        # print('self.times: ', self.times)
        ID = index #self.list_IDs[index]
        obs = self.Data[ID,:self.segment_len]
        t = self.times #Because it already set the number of points in the main script
        frames_to_drop = self.frames_to_drop[index]

        return obs, t, ID, frames_to_drop 
    
    def __len__(self):
        'Denotes the total number of points'
        return len(self.times)

class Dynamics_Dataset_LSTM(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, Data, times, output):
        'Initialization'
        self.times = times.float()
        self.Data = Data.float()
        # self.frames_to_drop = frames_to_drop
        self.output = output
        # self.batch_size = batch_size

    def __getitem__(self, index):
        # print('index: ',index)
        # print('self.list_IDs.shape: ',len(self.list_IDs))
        # print('self.Data: ',self.Data)
        # print('self.times: ', self.times)
        ID = index #self.list_IDs[index]
        obs = self.Data[ID]
        output = self.output[ID]
        t = self.times #Because it already set the number of points in the main script
        # frames_to_drop = self.frames_to_drop[index]


        return obs, t, ID, output 
    
    def __len__(self):
        'Denotes the total number of points'
        return len(self.times)
    
class Dynamics_Dataset2(Dataset):
    def __init__(self, Data, times, segment_len, segment_window_factor, frames_to_drop):
        self.times = times.float()
        self.Data = Data.float()
        self.segment_len=segment_len
        self.segment_window_factor = segment_window_factor
        self.frames_to_drop = frames_to_drop
        

    def __getitem__(self, index):
        # Here 'index' identifies the first allowed frame of the sequence
        # print('index: ',index)
        max_index = index+self.segment_window_factor*self.segment_len# 1050 + 1.5*50 - 50
        if max_index>=len(self.Data): max_index=len(self.Data)-self.segment_len #Just in case
        # print('max_index: ',int(max_index))
        
        if int(max_index)==index:
            first_id = index
        else: first_id = np.random.randint(index, int(max_index))
        
        IDs = torch.arange(first_id,first_id+self.segment_len)
        # print('IDs: ',IDs)
        
        
        obs = self.Data[IDs]
        frames_to_drop = self.frames_to_drop[index]
        # t = self.times[IDs]
        t = self.times #Because it already set the number of points in the main script 
        
        # print('IDs: ',IDs)
        # print('t: ',t)

        return obs, t, IDs, frames_to_drop#, index

    def __len__(self):
        return len(self.times)
    
    
class Dynamics_Dataset_Video(Dataset):
    def __init__(self, Data, times, range_segment):
        self.times = times.float()
        self.Data = Data.float()
        self.range_segment=range_segment
        

    def __getitem__(self, index):
        possible_IDs = torch.arange(index,index+self.range_segment)
        print('possible_IDs: ',possible_IDs)
        IDs = possible_IDs[torch.randint(len(possible_IDs))]
        print('IDs: ',IDs)
        
        
        obs = self.Data[IDs]
        t = self.times[IDs]
        
        print('IDs: ',IDs)
        print('t: ',t)

        return obs, t, ID

    def __len__(self):
        return len(self.times)
    
# class Val_Dynamics_Dataset(torch.utils.data.Dataset):
#     def __init__(self, Data,list_IDs, times):

#         self.times = times
#         self.Data = Data 
#         self.list_IDs = list_IDs

#     def __len__(self):
#         return len(self.list_IDs)

#     def __getitem__(self,index):

#         ID = self.list_IDs[index]
#         obs = self.Data[ID]
#         t = self.times[ID]

#         return obs, t, ID
    
class Test_Dynamics_Dataset(torch.utils.data.Dataset):
    def __init__(self, Data, times):

#         self.times = times
#         self.Data = Data 
#         self.list_IDs = list_IDs
        self.times = times.float()
        self.Data = Data.float()

    def __len__(self):
        return len(self.times)

    def __getitem__(self,index):

        ID = index #self.list_IDs[index]
        obs = self.Data[ID]
        t = self.times[ID]

        return obs, t, ID
    
class LRScheduler():

    def __init__(
        self, optimizer, patience=100, min_lr=1e-9, factor=0.1
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
        
    
    def get_last_lr(self):
        last_lr = self.lr_scheduler.get_last_lr()
        return last_lr
        
        
class EarlyStopping():

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
                
class SaveBestModel:

    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

        
    # def __call__(self, current_valid_loss, epoch, model, kernel, ode_func = None):
    def __call__(self, path, current_valid_loss, epoch, model, G_NN = None, kernel=None, F_func = None, f_func=None):
        if current_valid_loss < self.best_valid_loss:
            
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch+1}\n")
            
            if G_NN is not None: G_NN_state = {'state_dict': G_NN.state_dict()}
            if kernel is not None: kernel_state = {'state_dict': kernel.state_dict()}
            if F_func is not None: F_func_state = {'state_dict': F_func.state_dict()}
            if f_func is not None: f_func_state = {'state_dict': f_func.state_dict()}
            
            torch.save(model, os.path.join(path,'model.pt'))
            if G_NN is not None:
                torch.save(G_NN_state, os.path.join(path,'G_NN.pt'))
                if f_func is not None: torch.save(f_func_state, os.path.join(path,'f_func.pt'))
            else:
                if kernel is not None: torch.save(kernel_state, os.path.join(path,'kernel.pt'))
                if F_func is not None: torch.save(F_func_state, os.path.join(path,'F_func.pt'))
                if f_func is not None: torch.save(f_func_state, os.path.join(path,'f_func.pt'))
            
            
def load_checkpoint(path, G_NN, optimizer, scheduler, kernel, F_func, f_func=None):
    print('Loading ', os.path.join(path))
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
     
    checkpoint = torch.load(os.path.join(path, 'model.pt'), map_location=map_location)
    if G_NN is not None: 
        start_epoch = checkpoint['epoch']
        offset = start_epoch
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    if G_NN is not None: 
        checkpoint = torch.load(os.path.join(path, 'G_NN.pt'), map_location=map_location)
        G_NN.load_state_dict(checkpoint['state_dict'])
    
    if kernel is not None: 
        checkpoint = torch.load(os.path.join(path, 'kernel.pt'), map_location=map_location)
        kernel.load_state_dict(checkpoint['state_dict'])
    if F_func is not None: 
        checkpoint = torch.load(os.path.join(path, 'F_func.pt'), map_location=map_location)
        F_func.load_state_dict(checkpoint['state_dict'])
    if f_func is not None: 
        checkpoint = torch.load(os.path.join(path, 'f_func.pt'), map_location=map_location)
        f_func.load_state_dict(checkpoint['state_dict'])
    
    if G_NN is not None: 
        return G_NN, optimizer, scheduler, kernel, F_func, f_func
    else: 
        return checkpoint


def load_checkpoint_PDE_Brain(path, G_NN, optimizer, scheduler, kernel, F_func, f_func=None):
    print('Loading ', os.path.join(path))
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
     
    checkpoint = torch.load(os.path.join(path, 'model.pt'), map_location=map_location)
    if G_NN is not None: 
        start_epoch = checkpoint['epoch']
        offset = start_epoch
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    if G_NN is not None: 
        checkpoint = torch.load(os.path.join(path, 'G_NN.pt'), map_location=map_location)
        G_NN.load_state_dict(checkpoint['state_dict'])
    
    if kernel is not None: 
        checkpoint = torch.load(os.path.join(path, 'kernel.pt'), map_location=map_location)
        kernel.load_state_dict(checkpoint['state_dict'])
    if F_func is not None: 
        checkpoint = torch.load(os.path.join(path, 'F_func.pt'), map_location=map_location)
        F_func.load_state_dict(checkpoint['state_dict'])
    if f_func is not None: 
        checkpoint = torch.load(os.path.join(path, 'f_func.pt'), map_location=map_location)
        f_func.load_state_dict(checkpoint['state_dict'])
    
    # if G_NN is not None: 
    return G_NN, optimizer, scheduler, kernel, F_func, f_func
    # else: 
        # return checkpoint
                
class SaveLastState:
    ''' have to be redone. It shoudl be using torch.save'''
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

        
    def __call__(self, path, current_valid_loss, epoch, model, G_NN = None, kernel=None, F_func = None, f_func=None):
        if current_valid_loss < self.best_valid_loss:
            
            self.best_valid_loss = current_valid_loss
            print(f"\nBest training loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch+1}\n")
            
            torch.save(model, os.path.join(path,'model_train.pt'))
            
class Select_times_function():
    def __init__(self,times,max_index):
        self.max_index = max_index
        self.times = times

    def select_times(self,t):
            values = torch.Tensor([])
            indices = []
            for i in range(1,t.size(0)):
                if t[i]<= self.times[self.max_index-1]:
                    values = torch.cat([values,torch.Tensor([t[i]])])
                    indices += [i]
                else:
                    pass
            return values, indices

def to_np(x):
    return x.detach().cpu().numpy()


def normalization(Data):
    for i in range(Data.size(2)):
        di = Data[:,:,i]/torch.abs(Data[:,:,i]).max()
        di = di.unsqueeze(2)
        if i == 0:
            Data_norm = di
        else:
            Data_norm = torch.cat([Data_norm,di],2)
    return Data_norm


class Integral_part():
    def __init__(self,
                 times,
                 F,
                 kernel,
                 y,
                 lower_bound = lambda x: torch.Tensor([0]).to(device),
                 upper_bound = lambda x: x,
                 MC_samplings = 10000,
                 NNs = True):
        
        self.times = times
        self.F = F
        self.kernel = kernel
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.MC_samplings = MC_samplings
        self.NNs = NNs
        self.y = y

        def _interpolate_y(y):# -> torch.Tensor: #inter.interp1d:
            x=self.times.to(device)
            y = y.to(device)
            coeffs = natural_cubic_spline_coeffs(x, y)
            interpolation = NaturalCubicSpline(coeffs)

            def output(point:torch.Tensor):
                return interpolation.evaluate(point.to(device))

            return output
            ######################
        self.interpolated_y = _interpolate_y(y)
    
    def integral(self,x):
        
        def integrand(s):

            if self.NNs is True:
                x_aux = x.repeat(s.size(0)).view(s.size(0),1) 
                y_in=self.F.forward(self.interpolated_y(s).to(torch.float32),x.to(torch.float32)).squeeze(1)
                out = self.kernel.forward(y_in,x_aux.to(torch.float32),s.to(torch.float32))
            else:
                F_part = self.F(self.interpolated_y(s))
                out = torch.bmm(self.kernel(x,s),F_part.view(F_part.size(0),F_part.size(2),1))
              
            return out
        
        ####
        if self.lower_bound(x) < self.upper_bound(x):
            interval = [[self.lower_bound(x),self.upper_bound(x)]]
        else: 
            interval = [[self.upper_bound(x),self.lower_bound(x)]]
        ####

        return mc.integrate(
                      fn= lambda s: torch.sign(self.upper_bound(x)-self.lower_bound(x)).to(device)*integrand(s.to(device))[:,:],
                       dim= 1,
                       N=self.MC_samplings,
                       integration_domain = interval, 
                       out_dim = 0
                       )
    
    def return_whole_sequence(self):
        if self.NNs is True:
            out = torch.cat([self.integral(self.times[i]) for i in range(self.times.size(0))])
        else:
            out = torch.cat([self.integral(self.times[i]).view(1,self.y.size(1)) for i in range(self.times.size(0))])
        return out
    
    
def plot_dim_vs_time(obs_to_print, time_to_print, z_to_print, dummy_times_to_print, z_all_to_print, frames_to_drop, path_to_save_plots, name, epoch, args):
    # obs_to_print[0,:], time_to_print[0,:], z_real_to_print[0,:], dummy_times_to_print, z_all_to_print[0,:]
    verbose=False
    # obs_ = obs_[:-frames_to_drop]
    # ts_ = ts_[:-frames_to_drop]
    if verbose: 
        print('[plot_dim_vs_time] obs_to_print.shape: ',obs_to_print.shape)
        print('[plot_dim_vs_time] time_to_print.shape: ',time_to_print.shape)
        print('[plot_dim_vs_time] args.num_dim_plot: ',args.num_dim_plot)
        print('[plot_dim_vs_time] dummy_times_to_print.shape: ',dummy_times_to_print.shape)
        print('[plot_dim_vs_time] z_all_to_print.shape: ',z_all_to_print.shape)
        
        
    n_plots_x = int(np.ceil(np.sqrt(args.num_dim_plot)))
    n_plots_y = int(np.floor(np.sqrt(args.num_dim_plot)))
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=(10, 10), sharex=True, dpi=100, facecolor='w', edgecolor='k')
    ax=ax.ravel()
    for idx in range(args.num_dim_plot):
        # ax[idx].plot(time_to_print,z_to_print[:,idx],c='r', label='model')
        ax[idx].plot(dummy_times_to_print,z_all_to_print[:,idx],c='r', label='model')
        # plt.scatter(to_np(times)[:extrapolation_points],obs_print[:extrapolation_points,0]*scaling_factor,label='Data',c='blue')
        if frames_to_drop is not None and frames_to_drop>0:
            ax[idx].scatter(time_to_print[:-frames_to_drop],obs_to_print[:-frames_to_drop,idx],label='Data',c='blue', alpha=0.5)
            ax[idx].scatter(time_to_print[-frames_to_drop:],obs_to_print[-frames_to_drop:,idx],label='Hidden',c='green', alpha=0.5)
        else:
            ax[idx].scatter(time_to_print[:],obs_to_print[:,idx],label='Data',c='blue', alpha=0.5)
        ax[idx].set_xlabel("Time")
        ax[idx].set_ylabel("dim"+str(idx))
        #plt.scatter(to_np(times)[extrapolation_points:],obs_print[extrapolation_points:,0,0],label='Data extr',c='red')
        ax[idx].legend()
        # timestr = time.strftime("%Y%m%d-%H%M%S")
    fig.tight_layout()

    if args.mode=='train' or path_to_save_plots is not None:
        plt.savefig(os.path.join(path_to_save_plots, name + str(epoch)))
        plt.close('all')
    else: plt.show()
    
    del obs_to_print, time_to_print, z_to_print, frames_to_drop

def scatter_obs_vs_z_per_dim(obs_to_print, z_all_to_print, frames_to_drop, path_to_save_plots, name, epoch, args):
    # obs_to_print[0,:], time_to_print[0,:], z_real_to_print[0,:], dummy_times_to_print, z_all_to_print[0,:]
    verbose=False
    # obs_ = obs_[:-frames_to_drop]
    # ts_ = ts_[:-frames_to_drop]
    if verbose: 
        print('[plot_dim_vs_time] obs_to_print.shape: ',obs_to_print.shape)
        print('[plot_dim_vs_time] z_all_to_print.shape: ',z_all_to_print.shape)
        print('[plot_dim_vs_time] args.num_dim_plot: ',args.num_dim_plot)
        
        
    n_plots_x = int(np.ceil(np.sqrt(args.num_dim_plot)))
    n_plots_y = int(np.floor(np.sqrt(args.num_dim_plot)))
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')
    ax=ax.ravel()
    for idx in range(args.num_dim_plot):
        if frames_to_drop is not None and frames_to_drop>0:
            ax[idx].scatter(obs_to_print[:-frames_to_drop,idx],z_all_to_print[:-frames_to_drop,idx],label='Data',c='blue', alpha=0.5)
            ax[idx].scatter(obs_to_print[-frames_to_drop:,idx],z_all_to_print[-frames_to_drop:,idx],label='Hidden',c='green', alpha=0.5)
        else:
            ax[idx].scatter(obs_to_print[:],z_all_to_print[:,idx],label='Data',c='blue', alpha=0.5)
        ax[idx].set_xlabel("obs"+str(idx))
        ax[idx].set_ylabel("z"+str(idx))
        ax[idx].legend()
    fig.tight_layout()

    if args.mode=='train' or path_to_save_plots is not None:
        plt.savefig(os.path.join(path_to_save_plots, name + str(epoch)))
        plt.close('all')
    else: plt.show()
    
    del obs_to_print, z_all_to_print, frames_to_drop

    
    
def plot_reconstruction(data_to_plot, predicted_to_plot, frames_to_drop, path_to_save_plots, name, epoch, args):

    # print('data_to_plot.shape: ',data_to_plot.shape)
    num_points_to_plot = 20 if data_to_plot.shape[0]>20 else data_to_plot.shape[0]
    
    n_plots_x = 10 #I want 10 images on the horizontal
    n_plots_y = int(np.ceil(num_points_to_plot/n_plots_x))
    # print('n_plots_x: ',n_plots_x)
    # print('n_plots_y: ',n_plots_y)
    if frames_to_drop is not None and frames_to_drop>0:
        extrapolation = data_to_plot.shape[0]-frames_to_drop
    else: extrapolation = data_to_plot.shape[0]+1
    
    fig,ax = plt.subplots(int(2*n_plots_y),n_plots_x, figsize=(15,5), facecolor='w')
    c=0
    for idx_row in range (n_plots_y): 
        for idx_col in range(n_plots_x):
            if c < data_to_plot.shape[0]:
                # print('idx_row,idx_col,c: ',idx_row,idx_col,c)
                ax[2*idx_row,idx_col].imshow(data_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                ax[2*idx_row,idx_col].axis('off')
                _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[c,:].flatten(), predicted_to_plot[c,:].flatten())
                if c>=extrapolation:
                    ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2), color= 'green', fontweight='bold')
                else: ax[2*idx_row,idx_col].set_title('R2: {:.3f}'.format(r_value**2))
                ax[2*idx_row+1,idx_col].imshow(predicted_to_plot[c,:],vmin=args.range_imshow[0],vmax=args.range_imshow[1])
                ax[2*idx_row+1,idx_col].axis('off')
                c+=1
            ax[2*idx_row+1,idx_col].axis('off')
            ax[2*idx_row,idx_col].axis('off')
    fig.tight_layout()
    if args.mode=='train':
        plt.savefig(os.path.join(path_to_save_plots, name +str(epoch)))
        plt.close('all')
    
    del data_to_plot, predicted_to_plot, frames_to_drop
    
    
def plot_performance_rec(data_to_plot, predicted_to_plot, path_to_save_plots, name, epoch, args): 

    all_r2_scores = []
    all_mse_scores = []

    for idx_frames in range(len(data_to_plot)):
        _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,:].flatten(), predicted_to_plot[idx_frames,:].flatten())
        all_r2_scores.append(r_value)
        # print('data_to_plot[idx_frames,:].flatten().shape: ',data_to_plot[idx_frames,:].flatten().shape
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
    plt.savefig(os.path.join(path_to_save_plots, name +str(epoch)))
    plt.close('all')
    
    del all_r2_scores, all_mse_scores, tmp_mse_loss, data_to_plot, predicted_to_plot
    
    
class fun_interpolation():
    def __init__(self,y,points, verbose=False, given_points=None, start_point=None):
        self.y = y
        self.points = points
        self.verbose = verbose
        self.given_points = given_points
        self.start_point=start_point
        
        if self.verbose: 
            print('self.points: ',self.points)
            print('self.y.shape: ',self.y.shape)
    
    def step_interpolation(self,x):
        values = self.y[:,0,:][:,None,:] # Assign the first point
        if self.verbose: 
            print('values.shape: ',values.shape)

        for i in range(x.size(0)-1):
            all_dist = torch.abs(x[i] - self.points)
            min_idx = all_dist.argmin()
            if self.verbose:
                print('all_dist: ',all_dist)
                print('min_idx: ',all_dist.argmin())
                print('values.shape: ',values.shape)
                print('self.y[:,min_idx,:][:,None,:].shape: ',self.y[:,min_idx,:][:,None,:].shape)
            values = torch.cat((values,self.y[:,min_idx,:][:,None,:]),dim=1)

        return values
    
    def linear_interpolation(self,x):
        x = x.squeeze()
        batch_values = torch.zeros(self.y.size(0),x.size(0),self.y.size(2)) #[batch_size, number_points, dim]
        if self.verbose: 
            print('batch_values.shape: ',batch_values.shape)
            
        t_lin = self.points.repeat(self.y.size(2),1)
        if self.verbose: print('t_lin.shape: ',t_lin.shape)

        for idx_batch in range(batch_values.size(0)):
            x_lin = self.y[idx_batch,:].squeeze().T
            # y = y_orig[:,:-2]
            if self.verbose: print('x_lin.shape: ',x_lin.shape)

            t_in_lin = x.repeat(self.y.size(2),1)
            if self.verbose: print('t_in_lin.shape: ',t_in_lin.shape)

            yq_cpu = Interp1d()(t_lin, x_lin, t_in_lin, None)
            if self.verbose: 
                print('yq_cpu.T.shape: ',yq_cpu.T.shape)
            batch_values[idx_batch,:] = yq_cpu.T            
        
        return batch_values
    
    def spline_interpolation(self,x):
        coeffs = natural_cubic_spline_coeffs(self.points, self.y)
        spline = NaturalCubicSpline(coeffs)
        out = spline.evaluate(x)
        return out

    def cte_2nd_half(self, x, noise=None, c_scaling_factor=1):
        # frame_to_drop = int(self.y.shape[1]/2)
        frame_to_drop = self.given_points
        out = torch.zeros_like(self.y)
        # print('out.shape: ',out.shape)
        # print('torch.cat((self.y[:,:frame_to_drop,:], self.y[:,frame_to_drop-1,:].repeat(self.y[:,frame_to_drop:,:].shape[0],1)),dim=0).shape: ',torch.cat((self.y[:,:frame_to_drop,:], self.y[:,frame_to_drop-1,:].repeat(self.y[:,frame_to_drop:,:].shape[0],1)),dim=0).shape)
        for idx_batch in range(out.shape[0]):
            if self.verbose: 
                print('self.y[idx_batch,frame_to_drop:,:].shape: ',self.y[idx_batch,frame_to_drop:,:].shape)
                print('self.y[idx_batch,:frame_to_drop,:].shape: ',self.y[idx_batch,:frame_to_drop,:].shape)
                print('self.y[idx_batch,frame_to_drop,:]: ',self.y[idx_batch,frame_to_drop,:])
                
                # print('torch.ones_like(self.y[:,:frame_to_drop,:]).shape: ',torch.ones_like(self.y[:,:frame_to_drop,:]).shape)
            # out[idx_batch,:] = torch.cat((self.y[idx_batch,:frame_to_drop,:], self.y[idx_batch,frame_to_drop,:].repeat(self.y[idx_batch,:frame_to_drop,:].shape[0],1)),dim=0)
            if noise is not None:
                # print('adding perturbation: ')
                # perturb = torch.normal(mean=torch.zeros_like(self.y[idx_batch,frame_to_drop:,:]),std=std)#args.perturbation_to_obs0*obs_[:3,:].std(dim=0))
                # obs_orig = obs_.clone()
                out[idx_batch,:] = (torch.cat((self.y[idx_batch,:frame_to_drop,:], self.y[idx_batch,frame_to_drop-1,:].repeat(self.y[idx_batch,frame_to_drop:,:].shape[0],1)),dim=0)+noise[idx_batch,:])/c_scaling_factor
            else: 
                out[idx_batch,:] = torch.cat((self.y[idx_batch,:frame_to_drop,:], self.y[idx_batch,frame_to_drop-1,:].repeat(self.y[idx_batch,frame_to_drop:,:].shape[0],1)),dim=0)/c_scaling_factor
        
        return out

    def cte_2nd_half_shifted(self, x, std=0):
        # frame_to_drop = int(self.y.shape[1]/2)
        frame_to_drop = self.given_points
        out = torch.zeros_like(self.y)
        for idx_batch in range(out.shape[0]):
            if self.verbose: 
                print('self.y[idx_batch,frame_to_drop:,:].shape: ',self.y[idx_batch,frame_to_drop:,:].shape)
                print('self.y[idx_batch,:frame_to_drop,:].shape: ',self.y[idx_batch,:frame_to_drop,:].shape)
                print('self.y[idx_batch,frame_to_drop,:]: ',self.y[idx_batch,frame_to_drop,:])
                
                # print('torch.ones_like(self.y[:,:frame_to_drop,:]).shape: ',torch.ones_like(self.y[:,:frame_to_drop,:]).shape)
            # out[idx_batch,:] = torch.cat((self.y[idx_batch,:frame_to_drop,:], self.y[idx_batch,frame_to_drop,:].repeat(self.y[idx_batch,:frame_to_drop,:].shape[0],1)),dim=0)
            if std>0:
                print('adding perturbation: ')
                perturb = torch.normal(mean=torch.zeros_like(self.y[idx_batch,frame_to_drop:,:]),
                                          std=std)#args.perturbation_to_obs0*obs_[:3,:].std(dim=0))
                out[idx_batch,:] = torch.cat((self.y[idx_batch,:frame_to_drop,:], self.y[idx_batch,frame_to_drop-1,:].repeat(self.y[idx_batch,frame_to_drop:,:].shape[0],1)+perturb),dim=0)
            else: 
                if self.start_point is None:
                    out[idx_batch,:] = torch.cat((self.y[idx_batch,:frame_to_drop,:], self.y[idx_batch,frame_to_drop-1,:].repeat(self.y[idx_batch,frame_to_drop:,:].shape[0],1)),dim=0)
                else: #In this case, shift the given points by 'start_point'
                    out[idx_batch,:] = torch.cat((self.y[idx_batch,self.start_point,:].repeat(self.y[idx_batch,:self.start_point,:].shape[0],1), 
                                                  self.y[idx_batch,self.start_point:self.start_point+frame_to_drop,:], 
                                                  self.y[idx_batch,self.start_point+frame_to_drop-1,:].repeat(self.y[idx_batch,self.start_point+frame_to_drop:,:].shape[0],1)),dim=0)
        
        return out