import sys
import os

import torch
import math
import numpy as np
import scipy.io as io
from NetworkClasses_possion import LSMNetwork
from NetworkClasses_possion import SNN
from coding_and_decoding import poisson_spike, poisson_spike_multi
from torch.utils.tensorboard import SummaryWriter

def train_LSM_SNU(N_step=5,load_model=False,save_model=True,learning_rate=1e-4,iters=20000,gpu='0',possion_num=50):
    ###########Parameters##########
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(comment='LSM_SNU')
    dims = (10,10,10)
    bias = 40
    n_in = dims[0]*dims[1]*dims[2]
    w_mat = 4* 20 * np.array([[3, 6],[-2, -2]])
    steps = 50
    ch = 50 #input num
    best_loss = 10
    counter = 0
    ################################
    #load networks
    print(gpu)
    print(load_model)
    if load_model == False:
        reservoir_network = LSMNetwork(dims, 0.2, w_mat, 7, steps, ch, t_ref=0, ignore_frac=0)
        snu = SNN(batch_size=1,input_size=n_in,hidden_size=256,num_classes=bias,possion_num=possion_num,gpu=gpu)
        snu = snu.to(device)

    else:
        print( "loading model from " + "my_snu_model")
        reservoir_network = torch.load("LSM_SNU/my_lsm_model.pkl",encoding='unicode_escape')
        snu = torch.load("LSM_SNU/my_snu_model.pkl", map_location='cuda:'+gpu)
        snu.device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            snu = snu.to(device)
        for k, m in snu.named_modules():
	        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability

        print("load model successfully")


    #init matrix
    train_in_spikes = np.zeros((ch, steps))
    snu_output_np = np.zeros((N_step,4))

    possion_rate_coding = np.zeros([n_in,possion_num])

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(snu.parameters(), lr=learning_rate)

    loss_store = []
    snu_output_np_store = []
    opts_store = []

    for i in range(iters):
        if i % 50 == 0 and i != 0:
            print(i)
            if save_model == True:
                torch.save(snu,"LSM_SNU/my_snu_model.pkl")
                torch.save(reservoir_network,"LSM_SNU/my_lsm_model.pkl")
                print("saved model successfully")

        #create data
        delt_z = np.random.rand(1) * 100 - 10

        #delt_z = np.array([0])

        # if counter < 15 and counter > 0:
        #     delt_z = np.array([counter+20]).astype(np.float32)
        # else:
        #     delt_z = np.array([counter]).astype(np.float32)

        # if counter == 99:
        #     counter = 0
        # else:
        #     counter += 1

        ipts = np.zeros(5)
        ipts[0] = delt_z[0]
        #opts = np.array([math.sqrt(delt_z)/2,math.sqrt(delt_z)/2,math.sqrt(delt_z),math.sqrt(delt_z)])
        opts = np.array([0.7189618*delt_z, 1.287569*delt_z, 1.3333306*delt_z, 0.5672474*delt_z])
        opts = np.array([opts*(i+1) for i in range(5)])

        opts_ones = np.array([0.7189618, 1.287569, 1.3333306, 0.5672474])
        opts_ones = np.array([opts_ones*(i+1) for i in range(5)])

        #opts = np.log(opts)/np.log(1.04)

        ipts = ipts.astype(np.int32)
        opts = opts.astype(np.int32)

        temp_loss = None



        for iteration in range(N_step):
        #     #transfer input to possion spike
        #     for j in range(N_step):
        #         train_in_spikes[j] = poisson_spike(t=possion_num/10,f=ipts[j])           

            train_in_spikes = poisson_spike_multi(t=possion_num*0.1,f=ipts,dim=10).reshape(-1,50) * 10
            train_in_spikes = torch.from_numpy(train_in_spikes)


            #LSM mode
            reservoir_network.add_input(train_in_spikes)
            rate_coding = reservoir_network.simulate()
            
            #SNU mode
            temp_opts = np.reshape(opts[iteration,:],[1,4]).astype(np.float32)
            temp_opts = torch.from_numpy(temp_opts).float()
            

            temp_opts = temp_opts.to(device)
            snu_output = snu.forward(input=rate_coding,task="LSM",time_window=possion_num)

            #build next step data
            for m in range(4):
                if i == 0:
                    snu_output_np[iteration,m] = 0
                else:
                    snu_output_np[iteration,m] = torch.sum(snu_output[0][40*m:40*m+40]).cpu().item()
            
            

            snu_output_np[iteration,:] /= 10
            ipts,_,__ = np.split(ipts,[1,1],axis = 0) # get delt z
            ipts = np.hstack((ipts,snu_output_np[iteration,:])) #t+1 input


            mu_x = torch.sum(snu_output[0][0:40])/10
            mu_x_opt = temp_opts[0][0]

            mu_x_opt = torch.clamp(mu_x_opt,0,200).float()

            mu_y = torch.sum(snu_output[0][40:80])/10
            mu_y_opt = temp_opts[0][1]

            mu_y_opt = torch.clamp(mu_y_opt,0,200).float()

            sigma_x = torch.sum(snu_output[0][80:120])/10
            sigma_x_opt = temp_opts[0][2]

            sigma_x_opt = torch.clamp(sigma_x_opt,0,200).float()

            sigma_y = torch.sum(snu_output[0][120:160])/10
            sigma_y_opt = temp_opts[0][3]

            sigma_y_opt = torch.clamp(sigma_y_opt,0,200).float()


            #loss func
            if iteration == 0:
                loss = criterion(mu_x, mu_x_opt) + criterion(mu_y, mu_y_opt) + criterion(sigma_x, sigma_x_opt) + criterion(sigma_y, sigma_y_opt)
            else:
                loss = criterion(mu_x, mu_x_opt) + criterion(mu_y, mu_y_opt) + criterion(sigma_x, sigma_x_opt) + criterion(sigma_y, sigma_y_opt) + 0.9*loss


        if counter < 16 and counter > 1:
            snu_output_np -= opts_ones[0:N_step]*20
            snu_output_np = np.clip(snu_output_np,0,400)
            if np.any(snu_output_np == 0):
                snu_output_np = opts_ones[0:N_step]*(counter-1)*(np.random.rand(1)*2)
                


        loss_store.append(loss.item())
        snu_output_np_store.append(snu_output_np.copy())

        print(snu_output_np)

        opts_store.append(opts)
        #save the best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(snu,"LSM_SNU/my_snu_model_best.pkl")
            torch.save(reservoir_network,"LSM_SNU/my_lsm_model_best.pkl")
            print("saved best model successfully with loss",best_loss)
        
        # # Backward and optimize
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        io.savemat('./data1',{'output':np.array(snu_output_np_store)})
        io.savemat('./data2',{'target':np.array(opts_store)})

        print("loss",loss.item())
        with writer:
            writer.add_scalar('loss_lsm_snu', loss, i)
            writer.add_scalars('mu_x',{'mu_x': mu_x,'mu_x_opt': mu_x_opt}, i)
            writer.add_scalars('mu_y',{'mu_y': mu_y,'mu_y_opt': mu_y_opt}, i)
            writer.add_scalars('sigma_x',{'sigma_x': sigma_x,'sigma_x_opt': sigma_x_opt}, i)
            writer.add_scalars('sigma_y',{'sigma_y': sigma_y,'sigma_y_opt': sigma_y_opt}, i)

    
    
    return True

if __name__ == "__main__":

    load_model=False
    save_model=False
    gpu = '0'
    learning_rate = 1e-4
    N_step = 1
    iters = 100
    possion_num = 50

    train_LSM_SNU(N_step,load_model,save_model,learning_rate,iters,gpu,possion_num)