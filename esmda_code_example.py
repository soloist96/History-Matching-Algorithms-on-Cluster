#!/usr/bin/env python
# coding: utf-8

import numpy as np
# import standard modules
import pandas as pd
import os
import h5py
import sys
import shutil

# Import appuqa
sys.path.append('/data3/Astro/global/unified_uq_framework/')
from appuqa.UQGlobal import *
UQ_USE_MPI = True
from appuqa import SimMaster

from appuqa import UQMain
# Global option for using MPI or not
from mpi4py import MPI


# Import torch modules
import torch
from torch import FloatTensor, cat, from_numpy
from torch.autograd import Variable
from torchsummary import summary
sys.path.append('/data3/Astro/cnn_surrogate_meng/topics/channel_80x80x20_RSC/3D-CNNPCA/src/')
# Import transform net

from model_transform_net import ModelTransformNet
from utils import dot_dict, load_model, gram_matrix



# sys.path.append('/data3/Astro/cnn_surrogate_meng/topics/mcmc/')
# sys.path.append('/data3/Astro/cnn_surrogate_meng/topics/cnn_pca/')

# 3d recurrent R-U-Net path
sys.path.append('/data3/Astro/cnn_surrogate_meng/topics/channel_80x80x20_RSC/')

import argparse
import shutil
from mpi4py import MPI
import time
import numpy as np
import h5py
import unet_uae_filter_16_32_32_64 as vae_util
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL']="3" 
from keras import backend as K
from keras import layers
from keras.models import Model
from keras.optimizers import Adam, SGD
#from batch_cal_well_data import *

import tensorflow as tf 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

# used for 3d cnnpca
from cal_well_data_batch_newest import *
from pca import PCA




class ADGPRSSim(SimMaster):
    def __init__(self, options={}):
        super().__init__(options['dim'], 1)
        self.nx = 80
        self.ny = 80
        self.nz = 20
        self.nc = self.nx * self.ny * self.nz
        
        self.args = {}
        data_dir = '/data3/Astro/cnn_surrogate_meng/topics/channel_80x80x20_RSC/3D-CNNPCA/3d_chan_80x80x20_cond8w/data'
        self.args['m_petrel'] = os.path.join(data_dir,'chan_80x80x20_3000_reals_new_wide2.h5')
        self.args = dot_dict(self.args)
        self.device = 'cuda'

        # read pca construction file
        # self.top_dir = self.model_param['TOP_DIR']
        # self.pca_file = self.top_dir + self.model_param['PCA_FILE']
        print('load pca data ...')
        start = time.time()
        self.read_pca_data()
        print('load pca data takes %f seconds'%(time.time() - start))


        # construct cnn pca model
        print('load cnn pca model ...')
        start = time.time()
        self.transformer = self.get_cnn_pca_fn()
        print('load cnn pca model takes %f seconds'%(time.time() - start))


        # collect hist target data
        print('compute hist target ...')
        start = time.time()
        self.get_hist_target()
        print('compute hist target takes %f seconds'%(time.time() - start))
        
        
        # read pressure processing data
        print('collect dynamic property processing data ...')
        start = time.time()
        self.collect_p_data()
        self.sat_init = 0.1
        self.step_index = [0, 1, 3, 6, 8, 10, 12, 14, 17, 19]

        self.bhp = 300 #unit: bar

        self.pvdo_table = load_pvdo()
        print('collect dynamic property processing data takes %f seconds'%(time.time() - start))

        # read well location data
        # self.prod_well_loc_x, self.prod_well_loc_y = self.read_well_loc()
        

        # construct pressure and saturation proxy model
        print('load proxy model ...')
        start = time.time()
        self.vae_model_p, self.vae_model_sat = self.get_proxy_fn()
        print('load proxy model takes %f seconds'%(time.time() - start))

        #read observed data
        #self.orate_observed, self.wrate_observed, self.orate_std, self.wrate_std = self.read_true_data() 
        self.read_true_data() 
        
        

    def read_pca_data(self):
        data_dir = '/data3/Astro/cnn_surrogate_meng/topics/channel_80x80x20_RSC/3D-CNNPCA/3d_chan_80x80x20_cond8w/data'

        #nr, nz, ny, nx = 3000, 20, 80, 80
        #pca_model = OpcaBase(nc=nx*ny*nz, nr=nr, l=3000)

        dim = 400
        
        
        
        m_petrel = load_model(self.args.m_petrel).astype(np.float32)
        nr, nz, nx, ny = m_petrel.shape
        pca_model = PCA(nc=nx*ny*nz, nr=nr, l=3000)
        pca_model.construct_pca(m_petrel.reshape((nr, nx*ny*nz)).T)

        self.pca_model = pca_model
        return #pca_model

    def collect_p_data(self):

        # def load_data(data_path, array_name_list):
        #     hf_r = h5py.File(data_path, 'r')
        #     result = []
        #     for name in array_name_list:
        #         result.append(np.array(hf_r.get(name)))
        #     hf_r.close()
        #     return result
        # data_dir = '/data3/Astro/cnn_surrogate_meng/topics/channel_40x40x20_new_setup/data'
        # # load training data
        # data_path = os.path.join(data_dir, 'channel_40x40x20_21tsteps_2923_satifying_cases.h5')
        # p_t, logk = load_data(data_path, ['pressure', 'logk'])
        # train_nr = 2500

        # p_t_mean = np.mean(p_t[:train_nr, ...], axis = 0, keepdims = True)
        # p_t = p_t - p_t_mean
        # max_p = np.max(p_t[:train_nr, ...], axis=(0,1,2,3), keepdims = True)
        # min_p = np.min(p_t[:train_nr, ...], axis=(0,1,2,3), keepdims = True)

        hr = h5py.File('/data3/Astro/cnn_surrogate_meng/topics/channel_80x80x20_RSC/p_norm_data.h5', 'r')



        self.p_t_mean = np.array(hr.get('p_t_mean'))
        self.max_p = np.array(hr.get('max_p'))
        self.min_p = np.array(hr.get('min_p'))
        self.p_init = np.array(hr.get('p_init'))

        print('*****' * 10)
        print('p_init is ', self.p_init)
        self.epsilon = 1e-6


        return

    def get_cnn_pca_fn(self):
        device = 'cuda'
        data_dir = '/data3/Astro/cnn_surrogate_meng/topics/channel_80x80x20_RSC/3D-CNNPCA/3d_chan_80x80x20_cond8w/data'
        args = {}
        #args['m_pca'] = os.path.join(data_dir, 'm_pca_test4000_case1.h5')         # New PCA models
        args['m_petrel'] = os.path.join(data_dir, 'chan_80x80x20_3000_reals_new_wide2.h5')         # New PCA models
        args['save_model'] = '/data3/Astro/cnn_surrogate_meng/topics/channel_80x80x20_RSC/3D-CNNPCA/3d_chan_80x80x20_cond8w/weights/fw_weights_case1_sw100.0_rw500.0_hw50.0_9ep.pth' 
        args = dot_dict(args)

        trans_net = ModelTransformNet()
        trans_net = trans_net.to(device)
        trans_net.load_state_dict(torch.load(args.save_model, map_location=torch.device(device)))
        trans_net.eval()


        return trans_net


    def get_hist_target(self):
        
        m_petrel = load_model(self.args.m_petrel).astype(np.float32)
        m_petrel = m_petrel.transpose((0, 1,2,3))
        bins = 200
        hist, bins_target = np.histogram(m_petrel.flatten(), bins)
        cdf_target = hist.cumsum()
        cdf_target = cdf_target / cdf_target.max()
        cdf_target = np.concatenate(([0], cdf_target))

        self.bins = bins
        self.cdf_target = cdf_target
        self.bins_target = bins_target

        return 



    def hist_trans(self, data, cdf_target, bins_target, bins):


        hist, bins = np.histogram(data.flatten(), bins)
        cdf = hist.cumsum()
        cdf = cdf / cdf.max()
        cdf = np.concatenate(([0], cdf))
        # Histogram transformation
        cdf_values = np.interp(data.flatten(), bins, cdf)
        data_ht = np.interp(cdf_values, cdf_target, bins_target)
        data_ht = data_ht.reshape(data.shape)
        return data_ht


    def get_proxy_fn(self):
        input_shape=(20, 80, 80, 1)
        depth = 10
        vae_model_p, _ = vae_util.create_vae(input_shape, depth)
        vae_model_sat, _ = vae_util.create_vae(input_shape, depth)
        

        output_dir = '/data3/Astro/cnn_surrogate_meng/topics/channel_80x80x20_RSC/saved_models/'
        vae_model_p.load_weights(output_dir + 'saved-model-10steps-bs4-lr3e-4-pressure-detrend-hd-500-filter_16_32_32_64-mse-140-412.17.h5')
        vae_model_sat.load_weights(output_dir + 'saved-model-10-steps-bs4-lr3e-4-saturation-hd-500-filter_16_32_32_64-mse-200-454.44.h5')
        return vae_model_p,  vae_model_sat


    
    def cal_rate_multi_layer(self, z_penetrations, prod_well_p_true, prod_well_sat_true, bhp, prod_well_logk):
        wrate, orate = [], []
        z_ref = z_penetrations[0]
        rho_o, rho_w = 786.5064, 1037.836
        #z_penetrations.reverse()
        for z in z_penetrations:
            gh = 9.80665 * 2 # g = 9.8m/s2, dz = 2 meters  19.61333
            wrat, orat = cal_prod_rate(prod_well_p_true[:, z, :], prod_well_sat_true[:, z, :], bhp, prod_well_logk[:,z,:])
            bo, _ = cal_pvto(prod_well_p_true[:, z, :], self.pvdo_table)
            bw, _ = cal_pvtw(prod_well_p_true[:, z, :])
            #ave_density = Sw * rho_w * bw + (1 - Sw) * rho_o * bo
            ave_density = (rho_o / bo * orat + rho_w / bw * wrat) / (orat + wrat + 1e-3)
            bhp = bhp + ave_density * gh * 1e-5 # 900 is the estimated mixed density of water and oil, g=9.8, dz = 5, Pa to psi is 0.000145038
            wrate.append(wrat)
            orate.append(orat)
        return sum(orate), sum(wrate)


    def compute_rate(self, sat_true, p_true, logk):
        num_inj = 3
        well_loc_x = [15, 35, 61, 8, 22, 43, 54, 69]
        well_loc_y = [14, 57, 15, 64, 39, 24, 43, 61]
        prod_well_loc_z = [[x for x in range(12, 20)], [x for x in range(12, 20)], [x for x in range(8)], [x for x in range(8)], [x for x in range(8)]]
        producers_orates_true = []
        producers_wrates_true = []
        for well_ind in range(3, 8):
            #print('Processing producer {}'.format(well_ind - 1))
            prod_well_p_true = p_true[:, :, well_loc_y[well_ind], well_loc_x[well_ind], :]
            # well block saturation
            prod_well_sat_true = sat_true[:, :, well_loc_y[well_ind], well_loc_x[well_ind], :]
            # well block permeability
            prod_well_logk = logk[:, :, well_loc_y[well_ind], well_loc_x[well_ind], :]
            # well BHP
            bhp = np.array([self.bhp] * p_true.shape[-1])



            cur_orate_true, cur_wrate_true = \
                    self.cal_rate_multi_layer(prod_well_loc_z[well_ind - num_inj], prod_well_p_true, prod_well_sat_true, bhp, prod_well_logk)
            orates_true = cur_orate_true
            wrates_true = cur_wrate_true

            producers_orates_true.append(orates_true)
            producers_wrates_true.append(wrates_true)
        return np.array(producers_orates_true), np.array(producers_wrates_true)




    def read_true_data(self):

        #self.num_data = 5
        # hr = h5py.File('/data3/Astro/cnn_surrogate_meng/topics/RML/true_model/observed_orate_data.h5', 'r')
        # orate_observed = np.array(hr.get('case2orate_observed_10%'))
        # orate_std = np.array(hr.get('case2orate_std_10%')) #10%
        # hr.close()
        # return orate_observed[:, 1:1+self.num_data], orate_std[:, 1:1+self.num_data] # shape (18, 6) (18 wells and 6 time step)
        # hr = h5py.File('/data3/Astro/cnn_surrogate_meng/topics/RML/true_model/case800_true_data.h5', 'r')
        # orate_observed = np.array(hr.get('casd800_orate_observed_data'))[:, :self.num_data]
        # wrate_observed = np.array(hr.get('case800_wrate_observed_data'))[:, :self.num_data]
        # hr.close()
        # # perturb observed data
        # orate_std, wrate_std = 0.05 * orate_observed, 0.05 * wrate_observed
        # orate_error = orate_std * np.random.randn(orate_observed.shape[0], orate_observed.shape[1])
        # wrate_error = wrate_std * np.random.randn(wrate_observed.shape[0], wrate_observed.shape[1])

        # return orate_observed + orate_error, wrate_observed + wrate_error, orate_std, wrate_std
        hr = h5py.File('/data3/Astro/cnn_surrogate_meng/topics/channel_80x80x20_RSC/RS/true_data_3d_80x80x20_RS.h5', 'r')
        self.orate_observed = np.array(hr.get('true_orate')).transpose(1, 0, 2)
        self.wrate_observed =  np.array(hr.get('true_wrate')).transpose(1, 0, 2)
        self.orate_std = np.array(hr.get('orate_std')).transpose(1, 0, 2)
        self.wrate_std = np.array(hr.get('wrate_std')).transpose(1, 0, 2)

        nr = self.orate_observed.shape[0]

        self.obs_data = np.concatenate([self.orate_observed.reshape(nr, -1), self.wrate_observed.reshape(nr, -1)], axis = -1).transpose()
        self.obs_std = np.concatenate([self.orate_std.reshape(nr, -1), self.wrate_std.reshape(nr, -1)], axis = -1).transpose()

        print('self.obs_data is ', self.obs_data)
        print('self.obs_std is ', self.obs_std)

        self.nd_ = self.obs_data.shape[0]





        # print('&&&&' * 20)

        # print(self.orate_observed)
        # print(self.wrate_observed)
        # print(self.orate_std)
        # print(self.wrate_std)



    def batch_predict(self, vae_model, test_x, batch_size):
        pred_list = []
        test_batch_size = batch_size
        batch_num = int(np.ceil(test_x.shape[0] / test_batch_size))
        for i in range(batch_num):
            # if i % 10 == 0:
            #     print('processing sub batch {}'.format(i + 1))
            next_batch_idx = min((i+1)*test_batch_size, test_x.shape[0])
            pred_list.append(vae_model.predict(test_x[i*test_batch_size: next_batch_idx, ...]))
        pred = np.concatenate(pred_list, axis = 0)
        return pred
         

    def run_list_sim(self, xi):
        #
        neval = xi.shape[1]

        # save xi 

        np.savetxt('xi_2obs_15%std_10na.txt', xi)
        start1 = time.time()
        dim = 400
        max_, min_ = 1., 0.
        m_pca = self.pca_model.generate_pca_realization(xi, dim).T
        
        # m_pca = self.usig.dot(xi) 
        # print('sim mpca 1 takes %f seconds'%(time.time() - start))
        # start = time.time()
        # m_pca += self.xm
        # print('sim mpca 2 takes %f seconds'%(time.time() - start))
        # start = time.time()
        # m_pca = m_pca.T
        # print('sim mpca 3 takes %f seconds'%(time.time() - start))
        # start = time.time()
        m_pca = m_pca.reshape((neval, 1, self.nz, self.ny, self.nx)).astype('float')
        #print('sim mpca takes %f seconds'%(time.time() - start))
        #start = time.time()
        # Normalize model    
        m_pca = (m_pca - min_) / (max_ - min_) * 255.
        print('sim mpca takes %f seconds'%(time.time() - start1))
        start = time.time()


        batch_size = 5
        num_batch = int(np.ceil(neval / batch_size))
        m_pca_pred = np.zeros_like(m_pca)
        for ib in range(num_batch):
            if (ib+1) % 20==0:
                print(ib+1, num_batch)
            ind0 = ib * batch_size
            ind1 = min(ind0 + batch_size, neval)
            data = m_pca[ind0:ind1, ...]
            if ib == 0:
                print('data shape is ', data.shape)
            m_pca_pred[ind0:ind1, ...] = self.transformer(Variable(torch.from_numpy(data).float()).to(self.device)).data.cpu().numpy()
        

        print('sim cnnpca takes %f seconds'%(time.time() - start))
        start = time.time()

        # Histogram transform
        m_pca_pred_ht = np.zeros_like(m_pca_pred)
        print("Histogram Transform")
        for ib in range(num_batch):
            if (ib+1) % 20==0:
                print(ib+1, num_batch)
            ind0 = ib * batch_size
            ind1 = min(ind0 + batch_size, neval)
            data = m_pca_pred[ind0:ind1, ...]
            m_pca_pred_ht[ind0:ind1, ...] = self.hist_trans(data, self.cdf_target, self.bins_target, self.bins)
        m_pca_pred_ht = m_pca_pred_ht.round()
        
        print('sim histogram transform takes %f seconds'%(time.time() - start))
        start = time.time()


        m_cnn_pca = m_pca_pred_ht.transpose(0, 2, 3, 4, 1)

        start = time.time()
        # pressure prediction
        p_input = m_cnn_pca
        p_pred = self.batch_predict(self.vae_model_p, p_input, batch_size = batch_size)[:, :, :, :, :, 0].transpose(0, 2, 3, 4, 1)
        p_pred = ((self.max_p[:, :, :, :, self.step_index] - self.min_p[:, :, :, :, self.step_index] + self.epsilon) * (p_pred) \
              + self.min_p[:, :, :, :, self.step_index])  + self.p_t_mean[:, :, :, :, self.step_index]
        p_pred = p_pred * 0.0689476
        # saturation prediction
        sat_input = m_cnn_pca
        sat_pred = self.batch_predict(self.vae_model_sat, sat_input, batch_size = batch_size)[:, :, :, :, :, 0].transpose(0, 2, 3, 4, 1)
        # permeability construction
        k = 20 * np.exp(np.log(2000. / 20) * m_cnn_pca)

        print('sim sat and pressure takes %f seconds'%(time.time() - start))
        start = time.time()

        
        # prod well oil rate prediction
        orates, wrates = self.compute_rate(sat_pred, p_pred, k)  # orates shape (4, nr, 10), 4 wells, 10 time steps
        print('sim compute rate takes %f seconds'%(time.time() - start))


        hw = h5py.File('orates_wrates.h5', 'w')
        hw.create_dataset('orates', data = orates)
        hw.create_dataset('wrates', data = wrates)
        hw.close()
        # history matching part
        time_step_data_to_use = np.array([2, 3]) # second time step
        orate_pred, wrate_pred = orates[:, :, time_step_data_to_use].transpose(1, 0, 2), wrates[:, :, time_step_data_to_use].transpose(1, 0, 2)

        data = np.concatenate([orate_pred.reshape(neval, -1), wrate_pred.reshape(neval, -1)], axis = -1).transpose()
        print('###' * 20)
        print('data shape is ', data.shape)
        np.savetxt('data.txt', data)
        return data



##
def main():
    #data_dir = '/data3/Astro/personal/yiminliu/models/3d_chan_60x60x40_cond4wfar_wellsonly/'

    options = {}
    # options['nx'] = 60
    # options['ny'] = 60
    # options['nz'] = 40
    options['dim'] = 400
    # options['target_hist_file'] = os.path.join(data_dir, 'hm/target_hist.h5')
    # options['transform_net'] = os.path.join(data_dir, 'saved_models/cnnpca_chan_less_60x60x40_cond4wfar_ptb40_std1_l400_sw100.0_rw500.0_hw10.0_9ep.model')
    # options['pca_model_facies'] = os.path.join(data_dir, 'pca_model_chan_wellsonly_60x60x40_cond4wfar_l3000.h5')
    # options['pca_model_sand'] = os.path.join(data_dir, 'pca_model_bimodal_chan_logksand_wellsonly_60x60x40_cond4wfar_l3000.h5')
    # options['pca_model_mud'] = os.path.join(data_dir, 'pca_model_bimodal_chan_logkmud_wellsonly_60x60x40_cond4wfar_l3000.h5')
    options['topdir'] = '/data3/Astro/cnn_surrogate_meng/topics/channel_80x80x20_RSC/ESMDA/'
    options['casedir'] = os.path.join(options['topdir'], 'hm')
    # options['model_dir'] = os.path.join(options['topdir'], 'model')
    # options['perm_file'] = 'perm.dat'
    # options['poro_file'] = 'poro.dat'
    # options['sim_input_file'] = 'GPRS.txt'
    # options['num_thread'] = 8
    # options['simulator'] = 'ADGPRS_CentOS6'
    # options['verbose'] = 2
    # options['rates_file'] = 'OUTPUT.rates.txt'
    # options['hist_file'] = os.path.join(options['topdir'], 'true_model', options['rates_file'])
    # options['hist_time'] = np.linspace(100, 1000, num=10)
    # # options['hist_file'] = os.path.join(options['topdir'], 'test_true/run_0', options['rates_file'])
    # # options['hist_time'] = [1., 2.]
    # options['hist_data'] = ['I%d:WIR' % i for i in range(1,3)] + ['P%d:OPR' % i for i in range(1,3)] + ['P%d:WPR' % i for i in range(1,3)]
    # options['rate_std'] = 0.01
    # options['rate_std_min'] = 2.0
    options['ensemble_size'] = 400


    # =========================ES-MDA========================== #
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    my_sim = ADGPRSSim(options=options)
    nd, nr, nm = my_sim.nd_, options['ensemble_size'], options['dim']
    print(nd, nr, nm)
    es_mda = UQMain(nm, nd, nr, alg=UQAlg.ES_MDA, sim_master=my_sim)

    print("Rank %d" % rank)
    if rank == 0:
        if not os.path.exists(options['casedir']):
            os.mkdir(options['casedir'])

        cd = np.diag(my_sim.obs_std.flatten() ** 2)
        cm = np.eye(nm)
        print(cd.shape, cm.shape)

        # na = 4
        # alpha = np.array([9.333, 7.0, 4.0, 2.0])
        na = 10
        alpha = np.array([57.017, 35.0, 25.0, 20.0, 18.0, 15.0, 12.0, 8.0, 5.0, 3.0])

        print('my_sim my_sim.obs_data.flatten() shape is ', my_sim.obs_data.flatten().shape)

        print('nd is ', nd)

        np.random.seed(0)
        xi_prior = np.random.normal(0, 1, (nm, nr))
        print(xi_prior.shape)
        es_mda.solver.input_m_prior(xi_prior)
        es_mda.solver.input_d_obs(my_sim.obs_data.flatten())
        es_mda.solver.input_cm(cm)
        es_mda.solver.input_cd(cd)
        es_mda.solver.set_na(na)
        es_mda.solver.set_alpha(alpha)

    es_mda.solve()

if __name__ == "__main__":
    main()
