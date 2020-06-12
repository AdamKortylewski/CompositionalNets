from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from vcdist_funcs import vc_dis_paral, vc_dis_paral_full
import time
import pickle
import os
from config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, Astride, Apad, Arf,vMF_kappa, layer,init_path, dict_dir, sim_dir, extractor
from Code.helpers import getImg, imgLoader, Imgset, myresize
from torch.utils.data import DataLoader
import numpy as np
import math
import torch

paral_num = 10
nimg_per_cat = 5000
imgs_par_cat =np.zeros(len(categories))
occ_level='ZERO'
occ_type=''

print('max_images {}'.format(nimg_per_cat))

if not os.path.exists(sim_dir):
	os.makedirs(sim_dir)

#############################
# BEWARE THIS IS RESET TO LOAD OLD VCS AND MODEL
#############################
with open(dict_dir+'dictionary_{}_{}.pickle'.format(layer,vc_num), 'rb') as fh:
	centers = pickle.load(fh)
##HERE
bool_pytorch = True

for category in categories:
	cat_idx = categories.index(category)
	print('{} / {}'.format(cat_idx,len(categories)))
	imgs, labels, masks = getImg('train', [category], dataset, data_path, cat_test, occ_level, occ_type, bool_load_occ_mask=False)
	imgs=imgs[:nimg_per_cat]
	N=len(imgs)
	##HERE
	imgset = Imgset(imgs, masks, labels, imgLoader, bool_square_images=False,bool_cutout=False,bool_pytorch=bool_pytorch)
	data_loader = DataLoader(dataset=imgset, batch_size=1, shuffle=False)
	savename = os.path.join(sim_dir,'simmat_mthrh045_{}_K{}.pickle'.format(category,vc_num))
	if not os.path.exists(savename):
		r_set = [None for nn in range(N)]
		for ii,data in enumerate(data_loader):
			input, mask, label = data
			if imgs_par_cat[cat_idx]<N:
				with torch.no_grad():
					layer_feature = extractor(input.cuda(device_ids[0]))[0].detach().cpu().numpy()
				iheight,iwidth = layer_feature.shape[1:3]
				lff = layer_feature.reshape(layer_feature.shape[0],-1).T
				lff_norm = lff / (np.sqrt(np.sum(lff ** 2, 1) + 1e-10).reshape(-1, 1)) + 1e-10
				r_set[ii] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
				imgs_par_cat[cat_idx]+=1

		print('Determine best threshold for binarization - {} ...'.format(category))
		nthresh=20
		magic_thhs=range(nthresh)
		coverage = np.zeros(nthresh)
		act_per_pix = np.zeros(nthresh)
		layer_feature_b = [None for nn in range(100)]
		magic_thhs = np.asarray([x*1/nthresh for x in range(nthresh)])
		for idx,magic_thh in enumerate(magic_thhs):
			for nn in range(100):
				layer_feature_b[nn] = (r_set[nn]<magic_thh).astype(int).T
				coverage[idx] 	+= np.mean(np.sum(layer_feature_b[nn],axis=0)>0)
				act_per_pix[idx] += np.mean(np.sum(layer_feature_b[nn],axis=0))
		coverage=coverage/100
		act_per_pix=act_per_pix/100
		best_loc=(act_per_pix>2)*(act_per_pix<15)
		if np.sum(best_loc):
			best_thresh = np.min(magic_thhs[best_loc])
		else:
			best_thresh = 0.45
		layer_feature_b = [None for nn in range(N)]
		for nn in range(N):
			layer_feature_b[nn] = (r_set[nn]<best_thresh).astype(int).T

		print('Start compute sim matrix ... magicThresh {}'.format(best_thresh))
		_s = time.time()

		mat_dis1 = np.ones((N,N))
		mat_dis2 = np.ones((N,N))
		N_sub = 200
		sub_cnt = int(math.ceil(N/N_sub))
		for ss1 in range(sub_cnt):
			start1 = ss1*N_sub
			end1 = min((ss1+1)*N_sub, N)
			layer_feature_b_ss1 = layer_feature_b[start1:end1]
			for ss2 in range(ss1,sub_cnt):
				print('iter {1}/{0} {2}/{0}'.format(sub_cnt, ss1+1, ss2+1))
				_ss = time.time()
				start2 = ss2*N_sub
				end2 = min((ss2+1)*N_sub, N)
				if ss1==ss2:
					inputs = [(layer_feature_b_ss1, nn) for nn in range(end2-start2)]
					para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral)(i) for i in inputs))

				else:
					layer_feature_b_ss2 = layer_feature_b[start2:end2]
					inputs = [(layer_feature_b_ss2, lfb) for lfb in layer_feature_b_ss1]
					para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral_full)(i) for i in inputs))

				mat_dis1[start1:end1, start2:end2] = para_rst[:,0]
				mat_dis2[start1:end1, start2:end2] = para_rst[:,1]

				_ee = time.time()
				print('comptSimMat iter time: {}'.format((_ee-_ss)/60))

		_e = time.time()
		print('comptSimMat total time: {}'.format((_e-_s)/60))

		with open(savename, 'wb') as fh:
			print('saving at: '+savename)
			pickle.dump([mat_dis1, mat_dis2], fh)
