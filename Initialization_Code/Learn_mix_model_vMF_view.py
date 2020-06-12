from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist
import os
import pickle
import numpy as np
from config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, Astride, Apad, Arf,vMF_kappa, layer,init_path, dict_dir, sim_dir, extractor, model_save_dir
from Code.helpers import getImg, imgLoader, Imgset
from torch.utils.data import DataLoader
import cv2
import gc
import matplotlib.pyplot as plt
import scipy.io as sio

dictfile=dict_dir+'dictionary_{}_{}.pickle'.format(layer,vc_num)
print('loading {}'.format(dictfile))
with open(dictfile, 'rb') as fh:
	centers = pickle.load(fh)
#####################
# BEWARE
#####################
bool_pytroch = True
bool_plot_view_p3d=False

mixdir = init_path + 'mix_model_vmf_{}_EM_all/'.format(dataset)
if not os.path.exists(mixdir):
	os.makedirs(mixdir)
occ_level='ZERO'
occ_type=''
spectral_split_thresh=0.1


def learn_mix_model_vMF(category,num_layers = 2,num_clusters_per_layer = 2,frac_data=1.0):

	imgs, labels, masks = getImg('train', [category], dataset, data_path, cat_test, occ_level, occ_type, bool_load_occ_mask=False)
	# similarity matrix
	sim_fname = model_save_dir+'init_vgg/'+'similarity_vgg_pool4/'+'simmat_mthrh045_{}_K{}.pickle'.format(category, 512)

	# Spectral clustering based on the similarity matrix
	with open(sim_fname, 'rb') as fh:
		mat_dis1, _ = pickle.load(fh)

	mat_dis = mat_dis1
	subN = np.int(mat_dis.shape[0]*frac_data)
	mat_dis = mat_dis[:subN,:subN]
	print('total number of instances for obj {}: {}'.format(category, subN))
	N=subN
	img_idx = np.asarray([nn for nn in range(N)])
	imgs = imgs[:N]

	imgset = Imgset(imgs, masks, labels, imgLoader, bool_square_images=False)
	data_loader = DataLoader(dataset=imgset, batch_size=1, shuffle=False)

	r_set = []#[None for nn in range(N)]
	#layer_features 	  =	np.zeros((N,featDim,max_1,max_2),dtype=np.float32)
	for ii,data in enumerate(data_loader):
		if np.mod(ii,100)==0:
			print('{} / {}'.format(ii,N))
		input, mask, label = data
		layer_feature = extractor(input.cuda(device_ids[0]))[0].detach().cpu().numpy()
		iheight,iwidth = layer_feature.shape[1:3]
		lff = layer_feature.reshape(layer_feature.shape[0],-1).T
		lff_norm = lff / (np.sqrt(np.sum(lff ** 2, 1)+1e-10).reshape(-1, 1))+1e-10
		# compute dot product
		tmp = (1-cdist(lff_norm, centers, 'cosine').astype(np.float32))
		# compute vMF likelihood
		tmp = tmp
		tmp = np.exp(vMF_kappa*tmp)
		# reshape such that the spatial position is preserved during learning
		feat_map = tmp.reshape(iheight, iwidth, -1).astype(np.float32).T
		r_set.append(feat_map)

	# num cluster centers
	max_0 = vc_num
	# width
	max_1 = max([r_set[nn].shape[1] for nn in range(N)])
	# height
	max_2 = max([r_set[nn].shape[2] for nn in range(N)])
	print(max_0, max_1, max_2)
	layer_feature_vmf = np.zeros((N, max_0, max_1, max_2), dtype=np.float32)

	for nn in range(N):
		vnum, ww, hh = r_set[nn].shape
		assert (vnum == max_0)
		diff_w1 = int((max_1 - ww) / 2)
		diff_w2 = int(max_1 - ww - diff_w1)
		assert (max_1 == diff_w1 + diff_w2 + ww)
		diff_h1 = int((max_2 - hh) / 2)
		diff_h2 = int(max_2 - hh - diff_h1)
		assert (max_2 == diff_h1 + diff_h2 + hh)
		padded = np.pad(r_set[nn], ((0, 0), (diff_w1, diff_w2), (diff_h1, diff_h2)), 'constant',constant_values=0)
		r_set[nn] = []
		layer_feature_vmf[nn,:,:,:] = padded

	mat_full = mat_dis + mat_dis.T - np.ones((N,N))
	np.fill_diagonal(mat_full, 0)

	mat_sim = 1. - mat_full

	# setup caching variables
	tmp = list()
	tmp.append(np.zeros(mat_sim.shape[0]))
	LABELS 	= list()
	LABELS.append(tmp)
	tmp = list()
	tmp.append(mat_sim)
	MAT		= list()
	MAT.append(tmp)
	tmp = list()
	tmp.append(range(mat_sim.shape[0]))
	IMAGEIDX = list()
	IMAGEIDX.append(tmp)

	# start hierarchical spectral clustering
	FINAL_CLUSTER_ASSIGNMENT=[]
	for i in range(num_layers):
		MAT_SUB = list()
		LABELS_SUB = list()
		IMAGEIDX_SUB = list()

		print('Clustering layer {} ...'.format(i))
		for k in range(np.power(num_clusters_per_layer,i)):
			parent_counter 	= int(np.floor(k / num_clusters_per_layer))
			leaf_counter	= int(np.mod(k,num_clusters_per_layer))
			idx = np.where(LABELS[i][parent_counter] == leaf_counter)[0]
			if len(idx)>spectral_split_thresh*N:
				mat_sim_sub = MAT[i][parent_counter][np.ix_(idx, idx)] # subsample similarity matrix
				MAT_SUB.append(mat_sim_sub)
				IMAGEIDX_SUB.append(np.array(IMAGEIDX[i][parent_counter])[idx])
				cls_solver = SpectralClustering(n_clusters=num_clusters_per_layer, affinity='precomputed', random_state=0)
				cluster_result = cls_solver.fit_predict(mat_sim_sub)
				LABELS_SUB.append(cluster_result)

				print('{} {} {} {}'.format(i,k,sum(cluster_result==0),sum(cluster_result==1)))

				if i==num_layers-1:
					for ff in range(num_clusters_per_layer):
						idx_tmp=IMAGEIDX_SUB[k][cluster_result == ff]
						if len(idx_tmp)>0.02*N:
							FINAL_CLUSTER_ASSIGNMENT.append(np.array(idx_tmp))
			elif len(idx)>0.02*N:
				FINAL_CLUSTER_ASSIGNMENT.append(np.array(IMAGEIDX[i][parent_counter])[idx])
				LABELS_SUB.append([])
				IMAGEIDX_SUB.append([])
				MAT_SUB.append([])
			else:
				LABELS_SUB.append([])
				IMAGEIDX_SUB.append([])
				MAT_SUB.append([])
		MAT.append(MAT_SUB)
		LABELS.append(LABELS_SUB)
		IMAGEIDX.append(IMAGEIDX_SUB)

	mixmodel_lbs = np.ones(len(LABELS[0][0]))*-1
	K=len(FINAL_CLUSTER_ASSIGNMENT) # number of clusters
	for i in range(K):
		mixmodel_lbs[FINAL_CLUSTER_ASSIGNMENT[i]]=i

	mixmodel_lbs = mixmodel_lbs[:N]

	for kk in range(K):
		print('cluster {} has {} samples'.format(kk, np.sum(mixmodel_lbs==kk)))

	alpha = []
	for kk in range(K):
		# get samples for mixture component
		bool_clust = mixmodel_lbs==kk
		bidx = [i for i, x in enumerate(bool_clust) if x]
		num_clusters = vc_num#vmf.shape[1]
		# loop over samples
		for idx in bidx:
			# compute
			vmf_sum = np.sum(layer_feature_vmf[img_idx[idx]], axis=0)
			vmf_sum = np.reshape(vmf_sum, (1, vmf_sum.shape[0], vmf_sum.shape[1]))
			vmf_sum = vmf_sum.repeat(num_clusters, axis=0)+1e-3
			mask = vmf_sum > 0
			layer_feature_vmf[img_idx[idx]] = mask*(layer_feature_vmf[img_idx[idx]]/vmf_sum)

		N_samp = np.sum(layer_feature_vmf[img_idx[bidx]] > 0, axis=0) # stores the number of samples
		mask = (N_samp > 0)
		vmf_sum = mask * (np.sum(layer_feature_vmf[img_idx[bidx]], axis=0) / (N_samp + 1e-5)).astype(np.float32)
		alpha.append(vmf_sum)

	'''
	# ML updates of mixture model and vMF mixture coefficients
	'''
	numsteps = 10
	for ee in range(numsteps):
		changed = 0
		mixture_likeli = np.zeros((subN,K))
		print('\nML Step {} / {}'.format(ee, numsteps))
		changed_samples = np.zeros(subN)
		for nn in range(subN):
			if nn % 100 == 0:
				print('{}'.format(nn))
			#compute feature likelihood
			for kk in range(K):
				like_map = layer_feature_vmf[img_idx[nn]]*alpha[kk]
				likeli = np.sum(like_map, axis=0)+1e-10
				mixture_likeli[nn, kk] = np.sum(np.log(likeli))

			#compute new mixture assigment for feature map
			new_assignment = np.argmax(mixture_likeli[nn, :])
			if new_assignment!=mixmodel_lbs[nn]:
				changed+=1
				changed_samples[nn]=1
			mixmodel_lbs[nn] = new_assignment

		for kk in range(K):
			print('cluster {} has {} samples'.format(kk, np.sum(mixmodel_lbs == kk)))
		print('{} changed assignments'.format(changed))

		#update mixture coefficients here
		for kk in range(K):
			# get samples for mixture component
			bool_clust = mixmodel_lbs == kk
			if np.sum(bool_clust) > 0:
				bidx = [i for i, x in enumerate(bool_clust) if x]
				num_clusters = vc_num  # vmf.shape[1]
				# loop over samples
				for idx in bidx:
					# compute
					vmf_sum = np.sum(alpha[kk]*layer_feature_vmf[img_idx[idx]], axis=0)
					vmf_sum = np.reshape(vmf_sum, (1, vmf_sum.shape[0], vmf_sum.shape[1]))
					vmf_sum = vmf_sum.repeat(num_clusters, axis=0) + 1e-10
					mask = vmf_sum > 0
					layer_feature_vmf[img_idx[idx]] = mask * (alpha[kk]*layer_feature_vmf[img_idx[idx]] / vmf_sum)

				N_samp = np.sum(layer_feature_vmf[img_idx[bidx]] > 0, axis=0)  # stores the number of samples
				mask = (N_samp > 0)
				alpha[kk]= mask*(np.sum(layer_feature_vmf[img_idx[bidx]], axis=0) / (N_samp + 1e-5)).astype(np.float32)
				gc.collect()

		if changed/subN<0.01:
			break

	'''
	# write images of clusters
	'''
	clust_img_dir = os.path.join(mixdir,'clusters_K_AUTO_FEATDIM{}_{}_specific_view/'.format(vc_num,layer))
	if not os.path.exists(clust_img_dir):
		os.makedirs(clust_img_dir)
	for kk in range(K):
		img_ids_kk = img_idx[mixmodel_lbs == kk]
		width = 300
		height = 150
		canvas = np.zeros((0,4*width,3))
		cnt = 0
		for jj in range(4):
			row = np.zeros((height,0,3))
			for ii in range(4):
				if cnt< len(img_ids_kk):
					img = cv2.imread(imgs[img_ids_kk[cnt]])
				else:
					img = np.zeros((height,width,3))
				if not (dataset == 'mnist' or dataset == 'cifar10' or dataset == 'coco'):
					img = cv2.resize(img, (width,height))
				row = np.concatenate((row,img),axis=1)
				cnt+=1
			canvas = np.concatenate((canvas,row),axis=0)
		cv2.imwrite(clust_img_dir+category+'_{}.JPEG'.format(kk),canvas)

	savename = os.path.join(mixdir,'mmodel_{}_K4_FEATDIM{}_{}_specific_view.pickle'.format(category, vc_num, layer))
	with open(savename, 'wb') as fh:
		pickle.dump(alpha, fh)

if __name__=='__main__':
	for category in categories:
		for num_layers in [2]:
			learn_mix_model_vMF(category,num_layers=num_layers,num_clusters_per_layer=2)

	print('DONE')