import cv2
import numpy as np
import pickle
import torch
import os
import random
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader
from config import categories, categories_train, dataset, data_path, device_ids, dict_dir, layer, vMF_kappa, model_save_dir, init_path, backbone_type, compnet_type, num_mixtures
from config import config as cfg
from model import Net, resnet_feature_extractor
from helpers import getImg, Imgset, imgLoader, getVmfKernels, getCompositionModel, myresize
import torchvision.models as models

def frange(x, y, jump):
	while x < y:
		yield x
		x += jump

def eval_occ_detection(occ_scores,occ_mask_gt):
	locc_scores = []
	occ_scores = -occ_scores
	minval = -100
	maxval = 100
	for thresh in frange(minval, maxval, 0.05):
		occ_mask = (occ_scores < thresh)
		tp = np.sum((occ_mask == True) * (occ_mask_gt == True))
		if tp == 0:
			tp = 1
		tn = np.sum((occ_mask == False) * (occ_mask_gt == False))
		fp = np.sum((occ_mask == True) * (occ_mask_gt == False))
		fn = np.sum((occ_mask == False) * (occ_mask_gt == True))
		fpr = fp / (fp + tn)
		tpr = tp / (tp + fn)
		#dice = np.sum(occ_mask[occ_mask_gt == True]) / (np.sum(occ_mask) + np.sum(occ_mask_gt))
		prec = tp / (tp + fp)
		recall = tp / (tp + fn)
		fscore = 2 * prec * recall / (prec + recall)
		occ_acc = (tp + tn) / (tp + tn + fp + fn)
		locc_scores.append([fpr, tpr, fscore, occ_acc])

	return np.asarray(locc_scores)

def visualize_response_map(rmap,tit,cbarmax=10.0):

	fig, ax = plt.subplots(nrows=1, ncols=1)
	im = ax.imshow(rmap)
	plt.title(tit,fontsize=18)

	if cbarmax!=0:
		im.set_clim(0.0, cbarmax)
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)
		plt.colorbar(im, cax=cax,ticks=np.arange(0, cbarmax*2, cbarmax))
	else:
		im.set_clim(0.0, 7.0)
	plt.axis('off')
	occ_img_name = '/tmp/tmp'+str(random.randint(1,1000000))+'.png'
	plt.savefig(occ_img_name, bbox_inches='tight')
	img = cv2.imread(occ_img_name)
	os.remove(occ_img_name)

	# remove white border top and bottom
	loop = True
	while loop:
		img = img[1:img.shape[0],:,:]
		loop = np.sum(img[0,:,:]==255)==(img.shape[1]*3)
	loop = True
	while loop:
		img = img[0:img.shape[0]-2,:,:]
		loop = np.sum(img[img.shape[0]-1,:,:]==255)==(img.shape[1]*3)

	return img

def plot_ROC(occ_det_all,occ_types,title,filename):
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	colors = ['red', 'blue','green','orange']
	for i in range(len(occ_det_all[0])):
		for jj in range(len(occ_det_all)):
			if len(occ_det_all[jj])>0:
				if jj == 1:
					linestyle = 'dashed'
					method = ''
				else:
					linestyle = 'solid'
					method = ''

				res = occ_det_all[jj][i]
				lab = occ_types[i]
				if lab == '':
					lab = 'objects'
				else:
					lab = occ_types[i]
					lab = lab[1:len(lab)]
				# get eer
				try:
					eer_pos = np.nonzero((res[:, 0] - (1 - res[:, 1])) >= 0)[0][0]
				except:
					print('')
				m = (res[eer_pos, 1] - res[eer_pos - 1, 1]) / (res[eer_pos, 0] - res[eer_pos - 1, 0])
				eer = 1 / (1 + m)
				lab = lab + '_' + method

				fpr=0.2
				idx = np.where(res[:, 0] > fpr)[0][0]
				tabres = res[idx, :]
				print('%s \tFPR: 0.2 \tTPR: %.3f ' % (lab,tabres[1]))

				lab = lab + 'tpr@fpr0.2=%.2f'%(tabres[1])
				'''
				if len(mix_model_counts)>0:
					for m in mix_model_counts[i]:
						lab = lab + '-%d' % (m * 100)
				'''
				ax.plot(res[:, 0], res[:, 1], label=lab,linestyle=linestyle,color=colors[i])
	plt.legend(loc='lower right')
	# plt.show()
	plt.title(title)
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.savefig(filename)
	plt.close()

if __name__=='__main__':

	dataset = 'coco' # pascal3d+, coco
	nsamples_ratio = 1.0 # *100 percent of images used for evaluation
	likely = 0.6
	occ_levels = ['FIVE']#

	bool_mixture_model_bg = True
	bool_load_trained_model = False
	bool_median_filter = True

	plot_out_dir_root = 'results/{}_{}_bgmix{}_trained{}_median{}_backbone{}/'.format(layer, dataset,
																				   bool_mixture_model_bg,
																				   bool_load_trained_model,
																				   bool_median_filter, backbone_type)

	if dataset == 'pascal3d+':
		occ_types = ['_white', '_noise', '_texture', '']#
		colorbar_max= [10.0,10.0,7.0,7.0]
	elif dataset == 'coco':
		occ_types = ['']
		colorbar_max = [7.0]

	occ_likely = []
	for i in range(len(categories)):
		occ_likely.append(likely)

	if not os.path.exists(plot_out_dir_root):
		os.makedirs(plot_out_dir_root)

	if dataset=='coco':
		bool_plot_roc =False 	# only qualitative results, because GT occluder mask not available
	else:
		bool_plot_roc = True 	# qualitative + quantitative evluation of occluder localization

	for occ_level in occ_levels:
		if occ_level == 'ONE':
			area = 0.3
		elif occ_level == 'FIVE':
			area = 0.5
		else:
			area = 0.7
		plot_out_dir_level = plot_out_dir_root + occ_level+'/'
		if not os.path.exists(plot_out_dir_level):
			os.makedirs(plot_out_dir_level)

		OCCLUSION_RESULT_COMP = []

		mix_model_path = init_path + 'mix_model_vmf_pascal3d+_EM_all/'
		bool_mixture_model_bg_local = bool_mixture_model_bg

		if backbone_type=='vgg':
			if layer == 'pool4':
				extractor = models.vgg16(pretrained=True).features[0:24]
			else:
				extractor = models.vgg16(pretrained=True).features
		elif backbone_type=='resnet50' or backbone_type=='resnext':
			extractor = resnet_feature_extractor(backbone_type, layer)
		extractor.cuda(device_ids[0]).eval()
		weights = getVmfKernels(dict_dir, device_ids)
		mix_models = getCompositionModel(device_ids, mix_model_path, layer, categories_train,compnet_type=compnet_type,num_mixtures=num_mixtures)
		model = Net(extractor, weights, vMF_kappa, occ_likely, mix_models, bool_mixture_bg=bool_mixture_model_bg,compnet_type='vmf', num_mixtures=num_mixtures, vc_thresholds=cfg.MODEL.VC_THRESHOLD)

		if bool_load_trained_model:
			if backbone_type=='vgg':
				if layer == 'pool5':
					pth_file = model_save_dir+'vgg_pool5_p3d+/best.pth'

			if device_ids:
				load_dict = torch.load(pth_file, map_location='cuda:{}'.format(device_ids[0]))
			else:
				load_dict = torch.load(pth_file, map_location='cpu')
			model.load_state_dict(load_dict['state_dict'], strict=False)
		model = model.cuda(device_ids[0])
		model.occlusionextract.mix_model = model.mix_model
		if device_ids:
			model = model.cuda(device_ids[0])
			model.eval()

		OCCLUSION_RESULT_CAT = []
		for target_category in ['motorbike']:#categories:
			plot_out_dir = plot_out_dir_level + target_category + '/'
			if not os.path.exists(plot_out_dir):
				os.makedirs(plot_out_dir)

			with torch.no_grad():
				OCCLUSION_RESULT_TYPE = []
				for occ_type in occ_types:

					# Load data
					test_imgs, test_labels, test_masks = getImg('test', categories_train, dataset, data_path, [target_category], occ_level, occ_type, bool_load_occ_mask=True)
					nsamples = np.floor(len(test_imgs)*nsamples_ratio).astype(np.int)
					test_imgs 	= test_imgs[:nsamples]
					test_labels = test_labels[:nsamples]
					test_masks = test_masks[:nsamples]
					test_imgset = Imgset(test_imgs,test_masks,test_labels, imgLoader,bool_square_images=False)
					test_loader = DataLoader(dataset=test_imgset, batch_size=1, shuffle=False)
					eval_results = []

					for id, data in enumerate(test_loader):
						if id ==25:
							input, occ_gt, label = data
							if device_ids:
								input = input.cuda(device_ids[0])
							img_name = test_imgs[id]
							img_orig = cv2.imread(img_name)

							assert(os.path.exists(img_name))
							scores,*_ = model(input)
							pred_lab = np.argmax(scores.detach().cpu().numpy())

							if pred_lab ==label:
								print(img_name)
								score, occ_maps, part_scores = model.get_occlusion(input,label)
								grid = np.random.random((10,10))
								fig, ax = plt.subplots(nrows=1, ncols=1)
								occ_map = occ_maps[0].detach().cpu().numpy()
								part_scores = part_scores[0].detach().cpu().numpy()

								if bool_plot_roc:
									occ_gt = occ_gt[0].numpy() # only first channel
									occ_gt = cv2.resize(occ_gt, (occ_map.shape[1], occ_map.shape[0]))> 0
									img_eval = eval_occ_detection(occ_map,occ_gt) #fpr,tpr,fscore,accuracy
									#try:
									idx = np.where(img_eval[:, 0] > 0.2)[0][0]-1
									#except:
									#	print('')
									res = img_eval[idx,:]
									print('FPR: %.3f TPR: %.3f ACC: %.3f' % (res[0], res[1], res[3]))
									eval_results.append(img_eval)


								if bool_median_filter:
									occ_img_unf = visualize_response_map(occ_map, tit='', cbarmax=colorbar_max[occ_types.index(occ_type)])
									occ_map = cv2.medianBlur(occ_map.astype(np.float32), 3)
								occ_map = (occ_map>0)*occ_map
								occ_img = visualize_response_map(occ_map, tit='',
																 cbarmax=colorbar_max[occ_types.index(occ_type)])
								if occ_type == '':
									occ_name = '0_object'
								elif occ_type == '_white':
									occ_name = '1_white'
								elif occ_type == '_noise':
									occ_name = '2_noise'
								else:
									occ_name = '3_texture'
								faco = img_orig.shape[0] / occ_img.shape[0]
								occ_img_s = cv2.resize(occ_img, (int(occ_img.shape[1] * faco),img_orig.shape[0]))
								out_name = plot_out_dir + '{}_{}_{}{}.png'.format(target_category, id, occ_level, occ_name)
								canvas = np.concatenate((img_orig,occ_img_s), axis=1)
								cv2.imwrite(out_name, canvas)
								plt.close('all')

					if bool_plot_roc:
						OCCLUSION_RESULT_TYPE.append(eval_results)
					print('')

			if bool_plot_roc:
				OCCLUSION_RESULT_CAT.append(OCCLUSION_RESULT_TYPE)
		if bool_plot_roc:
			OCCLUSION_RESULT_COMP.append(OCCLUSION_RESULT_CAT)

		if bool_plot_roc:
			ctmp=[]
			nsamps = np.zeros(len(categories))
			for cid in range(len(categories)):
				otmp=[]
				mtmp=[]
				for oid in range(len(occ_types)):
					mtmp.append(np.mean(np.asarray(OCCLUSION_RESULT_COMP[0][cid][oid]),axis=0))
					nsamps[cid] += len(OCCLUSION_RESULT_COMP[0][cid][oid])
				otmp.append(mtmp)
				ctmp.append(otmp)
				title = 'Occlusion Localization -  Category: {}  Occluded area: {}%'.format(categories[cid], area * 100)
				filename = plot_out_dir_root + 'roc_occ_detection_cat-{}_level-{}.pdf'.format(categories[cid], occ_level)
				try:
					plot_ROC(otmp, occ_types, title, filename)
				except:
					print('')

			nsamps = nsamps/len(occ_types)

			ctmp=np.asarray(ctmp)
			avg_res = np.zeros((ctmp.shape[1],ctmp.shape[2],ctmp.shape[3],ctmp.shape[4]))
			for i in range(ctmp.shape[0]):
				avg_res += ctmp[i]*nsamps[i]
			avg_res = avg_res/np.sum(nsamps)
			tmp = []
			for jj in range(avg_res.shape[0]):
				tmp.append(avg_res[jj,:,:,:])

			title = 'Occlusion Localization -  Occluded area: {}%'.format( area * 100)
			filename = plot_out_dir_root + 'roc_occ_detection_level-{}_all.pdf'.format(occ_level)
			plot_ROC(tmp,occ_types,title,filename)

			with open(plot_out_dir_root+'occlusion_detection_result_{}.pkl'.format(occ_level),'wb') as fh:
				pickle.dump(ctmp,fh)

			print('')



