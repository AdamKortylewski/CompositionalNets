import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from config import categories, categories_train, dataset, data_path, device_ids, mix_model_path, dict_dir, layer, vMF_kappa, model_save_dir, compnet_type, backbone_type, num_mixtures
from config import config as cfg
from model import Net
from helpers import getImg, Imgset, imgLoader, getVmfKernels, getCompositionModel, update_clutter_model
from model import resnet_feature_extractor
import tqdm
import torchvision.models as models

###################
# Test parameters #
###################
likely = 0.6  # occlusion likelihood
occ_levels = ['ZERO', 'ONE', 'FIVE', 'NINE'] # occlusion levels to be evaluated [0%,20-40%,40-60%,60-80%]
bool_load_pretrained_model = True
bool_mixture_model_bg = False 	# use maximal mixture model or sum of all mixture models, not so important
bool_multi_stage_model = False 	# this is an old setup

def test(models, test_data, batch_size):
	test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
	print('Testing')
	nclasses = models[0].num_classes
	correct = np.zeros(nclasses)
	total_samples = np.zeros(nclasses)
	scores = np.zeros((0,nclasses))

	with torch.no_grad():
		for i, data in enumerate(tqdm.tqdm(test_loader)):
			input,mask, label = data
			input.requires_grad = False
			if device_ids:
				input = input.cuda(device_ids[0])
			c_label = label.numpy()

			output, *_ = models[0](input)
			out = output.cpu().numpy()

			scores = np.concatenate((scores,out))
			out = out.argmax(1)
			correct[c_label] += np.sum(out == c_label)

			total_samples[c_label] += 1

	for i in range(nclasses):
		if total_samples[i]>0:
			print('Class {}: {:1.3f}'.format(categories_train[i],correct[i]/total_samples[i]))
	test_acc = (np.sum(correct)/np.sum(total_samples))
	return test_acc, scores

if __name__ == '__main__':

	if compnet_type=='bernoulli':
		bool_mixture_model_bg = True

	if bool_load_pretrained_model:
		if backbone_type=='vgg':
			if layer=='pool5':
				if dataset=='pascal3d+':
					vc_dir = model_save_dir+'vgg_pool5_p3d+/'
				elif dataset=='coco':
					vc_dir = model_save_dir+'vgg_pool5_coco/'

		vc_file = vc_dir + 'best.pth'
		outdir = vc_dir
	else:
		vc_dir = model_save_dir+'compnet_{}_{}_{}_initialization/'.format(layer,compnet_type,likely,dataset)
		vc_path = ''
		outdir = vc_dir

	if not os.path.exists(outdir):
		os.makedirs(outdir)
	info = outdir + 'config.txt'

	occ_likely = []
	for i in range(len(categories_train)):
		occ_likely.append(likely)

	############################
	# Get CompositionalNet Init
	############################
	if backbone_type=='vgg':
		if layer=='pool4':
			extractor = models.vgg16(pretrained=True).features[0:24]
		else:
			extractor = models.vgg16(pretrained=True).features
	elif backbone_type=='resnet50' or backbone_type == 'resnet18' or backbone_type == 'resnext' or backbone_type=='densenet':
		extractor = resnet_feature_extractor(backbone_type, layer)

	extractor.cuda(device_ids[0]).eval()

	weights = getVmfKernels(dict_dir, device_ids)
	mix_models = getCompositionModel(device_ids, mix_model_path, layer, categories_train,compnet_type=compnet_type,num_mixtures=num_mixtures)
	net = Net(extractor, weights, vMF_kappa, occ_likely, mix_models, bool_mixture_bg=bool_mixture_model_bg,compnet_type=compnet_type, num_mixtures=num_mixtures, vc_thresholds=cfg.MODEL.VC_THRESHOLD)
	if device_ids:
		net = net.cuda(device_ids[0])
	nets=[]
	nets.append(net.eval())


	if bool_load_pretrained_model:
		if device_ids:
			load_dict = torch.load(vc_file, map_location='cuda:{}'.format(device_ids[0]))
		else:
			load_dict = torch.load(vc_file, map_location='cpu')
		net.load_state_dict(load_dict['state_dict'])
		if device_ids:
			net = net.cuda(device_ids[0])
		updated_clutter = update_clutter_model(net,device_ids)
		net.clutter_model = updated_clutter
		nets = []
		nets.append(net)

	############################
	# Test Loop
	############################
	for occ_level in occ_levels:

		if occ_level == 'ZERO':
			occ_types = ['']
		else:
			if dataset=='pascal3d+':
				occ_types = ['']#['_white','_noise', '_texture', '']
			elif dataset=='coco':
				occ_types = ['']

		for index, occ_type in enumerate(occ_types):
			# load images
			test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, categories, occ_level, occ_type,bool_load_occ_mask=True)
			print('Total imgs for test of occ_level {} and occ_type {} '.format(occ_level, occ_type) + str(len(test_imgs)))
			# get image loader
			test_imgset = Imgset(test_imgs, masks, test_labels, imgLoader, bool_square_images=False)
			# compute test accuracy
			acc,scores = test(models=nets, test_data=test_imgset, batch_size=1)
			out_str = 'Model Name: Occ_level:{}, Occ_type:{}, Acc:{}'.format(occ_level, occ_type, acc)
			print(out_str)

