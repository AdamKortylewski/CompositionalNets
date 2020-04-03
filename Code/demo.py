import torch
import numpy as np
from torch.utils.data import DataLoader
from config import categories, categories_train, dataset, data_path, device_ids, mix_model_path, dict_dir, layer, vMF_kappa, model_save_dir, compnet_type, backbone_type, num_mixtures
from config import config as cfg
from model import Net
from helpers import Imgset, imgLoader, getVmfKernels, getCompositionModel, update_clutter_model
from eval_occlusion_localization import visualize_response_map
import tqdm
import torchvision.models as models
import cv2

###################
# Test parameters #
###################
likely = 0.6  # occlusion likelihood
bool_load_pretrained_model = True
bool_mixture_model_bg = False 	# use maximal mixture model or sum of all mixture models, not so important
bool_multi_stage_model = False 	# this is an old setup

if __name__ == '__main__':

	occ_likely = []
	for i in range(len(categories_train)):
		occ_likely.append(0.6)
	############################
	# Get CompositionalNet Init
	############################
	# get pool5 feature extractor
	extractor = models.vgg16(pretrained=True).features
	extractor.cuda(device_ids[0]).eval()
	weights = getVmfKernels(dict_dir, device_ids)
	mix_models = getCompositionModel(device_ids, mix_model_path, layer, categories_train,compnet_type=compnet_type,num_mixtures=num_mixtures)
	net = Net(extractor, weights, vMF_kappa, occ_likely, mix_models, bool_mixture_bg=bool_mixture_model_bg,compnet_type=compnet_type, num_mixtures=num_mixtures, vc_thresholds=cfg.MODEL.VC_THRESHOLD)
	if device_ids:
		net = net.cuda(device_ids[0])
	pretrained_model = model_save_dir+'vgg_pool5_p3d+/best.pth'
	if device_ids:
		load_dict = torch.load(pretrained_model, map_location='cuda:{}'.format(device_ids[0]))
	else:
		load_dict = torch.load(pretrained_model, map_location='cpu')
	net.load_state_dict(load_dict['state_dict'])
	if device_ids:
		net = net.cuda(device_ids[0])
	updated_clutter = update_clutter_model(net,device_ids)
	net.clutter_model = updated_clutter

	########################################################
	# Classify image and extract occluder locations
	########################################################

	test_imgs = []
	test_imgs.append('demo/17029_0_val2017.jpg')
	test_imgs.append('demo/81819_0_train2017.jpg')
	test_imgs.append('demo/487059_4_train2017.jpg')

	test_imgset = Imgset(test_imgs, [[''],[''],['']], [5,4,8], imgLoader, bool_square_images=False)

	test_loader = DataLoader(dataset=test_imgset, batch_size=1, shuffle=False)

	with torch.no_grad():
		for i, data in enumerate(test_loader):
			#load data
			input, mask, label = data
			if device_ids:
				input = input.cuda(device_ids[0])
			c_label = label.numpy()
			img_name = test_loader.dataset.images[i]
			#classify
			output, *_ = net(input)
			out = output.cpu().numpy().argmax(-1)[0]
			pred_class = categories_train[out]
			print('\nImage {} classified as {}'.format(img_name,pred_class))
			#localize occluder
			score, occ_maps, part_scores = net.get_occlusion(input, label)
			occ_map = occ_maps[0].detach().cpu().numpy()
			occ_map = cv2.medianBlur(occ_map.astype(np.float32), 3)
			occ_img = visualize_response_map(occ_map, tit='',cbarmax=0)
			# concatenate original image and occluder map
			img_orig = cv2.imread(img_name)
			faco = img_orig.shape[0] / occ_img.shape[0]
			occ_img_s = cv2.resize(occ_img, (int(occ_img.shape[1] * faco), img_orig.shape[0]))
			out_name = 'demo/{}_predclass_{}_and_occluder_map.jpg'.format(img_name.split('/')[1].split('.')[0],pred_class)
			canvas = np.concatenate((img_orig, occ_img_s), axis=1)
			cv2.imwrite(out_name, canvas)
			print('Occlusion map written to: {}'.format(out_name))