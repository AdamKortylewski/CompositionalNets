import os
import torch
import cv2
import glob
import torch.nn.functional as F
from config import vc_num, categories, occ_types_vmf, occ_types_bern
from vMFMM import *
from torchvision import transforms
from PIL import Image


def update_clutter_model(net,device_ids,compnet_type='vmf'):
	idir = 'background_images/';
	updated_models = torch.zeros((0,vc_num))
	if device_ids:
		updated_models = updated_models.cuda(device_ids[0])

	if compnet_type=='vmf':
		occ_types=occ_types_vmf
	elif compnet_type=='bernoulli':
		occ_types=occ_types_bern

	for j in range(len(occ_types)):
		occ_type = occ_types[j]
		with torch.no_grad():
			files = glob.glob(idir + '*'+occ_type+'.JPEG')
			clutter_feats = torch.zeros((0,vc_num))
			if device_ids:
				clutter_feats=clutter_feats.cuda(device_ids[0])
			for i in range(len(files)):
				file = files[i]
				img,_ = imgLoader(file,[[]], bool_resize_images=False,bool_square_images=False)
				if device_ids:
					img =img.cuda(device_ids[0])

				feats 	 = net.activation_layer(net.conv1o1(net.backbone(img.reshape(1,img.shape[0],img.shape[1],img.shape[2]))))[0].transpose(1,2)
				feats_reshape = torch.reshape(feats, [vc_num, -1]).transpose(0,1)
				clutter_feats = torch.cat((clutter_feats, feats_reshape))

			mean_activation = torch.reshape(torch.sum(clutter_feats,dim=1),(-1,1)).repeat([1,vc_num])#clutter_feats.sum().reshape(-1,1).numpy().repeat(512,axis=1)#torch.reshape(torch.sum(clutter_feats,axis=1),(-1,1)).repeat(512,axis=1)
			if compnet_type=='bernoulli':
				boo = torch.sum(mean_activation, dim=1) != 0
				mean_vec = torch.mean(clutter_feats[boo]/mean_activation[boo], dim=0)
				updated_models = torch.cat((updated_models, mean_vec.reshape(1, -1)))
			else:
				if occ_type== '_white' or occ_type== '_noise':
					mean_vec = torch.mean(clutter_feats/mean_activation, dim=0)  # F.normalize(torch.mean(clutter_feats,dim=0),p=1,dim=0)#
					updated_models = torch.cat((updated_models, mean_vec.reshape(1, -1)))
				else:
					nc = 5
					model = vMFMM(nc, 'k++')
					model.fit(clutter_feats.cpu().numpy(), 30.0, max_it=150)
					mean_vec = torch.zeros(nc,clutter_feats.shape[1]).cuda(device_ids[0])
					clust_cnt = torch.zeros(nc)
					for v in range(model.p.shape[0]):
						assign = np.argmax(model.p[v])
						mean_vec[assign] += clutter_feats[v]
						clust_cnt[assign]+=1
					mean_vec = (mean_vec.t()/clust_cnt.cuda(device_ids[0])).t()
					updated_models = torch.cat((updated_models, mean_vec))



	return updated_models

def getVmfKernels(dict_dir, device_id):
	vc = np.load(dict_dir, allow_pickle=True)
	vc = vc[:, :, np.newaxis, np.newaxis]
	vc = torch.from_numpy(vc).type(torch.FloatTensor)
	if device_id:
		vc = vc.cuda(device_id[0])
	return vc

def getCompositionModel(device_id,mix_model_path,layer,categories,compnet_type='vmf',num_mixtures=4):

	mix_models = []
	msz = []
	for i in range(len(categories)):
		filename = mix_model_path + 'mmodel_' + categories[i] + '_K{}_FEATDIM{}_{}_specific_view.pickle'.format(num_mixtures, vc_num, layer)
		mix = np.load(filename, allow_pickle=True)
		if compnet_type=='vmf':
			mix = np.array(mix)
		elif compnet_type == 'bernoulli':
			mix = np.array(mix[0])
		mix = np.transpose(mix, [0, 1, 3, 2])
		mix_models.append(torch.from_numpy(mix).type(torch.FloatTensor))
		msz.append(mix.shape)

	maxsz = np.max(np.asarray(msz),0)
	maxsz[2:4] = maxsz[2:4] + (np.mod(maxsz[2:4], 2) == 0)
	if layer == 'pool4' and compnet_type=='vmf':
		# Need to cut down the model to enable training
		maxsz[2] = maxsz[2] - 20#42
		maxsz[3] = maxsz[3] - 40#92

	mm = torch.zeros(0,vc_num,maxsz[2],maxsz[3])
	for i in range(len(categories)):
		mix = mix_models[i]
		cm, hm, wm = mix.shape[1:]
		# pad height
		if layer =='pool5':
			# because feature height is odd (7) copmared to 14 in pool4
			diff1 = int(np.ceil((maxsz[2] - hm) / 2))
		else:
			diff1 = int(np.floor((maxsz[2] - hm) / 2))
		diff2 = maxsz[2] - hm - diff1
		if diff1 < 0 or diff2<0:
			mix = mix[:,:,np.abs(diff1):np.abs(diff1)+maxsz[2]]
		else:
			if compnet_type=='vmf':
				mix = F.pad(mix, (0, 0, diff1, diff2, 0, 0, 0, 0), 'constant', 0)
			elif compnet_type=='bernoulli':
				mix = F.pad(mix, (0, 0, diff1, diff2, 0, 0, 0, 0), 'constant', np.log(1 / (1 - 1e-3)))
		# pad width
		if layer =='pool5':
			# because feature height is odd (7) copmared to 14 in pool4
			diff1 = int(np.ceil((maxsz[3] - wm) / 2))
		else:
			diff1 = int(np.floor((maxsz[3] - wm) / 2))
		diff2 = maxsz[3] - wm - diff1
		if diff1 < 0 or diff2<0:
			mix = mix[:, :, :, np.abs(diff1):np.abs(diff1) + maxsz[3]]
		else:
			if compnet_type=='vmf':
				mix = F.pad(mix, (diff1, diff2, 0, 0, 0, 0, 0, 0), 'constant', 0)
			elif compnet_type == 'bernoulli':
				mix = F.pad(mix, (diff1, diff2, 0, 0, 0, 0, 0, 0), 'constant', np.log(1 / (1 - 1e-3)))
		mm = torch.cat((mm,mix),dim=0)
	if device_id:
		mm = mm.cuda(device_id[0])
	return mm

def pad_to_size(x, to_size):
	padding = [(to_size[1] - x.shape[3]) // 2, (to_size[1] - x.shape[3]) - (to_size[1] - x.shape[3]) // 2, (to_size[0] - x.shape[2]) // 2, (to_size[0] - x.shape[2]) - (to_size[0] - x.shape[2]) // 2]
	return F.pad(x, padding)

def myresize(img, dim, tp):
	H, W = img.shape[0:2]
	if tp == 'short':
		if H <= W:
			ratio = dim / float(H)
		else:
			ratio = dim / float(W)

	elif tp == 'long':
		if H <= W:
			ratio = dim / float(W)
		else:
			ratio = dim / float(H)

	return cv2.resize(img, (0, 0), fx=ratio, fy=ratio)

def getImg(mode,categories, dataset, data_path, cat_test=None, occ_level='ZERO', occ_type=None, bool_load_occ_mask = False):

	if mode == 'train':
		train_imgs = []
		train_labels = []
		train_masks = []
		for category in categories:
			if dataset == 'pascal3d+':
				if occ_level == 'ZERO':
					filelist = data_path + 'pascal3d+_occ/' + category + '_imagenet_train' + '.txt'
					img_dir = data_path + 'pascal3d+_occ/TRAINING_DATA/' + category + '_imagenet'
			elif dataset == 'coco':
				if occ_level == 'ZERO':
					img_dir = data_path +'coco_occ/{}_zero'.format(category)
					filelist = data_path +'coco_occ/{}_{}_train.txt'.format(category, occ_level)

			with open(filelist, 'r') as fh:
				contents = fh.readlines()
			fh.close()
			img_list = [cc.strip() for cc in contents]
			label = categories.index(category)
			for img_path in img_list:
				if dataset=='coco':
					if occ_level == 'ZERO':
						img = img_dir + '/' + img_path + '.jpg'
					else:
						img = img_dir + '/' + img_path + '.JPEG'
				else:
					img = img_dir + '/' + img_path + '.JPEG'
				occ_img1 = []
				occ_img2 = []
				train_imgs.append(img)
				train_labels.append(label)
				train_masks.append([occ_img1,occ_img2])

		return train_imgs, train_labels, train_masks

	else:
		test_imgs = []
		test_labels = []
		occ_imgs = []
		for category in cat_test:
			if dataset == 'pascal3d+':
				filelist = data_path + 'pascal3d+_occ/' + category + '_imagenet_occ.txt'
				img_dir = data_path + 'pascal3d+_occ/' + category + 'LEVEL' + occ_level
				if bool_load_occ_mask:
					if  occ_type=='':
						occ_mask_dir = data_path + 'pascal3d+_occ/' + category + 'LEVEL' + occ_level+'_mask_object'
					else:
						occ_mask_dir = data_path + 'pascal3d+_occ/' + category + 'LEVEL' + occ_level+'_mask'
					occ_mask_dir_obj = data_path + 'pascal3d+_occ/0_old_masks/'+category+'_imagenet_occludee_mask/'
			elif dataset == 'coco':
				if occ_level == 'ZERO':
					img_dir = data_path+'coco_occ/{}_zero'.format(category)
					filelist = data_path+'coco_occ/{}_{}_test.txt'.format(category, occ_level)
				else:
					img_dir = data_path+'coco_occ/{}_occ'.format(category)
					filelist = data_path+'coco_occ/{}_{}.txt'.format(category, occ_level)

			if os.path.exists(filelist):
				with open(filelist, 'r') as fh:
					contents = fh.readlines()
				fh.close()
				img_list = [cc.strip() for cc in contents]
				label = categories.index(category)
				for img_path in img_list:
					if dataset != 'coco':
						if occ_level=='ZERO':
							img = img_dir + occ_type + '/' + img_path[:-2] + '.JPEG'
							occ_img1 = []
							occ_img2 = []
						else:
							img = img_dir + occ_type + '/' + img_path + '.JPEG'
							if bool_load_occ_mask:
								occ_img1 = occ_mask_dir + '/' + img_path + '.JPEG'
								occ_img2 = occ_mask_dir_obj + '/' + img_path + '.png'
							else:
								occ_img1 = []
								occ_img2 = []

					else:
						img = img_dir + occ_type + '/' + img_path + '.jpg'
						occ_img1 = []
						occ_img2 = []

					test_imgs.append(img)
					test_labels.append(label)
					occ_imgs.append([occ_img1,occ_img2])
			else:
				print('FILELIST NOT FOUND: {}'.format(filelist))
		return test_imgs, test_labels, occ_imgs

def imgLoader(img_path,mask_path,bool_resize_images=True,bool_square_images=False):

	input_image = Image.open(img_path)
	if bool_resize_images:
		if bool_square_images:
			input_image.resize((224,224),Image.ANTIALIAS)
		else:
			sz=input_image.size
			min_size = np.min(sz)
			if min_size!=224:
				input_image = input_image.resize((np.asarray(sz) * (224 / min_size)).astype(int),Image.ANTIALIAS)
	preprocess =  transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	img = preprocess(input_image)

	if mask_path[0]:
		mask1 = cv2.imread(mask_path[0])[:, :, 0]
		mask1 = myresize(mask1, 224, 'short')
		try:
			mask2 = cv2.imread(mask_path[1])[:, :, 0]
			mask2 = mask2[:mask1.shape[0], :mask1.shape[1]]
		except:
			mask = mask1
		try:
			mask = ((mask1 == 255) * (mask2 == 255)).astype(np.float)
		except:
			mask = mask1
	else:
		mask = np.ones((img.shape[0], img.shape[1])) * 255.0

	mask = torch.from_numpy(mask)
	return img,mask

class Imgset():
	def __init__(self, imgs, masks, labels, loader,bool_square_images=False):
		self.images = imgs
		self.masks 	= masks
		self.labels = labels
		self.loader = loader
		self.bool_square_images = bool_square_images

	def __getitem__(self, index):
		fn = self.images[index]
		label = self.labels[index]
		mask = self.masks[index]
		img,mask = self.loader(fn,mask,bool_resize_images=True,bool_square_images=self.bool_square_images)
		return img, mask, label

	def __len__(self):
		return len(self.images)

def save_checkpoint(state, filename, is_best):
	if is_best:
		print("=> Saving new checkpoint")
		torch.save(state, filename)
	else:
		print("=> Validation Accuracy did not improve")
