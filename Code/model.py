import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from Code.lib import ops
from Code.config import device_ids, occ_types_vmf, occ_types_bern
from Code.helpers import imgLoader
import glob
from Code.vMFMM import *


class ActivationLayer(nn.Module):
    """Compute activation of a Tensor. The activation could be a exponent or a
    binary thresholding.
    """

    def __init__(self, vMF_kappa, compnet_type, threshold=0.0):
        super(ActivationLayer, self).__init__()
        self.vMF_kappa = vMF_kappa
        self.compnet_type = compnet_type
        self.threshold = threshold

    def forward(self, x):
        if self.compnet_type == 'vmf':
            x = torch.exp(self.vMF_kappa * x) * \
                (x > self.threshold).type_as(x)
        elif self.compnet_type == 'bernoulli':
            x = (x > self.threshold).type_as(x)
        return x


class Net(nn.Module):
    def __init__(self, backbone, weights, vMF_kappa, occ_likely, mix_model,
                 bool_mixture_bg, compnet_type, num_mixtures, vc_thresholds):
        super(Net, self).__init__()
        self.backbone = backbone
        self.occ_likely = occ_likely
        self.compnet_type = compnet_type
        self.num_mixtures = num_mixtures
        self.num_classes = mix_model.shape[0]//num_mixtures
        self.mix_model = torch.nn.Parameter(mix_model)
        self.use_mixture_bg = bool_mixture_bg
        self.conv1o1 = Conv1o1Layer(weights)
        self.activation_layer = ActivationLayer(vMF_kappa, compnet_type,
                                                threshold=vc_thresholds)

        clutter_model = self.get_clutter_model(
            compnet_type, vMF_kappa)

        self.pointwiseinference = PointwiseInferenceLayer(
            occ_likely, self.mix_model, clutter_model, self.num_classes,
            self.use_mixture_bg, self.compnet_type, self.num_mixtures)
        self.softmax = SoftMaxTemp(torch.tensor(2.0))
        self.occlusionextract = OcclusionMaskExtractor(
            occ_likely, mix_model, clutter_model, self.num_classes,
            self.use_mixture_bg, self.compnet_type, self.num_mixtures)

    def forward(self, x):
        vgg_feat = self.backbone(x)
        vc_activations = self.conv1o1(vgg_feat)
        vmf_activations = self.activation_layer(vc_activations)
        mix_likeli = self.pointwiseinference(vmf_activations)
        mix_likeli = mix_likeli/(vgg_feat.shape[2]*vgg_feat.shape[3])
        soft = self.softmax(mix_likeli)
        return soft, vgg_feat, mix_likeli

    def get_occlusion(self, x, label):
        vgg_feat = self.backbone(x)
        vc_activations = self.conv1o1(vgg_feat)
        vmf_activations = self.activation_layer(vc_activations)
        scores, masks, part_scores = self.occlusionextract(
            vmf_activations, label, self.compnet_type)
        return scores, masks, part_scores

    # TODO: merge this function with helpers.update_clutter_model
    def get_clutter_model(self, compnet_type, vMF_kappa):
        idir = 'background_images/'
        vc_num = self.conv1o1.weight.shape[0]

        updated_models = torch.zeros((0, vc_num))
        boo_gpu = (self.conv1o1.weight.device.type == 'cuda')
        gpu_id = self.conv1o1.weight.device.index
        if boo_gpu:
            updated_models = updated_models.cuda(gpu_id)

        if self.compnet_type == 'vmf':
            occ_types = occ_types_vmf
        elif self.compnet_type == 'bernoulli':
            occ_types = occ_types_bern

        for j in range(len(occ_types)):
            occ_type = occ_types[j]
            with torch.no_grad():
                files = glob.glob(idir + '*'+occ_type+'.JPEG')
                clutter_feats = torch.zeros((0, vc_num))
                if boo_gpu:
                    clutter_feats = clutter_feats.cuda(gpu_id)
                for i in range(len(files)):
                    file = files[i]
                    img, _ = imgLoader(file, [[]], bool_resize_images=False,
                                       bool_square_images=False)
                    if boo_gpu:
                        img = img.cuda(gpu_id)

                    feats = self.activation_layer(self.conv1o1(self.backbone(img.reshape(
                        1, img.shape[0], img.shape[1], img.shape[2]))))[0].transpose(1, 2)
                    feats_reshape = torch.reshape(
                        feats, [vc_num, -1]).transpose(0, 1)
                    clutter_feats = torch.cat((clutter_feats, feats_reshape))

                mean_activation = torch.reshape(
                    torch.sum(clutter_feats, dim=1), (-1, 1)).repeat([1, vc_num])
                if compnet_type == 'bernoulli':
                    boo = torch.sum(mean_activation, dim=1) != 0
                    mean_vec = torch.mean(
                        clutter_feats[boo]/mean_activation[boo], dim=0)
                    updated_models = torch.cat(
                        (updated_models, mean_vec.reshape(1, -1)))
                else:
                    if (occ_type == '_white' or occ_type == '_noise'):
                        mean_vec = torch.mean(
                            clutter_feats/mean_activation, dim=0)
                        updated_models = torch.cat(
                            (updated_models, mean_vec.reshape(1, -1)))
                    else:
                        nc = 5
                        model = vMFMM(nc, 'k++')
                        model.fit(clutter_feats.cpu().numpy(),
                                  vMF_kappa, max_it=150, tol=1e-10)
                        mean_vec = torch.zeros(
                            nc, clutter_feats.shape[1]).cuda(gpu_id)
                        mean_act = torch.zeros(
                            nc, clutter_feats.shape[1]).cuda(gpu_id)
                        clust_cnt = torch.zeros(nc)
                        for v in range(model.p.shape[0]):
                            assign = np.argmax(model.p[v])
                            mean_vec[assign] += clutter_feats[v]
                            clust_cnt[assign] += 1

                        mean_vec_final = torch.zeros(
                            sum(clust_cnt > 0), clutter_feats.shape[1]).cuda(gpu_id)
                        cnt = 0
                        for v in range(mean_vec.shape[0]):
                            if clust_cnt[v] > 0:
                                mean_vec_final[cnt] = (
                                    mean_vec[v]/clust_cnt[v].cuda(gpu_id)).t()
                        updated_models = torch.cat(
                            (updated_models, mean_vec_final))

                        if torch.isnan(updated_models.min()):
                            print('ISNAN IN CLUTTER MODEL')

        return updated_models


class PointwiseInferenceLayer(nn.Module):
    def __init__(self, occ_likely, mix_model, clutter_model, num_classes,
                 use_mixture_bg, compnet_type, num_mixtures):
        super(PointwiseInferenceLayer, self).__init__()
        self.bool_occ = np.sum(np.asarray(occ_likely)) != 0
        self.occ_likely = occ_likely
        self.mix_model = mix_model
        self.clutter_model = clutter_model.unsqueeze(2).unsqueeze(3)
        self.num_clutter_models = clutter_model.shape[0]
        self.num_classes = num_classes
        self.softmax = SoftMaxTemp(torch.tensor([2]))
        self.use_mixture_bg = use_mixture_bg
        self.compnet_type = compnet_type
        self.num_mixtures = num_mixtures
        if self.compnet_type == 'vmf':
            self.const_pad_val = 0.0
        elif self.compnet_type == 'bernoulli':
            self.const_pad_val = np.log(1 / (1 - 1e-3))

    def forward(self, *inputs):
        input, = inputs
        # Get clutter model, with shape (n_clutter, n_channel, 1, 1)
        clutter_model = F.normalize(torch.clamp(
            self.clutter_model, 0, 1), p=1, dim=1)
        occ_likely = self.occ_likely[0]

        # Step 1: compute the background scores.
        if self.compnet_type == 'vmf':
            # Compute the background score, with shape (n_batch, n_clutter,
            # height, weight)
            background = (input.unsqueeze(1) * clutter_model).sum(axis=2)
            background = torch.log(background * occ_likely + 1e-10)
            mm = F.normalize(torch.clamp(self.mix_model, 0, 1), p=1, dim=1)
        elif self.compnet_type == 'bernoulli':
            background = (input.unsqueeze(1) * torch.log(clutter_model + 1e-3)
                          + (1.0 - input.unsqueeze(1)) *
                          torch.log(1-(clutter_model + 1e-3))).sum(2) + \
                np.log(occ_likely)
            mm = self.mix_model

        # After this line, mm.shape should be (num_class*num_mixtures,
        # num_channels, height, width)
        mm = ops.crop_or_pad_as(mm, input, pad_val=self.const_pad_val)

        if self.use_mixture_bg:
            background = torch.max(background, dim=1, keepdims=True)[0]
            background = background.unsqueeze(1).unsqueeze(2)
        else:
            # Repeat the background score, the output shape will be (n_batch,
            # 1, 1, n_clutter, height, width)
            background = background.unsqueeze(1).unsqueeze(2)

        # Compute foreground score, after reshape, the shape will be (n_batch,
        # n_class, n_mixture, height, width)
        if self.compnet_type == 'vmf':
            foreground = torch.log(
                (input.unsqueeze(1) * mm).sum(2) * (1 - occ_likely) + 1e-10)
        elif self.compnet_type == 'bernoulli':
            obj_zero = torch.log(1.0-torch.exp(mm))
            foreground = ((input.unsqueeze(1) * mm) +
                          ((1.0 - input.unsqueeze(1)) * (obj_zero))
                          ).sum(2) + np.log(1.0 - occ_likely)
        else:
            raise ValueError(
                'Unknown compnet_type: {}'.format(self.compnet_type))
        # Reshape the foreground to (n_batch, n_class, n_mixture, 1, height,
        # width)
        foreground = foreground.reshape((-1, self.num_classes,
                                         self.num_mixtures, 1,
                                         *foreground.shape[2:]))
        # So far foreground and background will have 6 dimensions, with the
        # following meanings: n_batch, n_class, n_mixture, n_clutter, height,
        # width.
        if not self.bool_occ:
            background *= -np.inf
        # n_batch, n_class, n_mixture, n_clutter
        per_model_score = torch.max(foreground, background).sum((-1, -2))
        scores = per_model_score.max(axis=-1)[0].max(axis=-1)[0]
        return scores


class OcclusionMaskExtractor(nn.Module):
    def __init__(self, occ_likely, mix_model, clutter_model, num_classes, use_mixture_bg, compnet_type, num_mixtures):
        super(OcclusionMaskExtractor, self).__init__()
        self.bool_occ = np.sum(np.asarray(occ_likely)) != 0
        self.occ_likely = occ_likely
        self.mix_model = mix_model
        self.clutter_model = clutter_model
        self.num_clutter_models = clutter_model.shape[0]
        #self.scores = []
        self.num_classes = num_classes
        self.use_mixture_bg = use_mixture_bg
        self.compnet_type = compnet_type
        self.num_mixtures = num_mixtures

    def forward(self, x, label, cate_inx=None, attention=None):
        result = []
        occs = []
        parts = []
        bx, cx, hx, wx = x.shape

        for b in range(bx):
            score, occ, part_scores = self.clutter_likelihood(
                x[b], hx, wx, cx, label)
            #self.scores = torch.stack(scores)
            result.append(score)
            occs.append(occ)
            parts.append(part_scores)

        return result, occs, parts

    def clutter_likelihood(self, v, hx, wx, cx, label):
        #scores = []
        #v_mask = (v.sum(0) > 0).type(torch.cuda.FloatTensor).to(device_ids[0])
        #temp = 15
        if self.compnet_type == 'vmf':
            num_clutter_models = self.clutter_model.shape[0]
            # F.normalize(torch.exp(self.clutter_model * temp),p=1,dim=1)
            k = F.normalize(torch.clamp(self.clutter_model, 0, 1), p=1, dim=1)
        else:
            k = self.clutter_model

        # .type(torch.cuda.HalfTensor)
        k = k.unsqueeze(2).repeat(
            1, 1, hx * wx).reshape([k.shape[0], cx, hx, wx])
        occ_likely = self.occ_likely[0]

        if self.compnet_type == 'vmf':
            bg = (v * k).sum(1)
            background = torch.log(bg * occ_likely + 1e-10)
            mm = F.normalize(torch.clamp(self.mix_model, 0, 1), p=1, dim=1)
        elif self.compnet_type == 'bernoulli':
            background = (v * torch.log(k + 1e-3) + (1.0 - v) *
                          torch.log(1-(k + 1e-3))).sum(1) + np.log(occ_likely)
            mm = self.mix_model

        #attention = attention.repeat(self.num_clutter_models*4, 1, 1)
        # for inx in range(self.num_classes):
        #i = self.mix_model[inx]#
        cm, hm, wm = mm.shape[1:]
        if hm < hx:
            diff1 = (hx - hm) // 2
            diff2 = hx - hm - diff1
            mm = F.pad(mm, (0, 0, diff1, diff2, 0, 0, 0, 0), 'constant', 0)
        else:
            diff1 = (hm - hx) // 2
            diff2 = diff1 + hx
            mm = mm[:, :, diff1:diff2, :]
        if wm < wx:
            diff1 = (wx - wm) // 2
            diff2 = wx - wm - diff1
            mm = F.pad(mm, (diff1, diff2, 0, 0, 0, 0, 0, 0), 'constant', 0)
        else:
            diff1 = (wm - wx) // 2
            diff2 = diff1 + wx
            mm = mm[:, :, :, diff1:diff2]

        if self.use_mixture_bg:
            background = torch.max(background, dim=0)[0]
        else:
            background = background.repeat(self.num_mixtures, 1, 1)
#        scores = torch.zeros(self.num_classes)
#        if self.clutter_model.device.type != 'cpu':
#            scores = scores.to(device_ids[0])
        inx = label
        mix_class = mm[inx*self.num_mixtures:(inx+1)*self.num_mixtures]
        if self.compnet_type == 'vmf':
            foreground = torch.log((v * mix_class).sum(1)
                                   * (1 - occ_likely) + 1e-10)
        elif self.compnet_type == 'bernoulli':
            obj_zero = torch.log(1.0-torch.exp(mix_class))
            foreground = ((v * mix_class) + ((1.0 - v) * (obj_zero))
                          ).sum(1) + np.log(1.0 - occ_likely)
        if not self.use_mixture_bg:
            foreground = foreground.repeat(num_clutter_models, 1, 1)

        scores = torch.max(foreground, background).sum([1, 2])
        idx = torch.argmax(scores)
        score = scores[idx]
        if self.use_mixture_bg:
            occ = background - foreground[idx]
        else:
            occ = background[idx] - foreground[idx]
        if not self.use_mixture_bg:
            part_scores = torch.log((v * mix_class[idx/self.num_mixtures]) + 1e-10)
        else:
            part_scores = torch.log((v * mix_class[idx]) + 1e-10)

        return score, occ, part_scores


def resnet_feature_extractor(type, layer='last'):
    extractor = nn.Sequential()
    if type == 'resnet50':
        net = models.resnet50(pretrained=True)
        if layer == 'last':
            extractor.add_module('0', net.conv1)
            extractor.add_module('1', net.bn1)
            extractor.add_module('2', net.relu)
            extractor.add_module('3', net.maxpool)
            extractor.add_module('4', net.layer1)
            extractor.add_module('5', net.layer2)
            extractor.add_module('6', net.layer3)
            extractor.add_module('7', net.layer4)
        elif layer == 'second':
            extractor.add_module('0', net.conv1)
            extractor.add_module('1', net.bn1)
            extractor.add_module('2', net.relu)
            extractor.add_module('3', net.maxpool)
            extractor.add_module('4', net.layer1)
            extractor.add_module('5', net.layer2)
            extractor.add_module('6', net.layer3)
    elif type == 'resnext':
        net = models.resnext50_32x4d(pretrained=True)
        if layer == 'last':
            extractor.add_module('0', net.conv1)
            extractor.add_module('1', net.bn1)
            extractor.add_module('2', net.relu)
            extractor.add_module('3', net.maxpool)
            extractor.add_module('4', net.layer1)
            extractor.add_module('5', net.layer2)
            extractor.add_module('6', net.layer3)
            extractor.add_module('7', net.layer4)
        elif layer == 'second':
            extractor.add_module('0', net.conv1)
            extractor.add_module('1', net.bn1)
            extractor.add_module('2', net.relu)
            extractor.add_module('3', net.maxpool)
            extractor.add_module('4', net.layer1)
            extractor.add_module('5', net.layer2)
            extractor.add_module('6', net.layer3)
    else:
        extractor = []
    return extractor


class Conv1o1Layer(nn.Module):
    def __init__(self, weights):
        super(Conv1o1Layer, self).__init__()
        self.weight = nn.Parameter(weights)

    def forward(self, x):
        weight = self.weight
        xnorm = torch.norm(x, dim=1, keepdim=True)
        boo_zero = (xnorm == 0).type(torch.FloatTensor).to(device_ids[0])
        xnorm = xnorm + boo_zero
        xn = x / xnorm
        wnorm = torch.norm(weight, dim=1, keepdim=True)
        weightnorm2 = weight / wnorm
        out = F.conv2d(xn, weightnorm2)
        if torch.sum(torch.isnan(out)) > 0:
            print('isnan conv1o1')
        return out


class SoftMaxTemp(nn.Module):
    def __init__(self, temp):
        super(SoftMaxTemp, self).__init__()
        self.temp = temp

    def forward(self, x):
        x = torch.exp(torch.clamp(x*self.temp, -88.7, 88.7))
        return x / torch.sum(x, axis=1, keepdim=True)
