import logging
import os
from pathlib import PosixPath
import pprint
import torch

logger = logging.getLogger(__name__)


class AttrDict():

    _freezed = False
    """ Avoid accidental creation of new hierarchies. """

    def __getattr__(self, name):
        if self._freezed:
            raise AttributeError(name)
        if name.startswith('_'):
            # Do not mess with internals. Otherwise copy/pickle will fail
            raise AttributeError(name)
        ret = AttrDict()
        setattr(self, name, ret)
        return ret

    def __setattr__(self, name, value):
        if self._freezed and name not in self.__dict__:
            raise AttributeError(
                "Config was freezed! Unknown config: {}".format(name))
        super().__setattr__(name, value)

    def __str__(self):
        return pprint.pformat(self.to_dict(), indent=1, width=100,
                              compact=True)

    __repr__ = __str__

    def to_dict(self):
        """Convert to a nested dict. """
        return {k: v.to_dict() if isinstance(v, AttrDict) else v
                for k, v in self.__dict__.items() if not k.startswith('_')}

    def from_dict(self, d):
        self.freeze(False)
        for k, v in d.items():
            self_v = getattr(self, k)
            if isinstance(self_v, AttrDict):
                self_v.from_dict(v)
            else:
                setattr(self, k, v)

    def update_args(self, args):
        """Update from command line args. """
        for cfg in args:
            keys, v = cfg.split('=', maxsplit=1)
            keylist = keys.split('.')

            dic = self
            for i, k in enumerate(keylist[:-1]):
                assert k in dir(dic), "Unknown config key: {}".format(keys)
                dic = getattr(dic, k)
            key = keylist[-1]

            oldv = getattr(dic, key)
            if not isinstance(oldv, str):
                v = eval(v)
            setattr(dic, key, v)

    def freeze(self, freezed=True):
        self._freezed = freezed
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.freeze(freezed)

    # avoid silent bugs
    def __eq__(self, _):
        raise NotImplementedError()

    def __ne__(self, _):
        raise NotImplementedError()


config = AttrDict()
_C = config     # short alias to avoid coding

# Experiment name
_C.NAME = 'default'

# options: 'pascal3d+', 'mnist', 'cifar10', 'coco'
_C.DATA.DATASET = 'pascal3d+'
# categories for testing (vehicles only)
_C.DATA.CATEGORY = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train']
# categories for training
_C.DATA.CATEGORY_TRAIN = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
                          'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']

# None means load from env variable.
_C.DATA.BASE_DIR = PosixPath('data/')#None
_C.DATA.MODEL_DIR = PosixPath('models/')#None
#_C.DATA.TENSORBOARD_DIR = None

_C.GPUS = None

_C.MODEL.OCC_TYPES = ['_white', '_noise', 'general']
_C.MODEL.COMPNET_TYPE = 'vmf'  # options: 'bernoulli','vmf'
_C.MODEL.VMF_KAPPA = 30 # variance of vMF distribution
_C.MODEL.VC_NUM = 512 # number of vMF kernels
_C.MODEL.VC_THRESHOLD = 0.0 # this threshold would be needed if you would like to implement CompNets with Bernoulli distributions (see our WACV'20 paper)
_C.MODEL.MIXTURE_NUM = 4 # number of mixture models per class
# options: 'pool5', 'pool4' if _C.MODEL.BACKBONE_TYPE is 'vgg'; 'last',
# 'second' if _C.MODEL.BACKBONE_TYPE is 'resnet50', 'resnext'. None
# means default layer, i.e. 'pool5' for 'vgg' and 'last' for 'resnet50',
# 'resnext', 'densenet'.
_C.MODEL.LAYER = 'pool5'
# options: 'vgg', 'resnet50', 'resnext'
_C.MODEL.BACKBONE_TYPE = 'vgg'

###############################################################################
# Training configs
###############################################################################
# Occlusion levels for training
_C.TRAIN.OCC_LEVELS = ['ZERO']
# Batch size for training. Note that it is the total (i.e. effective) batch
# size. If there are multiprocess/multigpu involved, each process/gpu may take
# a subset of the total batch size.
_C.TRAIN.BATCHSIZE = 32
# The weight for cluster loss
_C.TRAIN.ALPHA = 3.0
# The weight for mixture loss
_C.TRAIN.BETA = 3.0
# Number of epochs for training
_C.TRAIN.NUM_EPOCHS = 20
# Initial training learning rate
_C.TRAIN.LR = 1e-2
# Backbone learning rate multiplier (i.e. backbone learning rate is TRAIN.LR *
# TRAIN.BACKBONE_LR_MULT)
_C.TRAIN.BACKBONE_LR_MULT = 1.0
# Training learning rate milestones
_C.TRAIN.LR_MILESTONES = []
# Training learning rate reduce ratio (gamma)
_C.TRAIN.LR_GAMMA = 0.1
# Train VC
_C.TRAIN.TRAIN_VC = True
# Train mixture model
_C.TRAIN.TRAIN_MIX = True
# The occlusion types for validation during the training
_C.TRAIN.VAL_OCCLUSIONS = ['ZERO']
# Frequency to run validation during the training
_C.TRAIN.VAL_FREQUENCY = 100
# Run validation before training
_C.TRAIN.VAL_AT_START = False
# Frequency to log training info
_C.TRAIN.INFO.LOG_FREQUENCY = 10
# Whether to write summaries to TensorBoard during the training
_C.TRAIN.INFO.TENSORBOARD = True

# Occlusion levels for testing
_C.TEST.OCC_LEVELS = ['ZERO', 'ONE', 'FIVE', 'NINE']

_C.freeze()  # avoid typo / wrong config keys.


def finalize_configs(is_training: bool):
    _C.freeze(False)

    if _C.DATA.BASE_DIR is None:
        _C.DATA.BASE_DIR = PosixPath(os.environ['BASE_DIR'])
    if _C.DATA.MODEL_DIR is None:
        _C.DATA.MODEL_DIR = PosixPath(os.environ['MODEL_DIR'])
#    if _C.DATA.TENSORBOARD_DIR is None:
#        _C.DATA.TENSORBOARD_DIR = PosixPath(os.environ['TENSORBOARD_DIR'])

    if _C.MODEL.BACKBONE_TYPE == 'vgg':
        if _C.MODEL.LAYER is None:
            _C.MODEL.LAYER = 'pool5'
        assert _C.MODEL.LAYER in {'pool5', 'pool4'}
    elif _C.MODEL.BACKBONE_TYPE in {'resnet50', 'resnext', 'densenet'}:
        if _C.MODEL.LAYER is None:
            _C.MODEL.LAYER = 'last'
        assert _C.MODEL.LAYER in {'last', 'second'}
    else:
        raise ValueError('Unknown MODEL.BACKBONE_TYPE: {}'.format(
            _C.MODEL.BACKBONE_TYPE))
    _C.DATA.INIT_PATH = _C.DATA.MODEL_DIR / \
        'init_{}'.format(_C.MODEL.BACKBONE_TYPE)
    _C.DATA.DICT_PATH = _C.DATA.INIT_PATH / \
        'dictionary_{}'.format(_C.MODEL.BACKBONE_TYPE)
    _C.DATA.DICT_DIR = _C.DATA.DICT_PATH / \
        'dictionary_{}.pickle'.format(_C.MODEL.LAYER)

    _C.DATA.MIX_MODEL_PATH = _C.DATA.INIT_PATH / \
        'mix_model_{}_{}_EM_all/'.format(_C.MODEL.COMPNET_TYPE,
                                         'pascal3d+')
    if _C.GPUS is None:
        _C.GPUS = list(range(torch.cuda.device_count()))

    _C.freeze()


# TODO: Everything below are supposed to be remove soon. Currently
# they serve as a transition solution, so that other files which use config
# do not need significant change to use the new config object.


# TODO: Remove this function/hack when we migrate the other files and
# let them to directly use cfg.
def old_fashioned_config(cfg: AttrDict):
    """
    A helper function to convert a cfg object to a dictionary, which will be
    later set as global variables. This is a temporary hack to avoid making
    too many changes in other files, so that we can make a smooth change to the
    new code base.

    Arguments:
    cfg: An AttrDict object.

    Returns:
    A dictionary containing keyword-value mappings which serve as the model
    configuration.
    """
    finalize_configs(cfg)
    vMF_kappa = cfg.MODEL.VMF_KAPPA
    dataset = cfg.DATA.DATASET
    device_ids = cfg.GPUS
    data_path = cfg.DATA.BASE_DIR
    model_save_dir = cfg.DATA.MODEL_DIR
    vc_num = cfg.MODEL.VC_NUM
    compnet_type = cfg.MODEL.COMPNET_TYPE
    num_mixtures = cfg.MODEL.MIXTURE_NUM
    bool_pytorch = True  # TODO: To be deprecated. Always to be true.
    backbone_type = cfg.MODEL.BACKBONE_TYPE
    layer = cfg.MODEL.LAYER
    init_path = cfg.DATA.INIT_PATH
    occ_types_vmf = cfg.MODEL.OCC_TYPES
    occ_types_bern = cfg.MODEL.OCC_TYPES
    mix_model_path = cfg.DATA.MIX_MODEL_PATH
    dict_path = cfg.DATA.DICT_PATH
    categories = cfg.DATA.CATEGORY
    categories_train = cfg.DATA.CATEGORY_TRAIN
    dict_dir = cfg.DATA.DICT_DIR

    # Return all local variables as a dict
    ret = locals()
    ret.pop('cfg')

    # HACK: below are hacks to convert PosixPath to str, since some
    # consumer code use '+' to manipulate paths. We should stick to PosixPath
    # later.
    for k, v in ret.items():
        if isinstance(v, PosixPath):
            if v.is_dir():
                ret[k] = str(v) + '/'
            else:
                ret[k] = str(v)
    return ret


# TODO: Remove this statment/hack when we migrate the other files and
# let them directly use cfg.
globals().update(old_fashioned_config(_C))
