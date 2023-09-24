# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    #
    enable_GPUs_id = [1]

    # data
    dataset_path = '/home/jyangcu/Dataset/Dataset_Fetus_Object_Detection/'
    slices = ['four_chamber_heart']
    selected_hospital = ['Hospital_1']
    min_size = 800  # image resize
    max_size = 1333 # image resize
    num_workers = 8
    test_num_workers = 8

    train_min_size_range = (-1, -1)
    train_min_size = (800,)
    train_max_size = 1333
    test_min_size = 800
    test_max_size = 1333
    pixel_mean = [0.40789654, 0.44719302, 0.47026115]
    pixel_std = [0.28863828, 0.27408164, 0.27809835]
    size_divisible = 32

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 0000
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 1000
    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    
    # debug
    debug_file = '/tmp/debugf'
    test_num = 10000

    # model
    load_path = None
    feat_channels = [0, 0, 512, 1024, 2048]
    out_channel = 256
    n_class = 80
    n_conv = 4
    prior = 0.01
    threshold = 0.05
    top_n = 1000
    nms_threshold = 0.6
    post_top_n = 100
    pos_radius = 1.5
    target_min_size = 0
    gamma = 2.0
    alpha = 0.25
    fpn_strides = [8, 16, 32, 64, 128]
    sizes = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 100000000]]
    
    top_blocks = True
    use_p5 = True
    center_sample = True
    iou_loss_type = 'giou'

    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    # def _parse(self, kwargs):
    #     state_dict = self._state_dict()
    #     for k, v in kwargs.items():
    #         if k not in state_dict:
    #             raise ValueError('UnKnown Option: "--%s"' % k)
    #         setattr(self, k, v)

    #     print('======user config========')
    #     pprint(self._state_dict())
    #     print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
