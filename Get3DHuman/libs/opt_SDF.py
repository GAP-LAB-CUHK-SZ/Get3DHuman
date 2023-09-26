import argparse
import os
import platform

class dataOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        g_data = parser.add_argument_group('Data')
        g_train = parser.add_argument_group('Training')
        g_sample = parser.add_argument_group('Sampling')
#        g_data.add_argument('--dataroot', type=str, default="/my_share/zhangyang/renderpeople/rp_360/pifu_data_20210323/",help='path to images (data folder)')
#        #g_data.add_argument('--dataroot', type=str, default=r'E:\PhD\program\2_human_reconstruction\code\1_generate_data\results\pifuHD',help='path to images (data folder)')
      
#        g_train.train_full_pifu
        
        if platform.system().lower() == 'windows':
            g_data.add_argument('--save_path', type=str, default=r'E:\PhD\program\human_reconstruction\Img_depth_sdf\img2depth2sdf\cache',
                            help='path to save')
            g_data.add_argument('--sdf_path', type=str, default=r'E:\PhD\program\my_idea\4_PifuHD_with_video\code\prepare_data\sdf_data_20210630',
                            help='path to save')
        
            g_data.add_argument('--dataroot', type=str, default=r'E:\PhD\program\2_human_reconstruction\code_generate_data\data\pifu_data\RP_data_20211130',help='path to images (data folder)')
#            g_data.add_argument('--dataroot', type=str, default=r'E:\PhD\program\2_human_reconstruction\code_generate_data\data\pifu_data\pifu_data_20210323',help='path to images (data folder)')
            g_data.add_argument('--dataroot_real', type=str, default=r'E:\PhD\data\full_body_withmask\icome_real_data',help='path to images (data folder)')
            g_data.add_argument('--dataroot_real_sdf', type=str, default=r'E:\PhD\data\full_body_withmask\icome_real_data_SDF\npy_real_3',help='path to images (data folder)')
#          
#          
            
            g_train.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
            g_train.add_argument('--batch_size', type=int, default=1, help='input batch size')
            g_sample.add_argument('--num_sample_inout', type=int, default=1250, help='# of sampling points')
            g_sample.add_argument('--num_workers', type=int, default=0)
#            print("windows")
        elif platform.system().lower() == 'linux':
            g_data.add_argument('--save_path', type=str, default="/data1/zhangyang/save_path/eccv/",
                            help='path to save')
            g_data.add_argument('--sdf_path', type=str, default="/data1/zhangyang/data/pifu_sdf/sdf_data_20210630/",
                            help='path to save')
        
            g_data.add_argument('--dataroot', type=str, default="/data1/zhangyang/data/RP_data_20211130/",
                                help='path to images (data folder)')
            #g_data.add_argument('--dataroot_real', type=str, default="/data1/zhangyang/data/real_img_20220216/real_data/",help='path to images (data folder)')
            g_data.add_argument('--dataroot_real', type=str, default="/data1/zhangyang/data/real_img_20220222/icome_real_data/",help='path to images (data folder)')
            g_data.add_argument('--dataroot_real_sdf', type=str, default="/data1/zhangyang/data/real_img_20220222/npy_real_3/",help='path to images (data folder)')
#          
            
            g_train.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
            g_train.add_argument('--batch_size', type=int, default=6, help='input batch size')
            g_sample.add_argument('--num_sample_inout', type=int, default=9000, help='# of sampling points')
            g_sample.add_argument('--num_workers', type=int, default=2)
                                  
        g_data.add_argument('--loadSize', type=int, default=1024, help='load size of input image')
        #g_data.add_argument('--batch_size', type=int, default=1, help='the batch size of training data')
        
        # extra arguments
        g_data.add_argument('--num_views', type=int, default=1, help='How many views to use for multiview network.')
#        g_data.add_argument('--num_sample_inout', type=int, default=1000, help='# of sampling points')
        g_data.add_argument('--num_sample_color', type=int, default=0, help='# of sampling points')
        g_data.add_argument('--aug_alstd', type=float, default=0.0, help='augmentation pca lighting alpha std')
        g_data.add_argument('--aug_bri', type=float, default=0.0, help='augmentation brightness')
        g_data.add_argument('--aug_con', type=float, default=0.0, help='augmentation contrast')
        g_data.add_argument('--aug_sat', type=float, default=0.0, help='augmentation saturation')
        g_data.add_argument('--aug_hue', type=float, default=0.0, help='augmentation hue')
        g_data.add_argument('--aug_blur', type=float, default=0.001, help='augmentation blur')
        g_data.add_argument('--random_multiview', action='store_true', help='Select random multiview combination.')
        g_data.add_argument('--sigma', type=float, default=5.0, help='perturbation standard deviation for positions')
        g_data.add_argument('--random_flip', action='store_true', help='if random flip')
        g_data.add_argument('--random_trans', action='store_true',default=False, help='if  ')
        g_data.add_argument('--random_scale', action='store_true', default=True, help='if  ')
        g_data.add_argument('--serial_batches', action='store_true',
                             help='if true, takes images in order to make batches, otherwise takes them randomly')
        g_data.add_argument('--pin_memory', action='store_true', help='pin_memory')
        
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt_data = self.gather_options()
        return opt_data
    

class G_sdf():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        # Experiment related
        g_exp = parser.add_argument_group('Experiment')
        g_exp.add_argument('--name', type=str, default='example',
                           help='name of the experiment. It decides where to store samples and models')
        g_exp.add_argument('--debug', action='store_true', help='debug mode or not')

        # Training related
        
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--loadSize', type=int, default=512, help='load size of input image')

        g_train.add_argument('--learning_rate', type=float, default=1e-4, help='adam learning rate')

        g_train.add_argument('--num_epoch', type=int, default=200, help='num epoch to train')

#        g_train.add_argument('--freq_plot', type=int, default=10, help='freqency of the error plot')
#        g_train.add_argument('--freq_save', type=int, default=50, help='freqency of the save_checkpoints')
#        g_train.add_argument('--freq_save_ply', type=int, default=100, help='freqency of the save ply')
#       
        g_train.add_argument('--no_gen_mesh', action='store_true')
        g_train.add_argument('--no_num_eval', action='store_true')
        
        g_train.add_argument('--resume_epoch', type=int, default=-1, help='epoch resuming the training')
        g_train.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')


        # Sampling related
        
        g_sample = parser.add_argument_group('Sampling')
        g_sample.add_argument('--z_size', type=float, default=200.0, help='z normalization factor')

        g_model = parser.add_argument_group('Model')
        
        
        # General
        g_model.add_argument('--norm', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')
        g_model.add_argument('--norm_color', type=str, default='instance',
                             help='instance normalization or batch normalization or group normalization')

        # hg filter specify
        g_model.add_argument('--num_stack', type=int, default=1, help='# of hourglass')
        g_model.add_argument('--skip_hourglass', action='store_true', help='skip connection in hourglass')
        g_model.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        
        
        g_model.add_argument('--hg_depth', type=int, default=2, help='# of stacked layer of hourglass')
        g_model.add_argument('--hg_dim', type=int, default=256, help='256 | 512')
    
        g_model.add_argument('--mlp_norm', type=str, default='group', help='normalization for volume branch')
       
        # Classification General
#        g_model.add_argument('--mlp_dim', nargs='+', default=[257, 1024, 512, 256, 128, 1], type=int, help='# of dimensions of mlp')
        g_model.add_argument('--mlp_dim', nargs='+', default=[257, 512, 256, 128, 1], type=int,help='# of dimensions of mlp. no need to put the first channel')
                             #g_model.add_argument('--mlp_dim', nargs='+', default=[257, 512, 256, 128, 64, 1], type=int,help='# of dimensions of mlp')
#        g_model.add_argument('--mlp_dim_color', nargs='+', default=[1024, 512, 256, 128, 3], type=int, help='# of dimensions of color mlp')
        g_model.add_argument('--merge_layer', type=int, default=2)
        g_model.add_argument('--mlp_res_layers', nargs='+', default=[1, 2], type=int,
                             help='leyers that has skip connection. use 0 for no residual pass')
        
        g_model.add_argument('--use_tanh', action='store_true',
                             help='using tanh after last conv of image_filter network')

        # for train
        parser.add_argument('--no_residual', action='store_true', help='no skip connection in mlp')
        parser.add_argument('--schedule', type=int, nargs='+', default=[60, 80],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        parser.add_argument('--color_loss_type', type=str, default='l1', help='mse | l1')

        # for eval
        parser.add_argument('--val_test_error', action='store_true', help='validate errors of test data')
        parser.add_argument('--val_train_error', action='store_true', help='validate errors of train data')
        parser.add_argument('--gen_test_mesh', action='store_true', help='generate test mesh')
        parser.add_argument('--gen_train_mesh', action='store_true', help='generate train mesh')
        parser.add_argument('--all_mesh', action='store_true', help='generate meshs from all hourglass output')
        parser.add_argument('--num_gen_mesh_test', type=int, default=1,
                            help='how many meshes to generate during testing')

        # path
        parser.add_argument('--checkpoints_path', type=str, default='./checkpoints', help='path to save checkpoints')
        parser.add_argument('--load_netG_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--load_netC_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--results_path', type=str, default='./results', help='path to save results ply')
        parser.add_argument('--load_checkpoint_path', type=str, help='path to save results ply')
        parser.add_argument('--single', type=str, default='', help='single data for training')
        # for single image reconstruction
        parser.add_argument('--mask_path', type=str, help='path for input mask')
        parser.add_argument('--img_path', type=str, help='path for input image')



        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt_data = self.gather_options()
        return opt_data
    
    

class G_normal():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
      
        ''''''
        
        parser.add_argument('--modelName', type=str, default='img2norm', help='model name')
        parser.add_argument('--isTrain', type=bool, default=True, help='train or test the model')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        #
        
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints_dir', help='log path')
        #parser.add_argument('--log', type=str, default='log_aug', help='log path')
        parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        
        
        ## for input and output
        parser.add_argument('--input_nc', type=int, default=3, help='input image channel')
        parser.add_argument('--output_nc', type=int, default=3, help='output channel')  # 1 for depth map; 3 for normal map
        #parser.add_argument('--output_dc', type=int, default=1, help='output channel')  # 1 for depth map; 3 for normal map
        
        ## for generator
        parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')
        
        ## for discriminators
        parser.add_argument('--num_D', type=int, default=3, help='number of discriminators to use')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', type=bool, default=True, help='if true, do *not* use VGG feature matching loss')
        parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        
        # for training
        parser.add_argument('--nepoch', type=int, default=100, help='number of epochs for training')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=10, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0004, help='initial learning rate for adam')
        parser.add_argument('--pretrain_path_n', type=str, default=None, help='load the pretrained# model from the specified location')
        #parser.add_argument('--pretrain_path_d', type=str, default=None, help='load the pretrained# model from the specified location')
#        opt = parser.parse_args()

        
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt_data = self.gather_options()
        return opt_data

class G_depth():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--gpu_ids', type=str, default='2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
            
        parser.add_argument('--modelName', type=str, default='dep', help='model name')
        parser.add_argument('--isTrain', type=bool, default=True, help='train or test the model')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        #
        
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints_dir', help='log path')
        #parser.add_argument('--log', type=str, default='log_aug', help='log path')
        parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        
        
        ## for input and output
        parser.add_argument('--input_nc', type=int, default=6, help='input image channel')
        parser.add_argument('--output_nc', type=int, default=1, help='output channel')  # 1 for depth map; 3 for normal map
        #parser.add_argument('--output_dc', type=int, default=1, help='output channel')  # 1 for depth map; 3 for normal map
        
        ## for generator
        parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')
        
        ## for discriminators
        parser.add_argument('--num_D', type=int, default=3, help='number of discriminators to use')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', type=bool, default=True, help='if true, do *not* use VGG feature matching loss')
        parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        
        # for training
        parser.add_argument('--nepoch', type=int, default=100, help='number of epochs for training')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=10, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0004, help='initial learning rate for adam')
#        parser.add_argument('--pretrain_path_n', type=str, default=None, help='load the pretrained# model from the specified location')
        parser.add_argument('--pretrain_path_d', type=str, default=None, help='load the pretrained# model from the specified location')
#        opt = parser.parse_args()

        
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt_data = self.gather_options()
        return opt_data


















   