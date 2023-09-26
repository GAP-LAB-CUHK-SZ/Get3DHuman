

opt_data = { "random_scale":True,
            "aug_bri": 0.0, "aug_con":0.0, "aug_sat":0.0, "aug_hue":0.0, "load_size":1024 ,
            "random_trans":True, "aug_blur":0.001,"sdf_loadSize":512,"loadSize":512, "z_size":200}
    

opt_stylegan = {'G_kwargs': {'class_name': 'training.triplane_I3D.I3DGAN_Generator',
  'z_dim': 512,
  'w_dim': 512,
  'mapping_kwargs': {'num_layers': 2},
  'channel_base': 32768,
  'channel_max': 512,
   'fused_modconv_default': 'inference_only',
  'num_fp16_res': 0,},
 'G_reg_interval': 4.0,
 'run_dir': '00000-True-icome_real_data_img-gpus1-batch1-gamma1'}


import argparse
def load_opt():
    opt_all = [argparse.Namespace(**opt_data), argparse.Namespace(**opt_stylegan)]
    return opt_all



