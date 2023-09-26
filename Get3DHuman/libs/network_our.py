#from HGPIFuMRNet import HGPIFuMRNet, HGPIFuNetwNML, HGPIFuNet 
from HGPIFuMRNet import HGPIFuMRNet
from HGPIFuNetwNML import HGPIFuNetwNML
import torch
import opt_all
from TrainDataset import TrainDataset
from torch.utils.data import DataLoader
from networks import define_G
import numpy as np
import time
import cv2

train_normal, train_coarse, train_fine = False, False, True

save_path = r'E:\PhD\program\2_human_reconstruction\compare_code\HD_pifu_v1\lib_ours\cache'

opt_data, opt_c, opt_f  = opt_all.dataOptions().parse(), opt_all.CoarseOptions().parse(), opt_all.FineOptions().parse()

# generate data
train_dataset = TrainDataset(opt_data, phase='train')
test_dataset = TrainDataset(opt_data, phase='test')
train_data_loader = DataLoader(train_dataset,
                               batch_size=opt_data.batch_size, shuffle=not opt_data.serial_batches,
                               num_workers=0, pin_memory=opt_data.pin_memory)

print('train data size: ', len(train_data_loader))
test_data_loader = DataLoader(test_dataset,
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=opt_data.pin_memory)
print('test data size: ', len(test_data_loader))

cuda = torch.device("cpu")



projection_mode = train_dataset.projection_mode

input_test = torch.from_numpy(np.ones([1,3,512,512])).type(torch.FloatTensor)

# define normal net
#net_norm_F = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")  # Front normal
#net_norm_B = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")  # Back normal
#optimizer_norm = torch.optim.RMSprop(net_norm_F.parameters(), lr=opt_c.learning_rate, momentum=0, weight_decay=0)

#test normal net

#a = net_norm_F(input_test)


norm_f = torch.from_numpy(np.ones([1,3,512,512])).type(torch.FloatTensor)
norm_b = torch.from_numpy(np.ones([1,3,512,512])).type(torch.FloatTensor)

# define coarse pifu
netG_coarse_pifu = HGPIFuNetwNML(opt_c, projection_mode).to(device=cuda)  # coarse pifu
optimizer_c = torch.optim.RMSprop(netG_coarse_pifu.parameters(), lr=opt_c.learning_rate, momentum=0, weight_decay=0)

# define fine pifu
netG_fine_pifu = HGPIFuMRNet(opt_f, netG_coarse_pifu, projection_mode).to(device=cuda)
optimizer_f = torch.optim.RMSprop(netG_fine_pifu.parameters(), lr=opt_f.learning_rate, momentum=0, weight_decay=0)

'''
# test save model
torch.save(net_norm_F, save_path+'\model_F.pkl')
torch.save(net_norm_B, save_path+'\model_B.pkl')
torch.save(netG_coarse_pifu, save_path+'\model_coarse.pkl')
torch.save(netG_fine_pifu, save_path+'\model_fine.pkl')
'''

# define train procedure
def set_train_normal():
#    net_norm_F.train()
#    net_norm_B.train()
    netG_coarse_pifu.eval()
    netG_fine_pifu.eval()
    
def set_train_coarse():
    netG_coarse_pifu.train()
#    net_norm_F.eval()
#    net_norm_B.eval()

def set_train_fine():
    netG_fine_pifu.train()
#    net_norm_F.eval()
#    net_norm_B.eval()
    netG_coarse_pifu.eval()
    

for epoch in range(100):
    epoch_start_time = time.time()
    
    # train normal net
    if train_normal == True:
        set_train_normal()
        for train_idx, train_data in enumerate(train_data_loader):

            #  train_idx, train_data =  train_dataset.get_subjects()
            
            iter_start_time = time.time()
            # retrieve the data
            image_tensor = train_data['img_1024'].to(device=cuda)
            normal_f_tensor = train_data['normal_f_img_list'].to(device=cuda)
            normal_b_tensor = train_data['normal_b_img_list'].to(device=cuda)
            label =  train_data['labels'].data.numpy()

            
#            normal_f = net_norm_F(image_tensor.squeeze(1))
#            normal_b = net_norm_B(image_tensor.squeeze(1))
            # normal loss
            pass
            '''
            # test data
            test_i = 0
            image_train = (image_tensor.squeeze(0).cpu().data.numpy()+1)[test_i]/2
            normal_f_ =  (normal_f_tensor.squeeze(0).cpu().data.numpy()+1)[test_i]/2
            normal_b  = (normal_b_tensor.squeeze(0).cpu().data.numpy()+1)[test_i]/2
            
            cv2.imshow("image_train",np.transpose(image_train,[1,2,0]))
            cv2.imshow("normal_f_",np.transpose(normal_f_,[1,2,0]))
            cv2.imshow("normal_b",np.transpose(normal_b,[1,2,0]))
            cv2.waitKey()
            '''

    
    # train coarse net
    elif train_coarse == True:        
        set_train_coarse()
        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()
            # retrieve the data
            image_tensor = train_data['img_512'].to(device=cuda)
            calib_tensor = train_data['calib'].to(device=cuda)
            sample_tensor = train_data['samples'].to(device=cuda)
            label_tensor = train_data['labels'].to(device=cuda)
            normal_f_tensor = train_data['normal_f_img_list'].to(device=cuda)
            normal_b_tensor = train_data['normal_b_img_list'].to(device=cuda)
            
            #test model
#            if train_idx ==1:
#                print(stop2)
        
#            image_tensor_1024, image_tensor, sample_tensor, calib_tensor, None, labels=label_tensor[None,:])
            
            
            error_c, res_c = netG_coarse_pifu.forward(image_tensor.squeeze(1), 
                                             normal_f_tensor.squeeze(1), 
                                             normal_b_tensor.squeeze(1),
                                             sample_tensor, 
                                             calib_tensor.squeeze(1),
                                             label_tensor[None,:])
            

            optimizer_c.zero_grad()
            error_c['Err(occ)'].backward()
            optimizer_c.step()
            print("Coarse pifu loss: %06s"%error_c['Err(occ)'].detach().numpy())
    
    # train fine net
    elif train_fine == True:        
        set_train_fine()
        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()
            # retrieve the data
            image_tensor_512 = train_data['img_512'].to(device=cuda)
            image_tensor_1024 = train_data['img_1024'].to(device=cuda)
            image_tensor = train_data['img'].to(device=cuda)
            calib_tensor = train_data['calib'].to(device=cuda)
            sample_tensor = train_data['samples'].to(device=cuda)
            label_tensor = train_data['labels'].to(device=cuda)
            normal_f_tensor = train_data['normal_f_img_list'].to(device=cuda)
            normal_b_tensor = train_data['normal_b_img_list'].to(device=cuda)
             

            
            error_f, res_f = netG_fine_pifu.forward(image_tensor_1024, 
                                             normal_f_tensor, 
                                             normal_b_tensor,
                                             image_tensor_512,
                                             sample_tensor.unsqueeze(1), 
                                             calib_tensor,
                                             calib_tensor.squeeze(1),
                                             label_tensor[None,:])
            
#            img_c = np.ones([1,3,1024,1024])
#            print(nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)(torch.Tensor(img_c).float()).shape)
#            
#            a = netG_fine_pifu.filter_global(image_tensor_512.squeeze(1), norm_f.squeeze(1), norm_b.squeeze(1))
            #test model
#            if train_idx ==1:
#                print(stop3)
            optimizer_f.zero_grad()
            error_f['Err(occ:fine)'].backward()
            optimizer_f.step()
            print("fine pifu loss: %06s"%error_f['Err(occ:fine)'].detach().numpy())
            

a = netG_coarse_pifu(input_test)



# test fine pifu



















