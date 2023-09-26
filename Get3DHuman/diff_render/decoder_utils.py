import torch
import json
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

def load_decoder(experiment_directory, checkpoint_num=None, color_size=None, experiment_directory_color=None, parallel=True):
    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))
    if color_size is not None:
        specs['NetworkSpecs']['dims'][3] = specs['NetworkSpecs']['dims'][3] + color_size

    basedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    arch = __import__("diff_render.graph." + specs["NetworkArch"], fromlist=["Decoder"])

    if color_size is not None:
        latent_size = specs["CodeLength"] + color_size
        decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"], last_dim=3)
    else:
        latent_size = specs["CodeLength"]
        decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    if parallel:
        decoder = torch.nn.DataParallel(decoder)

    if checkpoint_num != None:
        if color_size is not None:
            saved_model_state = torch.load(
            os.path.join(experiment_directory_color, "ModelParameters", checkpoint_num + ".pth")
        )
            # since there is no prefix "module" in the saved decoder_color model, we add them
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in saved_model_state["model_state_dict"].items():
                name = 'module.' + k
                new_state_dict[name] = v
            saved_model_state["model_state_dict"] = new_state_dict
        else:
            saved_model_state = torch.load(
            os.path.join(experiment_directory, "ModelParameters", checkpoint_num + ".pth")
        )

        saved_model_epoch = saved_model_state["epoch"]
        decoder.load_state_dict(saved_model_state["model_state_dict"])
    return decoder



def decode_sdf_save(decoder, latent_vector, points, clamp_dist=0.1, MAX_POINTS=100000, no_grad=False):
    start, num_all = 0, points.shape[0]
    output_list = []
    while True:
        end = min(start + MAX_POINTS, num_all)
        if latent_vector is None:
            inputs = points[start:end]
        else:
            latent_repeat = latent_vector.expand(end - start, -1)
            inputs = torch.cat([latent_repeat, points[start:end]], 1)
        sdf_batch = decoder.inference(inputs)
        start = end
        if no_grad:
            sdf_batch = sdf_batch.detach()
        output_list.append(sdf_batch)
        if end == num_all:
            break
    sdf = torch.cat(output_list, 0)

    if clamp_dist != None:
        sdf = torch.clamp(sdf, -clamp_dist, clamp_dist)
    return sdf


def decode_sdf(decoder, latent_list, points, clamp_dist=0.1, MAX_POINTS=300000, no_grad=False):
    start, num_all = 0, points.shape[0]
    output_list = []
    while True:
        end = min(start + MAX_POINTS, num_all)
        latent_pifu = latent_list[0]
        calibs = latent_list[1]
        
        points_list = points[None]
        points_list = points_list.transpose(2,1)

        
        xyz = latent_list[2](points_list, calibs, transforms=None)
        xy = xyz[:, :2, :]
        sp_feat = latent_list[3](xyz, calibs=calibs)
        point_local_feat_list = [latent_list[4](latent_pifu, xy), sp_feat]       
        latent_vector = torch.cat(point_local_feat_list, 1)
        
        latent_vector = latent_vector.transpose(2,1)[0]
        
        if latent_vector is None:
            inputs = points[start:end]
        else:
            latent_repeat = latent_vector.expand(end - start, -1)
#            inputs = torch.cat([latent_repeat, points[start:end]], 1)
            inputs = latent_repeat.transpose(1,0)[None]
        sdf_batch = decoder(inputs)[0][0]
        
        sdf_batch = (2*sdf_batch-1)/10
        
        sdf_batch = sdf_batch.transpose(1,0)
        start = end
        if no_grad:
            sdf_batch = sdf_batch.detach()
        output_list.append(sdf_batch)
        if end == num_all:
            break
    sdf = torch.cat(output_list, 0)
#    print(sdf.shape)
    if clamp_dist != None:
        sdf = torch.clamp(sdf, -clamp_dist, clamp_dist)
    return sdf


def decode_sdf_gradient(decoder, latent_list, points, clamp_dist=0.1, MAX_POINTS=100000, no_grad=False):
    start, num_all = 0, points.shape[0]
    output_list = []
    while True:
        end = min(start + MAX_POINTS, num_all)
        points_batch = points[start:end]
        sdf = decode_sdf(decoder, latent_list, points_batch, clamp_dist=clamp_dist)
#        print(sdf.shape)
#        print(points_batch.shape)
        start = end
#        grad_tensor = torch.autograd.grad(outputs=sdf, inputs=points_batch, grad_outputs=torch.ones_like(points_batch), create_graph=True, retain_graph=True)
        grad_tensor = torch.autograd.grad(outputs=sdf, inputs=points_batch, grad_outputs=torch.ones_like(sdf), create_graph=True, retain_graph=True)
        grad_tensor = grad_tensor[0]
        if no_grad:
            grad_tensor = grad_tensor.detach()
        output_list.append(grad_tensor)
        if end == num_all:
            break
    grad_tensor = torch.cat(output_list, 0)
    return grad_tensor

def decode_color(decoder, color_code, shape_code, points, MAX_POINTS=100000, no_grad=False):
    start, num_all = 0, points.shape[0]
    output_list = []
    while True:
        end = min(start + MAX_POINTS, num_all)

        color_code_batch = color_code.expand(end - start, -1)
        shape_code_batch = shape_code.expand(end - start, -1)
        inputs = torch.cat([shape_code_batch, color_code_batch, points[start:end]], 1)

        color_batch = decoder.inference(inputs)
        start = end
        if no_grad:
            color_batch = color_batch.detach()
        output_list.append(color_batch)
        if end == num_all:
            break
    color = torch.cat(output_list, 0)
    return color

