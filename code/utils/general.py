import os
from glob import glob
import torch
import numpy as np

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def split_input(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    n_pixels = 10000
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        split.append(data)
    return split

def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    for entry in res[0]:
        print(res[0][entry].shape)

    model_outputs = {}
    for entry in res[0]:
        print(res[0][entry].shape)
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs

def cartesian_to_spherical(lightpos, pointpos):


    #pointpos = torch.tile(pointpos,(1,3,1))
    #pointpos = pointpos.repeat(1, lightpos.shape[0], 1)
    #lightpos = torch.repeat_interleave(lightpos,pointpos.shape[1],0)
    #lightpos = lightpos.unsqueeze(0)

    #ptsnew = torch.zeros([lightpos.shape[0], lightpos.shape[1], 4])

    lightpos = torch.clamp(lightpos, min=0.001)
    pointpos = torch.clamp(pointpos, min=0.001)

    xy = lightpos[:,:,0] ** 2 + lightpos[:,:,1] ** 2
    ptsnew0 = torch.atan2(torch.sqrt(xy), lightpos[:,:,2])  # for elevation angle defined from Z-axis down
    ptsnew1 = torch.atan2(lightpos[:,:,1], lightpos[:,:,0])

    pointpos = pointpos - lightpos
    pointpos = torch.clamp(pointpos, min=0.001)
    
    xy = pointpos[:,:,0]**2 + pointpos[:,:,1]**2
    ptsnew2 = torch.atan2(torch.sqrt(xy), pointpos[:,:,2]) # for elevation angle defined from Z-axis down
    ptsnew3 = torch.atan2(pointpos[:,:,1], pointpos[:,:,0])

    ptsnew = torch.stack([ptsnew0,ptsnew1,ptsnew2,ptsnew3],2).cuda()
    ptsnew = torch.squeeze(ptsnew,0)

    return ptsnew

def appendSpherical_np(xyz):
    ptsnew = np.zeros([lightpos.shape[0],lightpos.shape[1],4])
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew