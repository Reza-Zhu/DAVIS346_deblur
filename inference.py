import torch
import torch.nn as nn
import torchvision.transforms
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
# my packages
from model import EFNet
from event_utils import events_to_accumulate_voxel_torch, RobustNorm
from model import load_network
import cv2
# added mask for model input
def davis346_inference(img,event,img_timestamp,model):



    expo_time = 40000
    sensor_size = (260, 346)  # -》 (256, 344)
    num_bins = 6  # voxel channel number
    # 控制最大最小值的比例，消除hot pixel影响
    robustnorm = RobustNorm(low_perc=0, top_perc=100)  # 95 85

    expo_sync_mode = 'start'

    ### 数据读取
    # input_image = Image.open(img_path)
    input_image = np.zeros((260,346,3),dtype='uint8')
    input_image[:,:,0] = img
    input_image[:, :, 1] = img
    input_image[:, :, 2] = img
    # input_image =img

    # if input_image is None:
    #     print(f'{img_path} does not exit!')
    input_image = torch.from_numpy(input_image.transpose(2, 0, 1)).float() / 255  # 0-1
    # print(f'DEBUG: input_image.shape:{input_image.shape}')

    input_events = event
    # print(f'DEBUG: input_events:{input_events}')

    ## image 预处理
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    # image normalize
    # input_image = preprocess(input_image)/255 # add image normalize to 0-1
    input_tensor = input_image.unsqueeze(0).cuda()
    # input_tensor = torchvision.transforms.functional.rgb_to_grayscale(input_tensor)
    # print(f'img tensor.max:{input_tensor.max()}')
    # print(f'img tensor.min:{input_tensor.min()}')
    ## 1channel -> 3channels n c h w
    # input_tensor = torch.cat([input_tensor,input_tensor,input_tensor], dim=1)

    ## Event 预处理
    ## filter the events that in expotime
    ts_img = img_timestamp
    # print(f'DEBUG: ts_img:{ts_img}')
    if expo_sync_mode == 'mid':
        ts_img_start = ts_img - expo_time // 2
        ts_img_end = ts_img + expo_time // 2
    elif expo_sync_mode == 'start':
        ts_img_start = ts_img
        ts_img_end = ts_img + expo_time

    ## find events
    idxs = np.searchsorted(input_events[:, 0], [ts_img_start, ts_img_end])
    filtered_event = input_events[idxs[0]:idxs[1], :]
    # print(f'DEBUG: filtered_event:{filtered_event}')

    ts = filtered_event[:, 0]
    # print(f'DEBUG: ts:{ts}')
    xs = filtered_event[:, 1]
    ys = filtered_event[:, 2]
    ps = filtered_event[:, 3]

    ## events->voxel
    xs = torch.from_numpy(xs.astype(np.float32)).cuda()
    ys = torch.from_numpy(ys.astype(np.float32)).cuda()
    # ts = torch.from_numpy((ts-ts_0).astype(np.float32)) # ts start from 0
    ts = ts - ts[0]
    ts = torch.from_numpy(ts.astype(np.float32)).cuda()  # !
    ps = torch.from_numpy(ps.astype(np.float32)).cuda()
    ps = ps * 2. - 1.

    # print(f'DEBUG: xs:{xs}')
    # print(f'DEBUG: ys:{ys}')
    # print(f'DEBUG: ts:{ts}')
    # print(f'DEBUG: ps:{ps}')

    voxel = events_to_accumulate_voxel_torch(xs, ys, ts, ps, B=num_bins, sensor_size=sensor_size, keep_middle=False)
    # voxel 没问题
    # print(f'DEBUG: voxel:{voxel}')

    # normalize CHANGED TO ROBUSTNORM!!
    # voxel = voxel / abs(max(voxel.min(), voxel.max(), key=abs))  # MaxNorm. DO NOT USE IT FOR DAVIS346!
    voxel *=1.2;
    voxel = robustnorm(voxel)

    # voxel = voxel * abs(max(voxel.min(), voxel.max(), key=abs))
    # print(f'voxel.max:{voxel.max()}')
    # print(f'voxel.min:{voxel.min()}')

    # TODO，调整该值的相对大小可以起到类似调整阈值的作用，但是要将它的数据范围限定在-1～1！
    voxel = voxel.unsqueeze(0)
    # print(f'DEBUG: voxel after normalize:{voxel}')

    # 截取至(256, 344)
    input_tensor = input_tensor[:, :, :256, :344]
    # print(f'DEBUG: input_tensor:{input_tensor}')
    voxel = voxel[:, :, :256, :344]

    ## mask
    mask = voxel
    mask = mask.sum(dim=1)
    mask = mask.unsqueeze(1)
    mask[mask != 0.] = 1.
    # print(mask)
    mask = mask.cuda()
    # print(f'DEBUG: voxel:{voxel}')

    # voxel *= 100000000000000000.
    # print(f'DEBUG: voxel:{voxel}')

    # 进行推理
    with torch.no_grad():
        output_tensor = model(x=input_tensor, event=voxel, mask=mask)  # added mask
        # print(f'DEBUG: output_tensor:{output_tensor}')

    output_tensor = output_tensor.cpu()
    # output_tensor = torchvision.transforms.functional.grayscale_to_rgb(output_tensor)
    output_image = output_tensor.squeeze().permute(1,2,0).clamp(0.0,1.0).numpy()
    output_image = (output_image * 255.0).astype('uint8')

    # # 保存输出图像
    # output_image = Image.fromarray(output_image)
    #
    # output_image.save(result_path)
    return output_image
