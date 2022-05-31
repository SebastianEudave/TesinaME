import os
import numpy as np
from PIL import Image
import cv2
import imageio
import torch

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

def returnCAM(feature_conv, weight_softmax, class_idx, range2):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        sum = 0
        listSum = []
        count = 0
        for i in range(range2, range2 + nc*h*w):
            sum += weight_softmax[idx][i]
            count += 1
            if(count == h*w):
                listSum.append(sum / h*w)
                count = 0
                sum = 0
        cam = np.array(listSum).dot(feature_conv.reshape((nc,h*w)).detach().numpy())
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam, h

def show_cam(CAMs, width, height, orig_image, class_idx, all_classes, num_image, heatmapPath):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + np.asarray(orig_image) * 0.5
        # put class label text on the result
        cv2.putText(result, all_classes[class_idx[i]] + " frame: " +str(num_image + 1) , (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
        #cv2.imshow('CAM', result/255.)
        #cv2.waitKey(0)
        cv2.imwrite(heatmapPath + f"{num_image}.jpg", result)

def visualization(model, layerName, path, heatmapPath, gifPath, size):
    model = model.to("cpu")
    model.eval()
    sample1 = []
    images = []
    files = sorted(os.listdir(path))
    for file in files:
        image = Image.open(os.path.join(path, file))
        image = image.resize((size, size))
        images.append(image)
        sample1.append(np.asarray(image))

    sample1 = torch.tensor(sample1)
    # Transforming using CUDA to speed up, return to CPU to save memory,
    # pending to check memory transfer overhead.
    sample = torch.unsqueeze(sample1.permute(3, 0, 1, 2), 0).float().cpu()
    nodes, _ = get_graph_node_names(model)
    print(nodes)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    print(weight_softmax)
    feature_extractor = create_feature_extractor(
        model, return_nodes=[layerName])
    out = feature_extractor(sample)
    out[layerName] = out[layerName].permute(2, 0, 1, 3, 4)
    print(out[layerName].shape)

    # show and save the results
    range2 = 0
    for i in range(len(out[layerName])):
        CAMs, h= returnCAM(out[layerName][i], weight_softmax, [2], range2)
        show_cam(CAMs, size, size, images[i], [2], ['happiness', 'disgust', 'surprise'], i, heatmapPath)
        range2 += (i+1) * h * h

    images = []
    filenames = list()
    for file in os.listdir(heatmapPath):
        filenames.append(file)
    filenames.sort(key=lambda f: int(f.split('.')[0]))
    for filename in filenames:
        images.append(imageio.imread(heatmapPath + filename))
    imageio.mimsave(gifPath + 'heatmap.gif', images,duration=.25)