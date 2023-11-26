import pandas as pd
import numpy as np
import torch
from imgaug import augmenters as iaa
import cv2
from PIL import Image
from torchvision import transforms
from rs_aug import rs_aug as T

# 获取类别和颜色的映射关系，返回颜色映射的列表
def get_label_colormap(csv_path='./datasets/class_dict.csv'):
    pd_label_color = pd.read_csv(csv_path, sep=',')

    classes = []
    colormap = []
    for i in range(len(pd_label_color.index)):

        tmp = pd_label_color.iloc[i]
        color = []
        color.append(tmp['r'])
        color.append(tmp['g'])
        color.append(tmp['b'])
        colormap.append(color)
        classes.append(tmp['name'])

    return colormap


# 把3D的RGB图像转化为一个二维数据，并且数组中的每个位置的取值对应着图像在该像素点的类别
def image2label(label,label_colormap=get_label_colormap()):
    cm2lbl = np.zeros(256**3)                     # 每个像素点有0 ~ 255的选择，RGB 三个通道  创建256*256*256的零矩阵
    for i,  cm in enumerate(label_colormap):
        cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i         # 建立索引

    data = np.array(label, dtype='int32')                   # 这里输入的image的形状为H*W*C
    idx =(data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    image2 = np.array(cm2lbl[idx], dtype='int64')           # 根据索引得到label矩阵，形状为H*W
    return image2                                           # 返回numpy形式的image2


# 函数返回该字典，其中包含每个标签类别的名称和对应的 RGB 颜色值。
def get_label_info(csv_path='./datasets/class_dict.csv'):
    data = pd.read_csv(csv_path)
    label = {}

    for _, row in data.iterrows():

        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        label[label_name] = [int(r), int(g), int(b)]

    return label



def one_hot_it(label, label_info=get_label_info()):
    # return semantic_map -> [H, W, num_classes]
    semantic_map = []
    for info in label_info:
        color = label_info[info]
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def reverse_one_hot(image):
    image = image.permute(1, 2, 0)   # [2, 512, 512] ==> [512, 512, 2]
    x = torch.argmax(image, dim=-1)  # [512, 512, 2] ==> [512, 512]
    return x


def colour_code_segmentation(image, label_values):
	"""
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

	# w = image.shape[0]
	# h = image.shape[1]
	# x = np.zeros([w,h,3])
	# colour_codes = label_values
	# for i in range(0, w):
	#     for j in range(0, h):
	#         x[i, j, :] = colour_codes[int(image[i, j])]
	label_values = [label_values[key] for key in label_values]  #  [[128, 0, 0], [0, 128, 0], [0, 0, 0]]
	colour_codes = np.array(label_values)  #[[128   0   0][  0 128   0][  0   0   0]]
	x = colour_codes[image.astype(int)]
	return x


def predict_on_image(model, height,width, csv_path,read_path,save_path):
    # pre-processing on image
    # image = cv2.imread("demo/ceshi.png", -1)
    image = cv2.imread(read_path, -1)  # 读进来直接是BGR(不
    # 是我们最常见的RGB格式，颜色肯定有区别。) 格式数据格式在 0~255,
    # flag = -1,   8位深度，原通道
    # flag = 0，   8位深度，1通道
    # flag = 1，   8位深度，3通道
    # flag = 2，   原深度， 1通道
    # flag = 3，   原深度， 3通道
    # flag = 4，   8位深度，3通道

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
    resize = iaa.Scale({'height': height, 'width': width})  # 将图像缩放到固定大小
    # 将增强序列应用于图像和关键点。 为了确保以相同的方式增加两种数据类型，我们首先将增强序列切换到确定模式。 没有它，我们会获得图像和关键点间不匹配旋转。 请注意，每转换一批切换一次确定模式。
    resize_det = resize.to_deterministic()
    image = resize_det.augment_image(image)
    image = Image.fromarray(image).convert('RGB')  # Opencv转PIL.Image
    image = transforms.ToTensor()(image).unsqueeze(0)  # 加一个batch_size 的维度，这样才能放进模型里去跑
    # read csv label path
    label_info = get_label_info(csv_path)
    # predict
    model.eval()
########################单loss输出###################################
    predict = model(image.cuda())[0]
########################多loss输出################################
    # predict = model(image.cuda())[0]
##################################################################
    #with torch.no_grad():
        #image1 = cv2.imread("demo/ceshi.png", -1)
        #predict = model(image.cuda())
        #predict=predict.cpu().numpy()
        #predict=predict[0,1,:,:]


        #pmin=np.min(predict)
        #pmax=np.max(predict)
        #predict=((predict-pmin)/(pmax-pmin+0.000001))*225
        #predict=predict.astype(np.uint8)
        #predict=cv2.applyColorMap(predict,cv2.COLORMAP_JET)
        #predict=predict[:,:,::-1]
        #predict = image1+predict*0.3
        #plt.imshow(predict, cmap='gray')
        #save_path = 'demo/epoch_%d.png' % (epoch)
        #cv2.imwrite(save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
    w =predict.size()[-1]
    h =predict.size()[-2]
    c =predict.size()[-3]
    predict = predict.resize(c,h,w)
    predict = reverse_one_hot(predict)  # (h,w)一张图上面每一个像素点对应分类好的数字
    predict = colour_code_segmentation(np.array(predict.cpu()), label_info)  # 效果看123.py
    predict = cv2.resize(np.uint8(predict), (height, width))  # 用cv2读图的时候，已经转换成了 (h,w,c)格式了
    # save_path = 'demo/epoch_%d.png' % (epoch+1)
    cv2.imwrite(save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))  # 指定路径存储图片


def predict_on_image_npy(model, epoch, csv_path,read_path,save_path):
    # pre-processing on image

    transform = T.Compose([
        T.NDVI(r_band=3, nir_band=4),
        T.NDWI(g_band=2, nir_band=4),
        T.Resize(target_size=(512, 512), interp='NEAREST'),
        T.Normalize(mean=([0] * 10), std=([1] * 10), bit_num=16, band_num=10)
    ])
    img, _, = transform(img=read_path)
    img = img.transpose(2, 0, 1)  # [512,512,12]-->[12,512,512]
    # img2, _, label2 = self.data_extend(img=image_path, label=label_path)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # 归一化处理,把数据集中到[0~1]

    image = torch.from_numpy(img).float()  # 已经归一化且为c,h,w,格式，不需要totensor
    image = image.unsqueeze(0)  # 加一个batch_size 的维度，这样才能放进模型里去跑
    # read csv label path
    label_info = get_label_info(csv_path)
    # predict
    model.eval()
########################单loss输出###################################
    predict = model(image.cuda()) # [n,5,512,512]
########################多loss输出################################
    # predict = model(image.cuda())[0]
##################################################################
    #with torch.no_grad():
        #image1 = cv2.imread("demo/ceshi.png", -1)
        #predict = model(image.cuda())
        #predict=predict.cpu().numpy()
        #predict=predict[0,1,:,:]


        #pmin=np.min(predict)
        #pmax=np.max(predict)
        #predict=((predict-pmin)/(pmax-pmin+0.000001))*225
        #predict=predict.astype(np.uint8)
        #predict=cv2.applyColorMap(predict,cv2.COLORMAP_JET)
        #predict=predict[:,:,::-1]
        #predict = image1+predict*0.3
        #plt.imshow(predict, cmap='gray')
        #save_path = 'demo/epoch_%d.png' % (epoch)
        #cv2.imwrite(save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
    w =predict.size()[-1]
    h =predict.size()[-2]
    c =predict.size()[-3]
    predict = predict.resize(c,h,w)
    predict = reverse_one_hot(predict)  # (h,w)一张图上面每一个像素点对应分类好的数字
    predict = colour_code_segmentation(np.array(predict.cpu()), label_info)  # 效果看123.py
    predict = cv2.resize(np.uint8(predict), (512, 512))  # 用cv2读图的时候，已经转换成了 (h,w,c)格式了
    # save_path = 'demo/epoch_%d.png' % (epoch+1)
    cv2.imwrite(save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))  # 指定路径存储图片

