import os #处理文件
import numpy as np
import medpy.io as medio #专门用来读取和保存医学图像（如 .nii.gz、.mha 等格式）。

src_path = '/media/s3/yzh1/DataSet/BraTS2023' #初始数据集路径
tar_path = '/media/s2/wyb/Dataset/BraTS2023PRE' #处理后的数据集路径
name_list = os.listdir(src_path) #读取src下所有文件的文件名并打包为列表

#定义一个函数，让处理后的大脑图像在长宽高三个维度上都不小于128个像素
def sup_128(xmin, xmax):
    if xmax - xmin < 128:
        print ('#' * 100)

        ecart = int((128-(xmax-xmin))/2) #需要补齐的宽度
        xmax = xmax+ecart+1
        xmin = xmin-ecart
    #修改边界至零上
    if xmin < 0:
        xmax-=xmin
        xmin=0
    return xmin, xmax

def crop(vol):
    if len(vol.shape) == 4:
        vol = np.amax(vol, axis=0)
    assert len(vol.shape) == 3

    x_dim, y_dim, z_dim = tuple(vol.shape) #获取非0像素的三维坐标
    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0) #返回 3 个数组，分别存着所有非黑像素点的 x、y、z 坐标。

    x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
    y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
    z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)

    x_min, x_max = sup_128(x_min, x_max)
    y_min, y_max = sup_128(y_min, y_max)
    z_min, z_max = sup_128(z_min, z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max

def normalize(vol):
    mask = vol.sum(0) > 0
    for k in range(4):
        x = vol[k, ...]
        y = x[mask]
        x = (x - y.mean()) / y.std()
        vol[k, ...] = x

    return vol

if not os.path.exists(os.path.join(tar_path, 'vol')):
    os.makedirs(os.path.join(tar_path, 'vol'))

if not os.path.exists(os.path.join(tar_path, 'seg')):
    os.makedirs(os.path.join(tar_path, 'seg'))

for file_name in name_list:
    print (file_name)
    
    # 1. 对应你的4个模态后缀进行替换，注意连接符是 '-'
    # t2f对应FLAIR, t1c对应T1ce, t1n对应T1, t2w对应T2
    flair, _ = medio.load(os.path.join(src_path, file_name, file_name + '-t2f.nii.gz'))
    t1ce, _ = medio.load(os.path.join(src_path, file_name, file_name + '-t1c.nii.gz'))
    t1, _ = medio.load(os.path.join(src_path, file_name, file_name + '-t1n.nii.gz'))
    t2, _ = medio.load(os.path.join(src_path, file_name, file_name + '-t2w.nii.gz'))

    # 堆叠起来，保持模型期望的通道顺序不变
    vol = np.stack((flair, t1ce, t1, t2), axis=0).astype(np.float32)
    x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)
    vol1 = normalize(vol[:, x_min:x_max, y_min:y_max, z_min:z_max])
    vol1 = vol1.transpose(1,2,3,0)
    print (vol1.shape)

    # 2. 读取你的标签文件，后缀为 '-seg'
    seg, _ = medio.load(os.path.join(src_path, file_name, file_name + '-seg.nii.gz'))
    seg = seg.astype(np.uint8)
    seg1 = seg[x_min:x_max, y_min:y_max, z_min:z_max]
    
    # 将标签4映射为3（BraTS通用做法）
    seg1[seg1==4]=3

    # 3. 保存为 npy (删除了原代码没用的HLG前缀)
    np.save(os.path.join(tar_path, 'vol', file_name + '_vol.npy'), vol1)
    np.save(os.path.join(tar_path, 'seg', file_name + '_seg.npy'), seg1)