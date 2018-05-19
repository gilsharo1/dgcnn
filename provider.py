import os
import sys
import numpy as np
import h5py
from imgaug.imgaug import augmenters as iaa
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
  os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
  www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
  zipfile = os.path.basename(www)
  os.system('wget %s; unzip %s' % (www, zipfile))
  os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
  os.system('rm %s' % (zipfile))

sometimes = lambda aug: iaa.Sometimes(0.9, aug)

seq = iaa.Sequential([
    sometimes([
    iaa.Pad(px=(0,3)),
    iaa.Crop(px=(0, 3)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    #iaa.GaussianBlur(sigma=(0, 1.0)), # blur images with a sigma of 0 to 3.0
    #iaa.AdditiveGaussianNoise(scale=(0,0.03*255)),
    iaa.Add(value=(-20,20)),
    iaa.Multiply(mul=(0.8,1.2)),
    iaa.Dropout((0.0, 0.05)),])
])

def shuffle_data(data, labels):
  """ Shuffle data and labels.
    Input:
      data: B,N,... numpy array
      label: B,... numpy array
    Return:
      shuffled data, label and shuffle indices
  """
  idx = np.arange(len(labels))
  np.random.shuffle(idx)
  return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
  """ Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in xrange(batch_data.shape[0]):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                  [0, 1, 0],
                  [-sinval, 0, cosval]])
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
  """ Rotate the point cloud along up direction with certain angle.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in xrange(batch_data.shape[0]):
    #rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                  [0, 1, 0],
                  [-sinval, 0, cosval]])
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
  """ Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in xrange(batch_data.shape[0]):
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
             [0,np.cos(angles[0]),-np.sin(angles[0])],
             [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
             [0,1,0],
             [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
             [np.sin(angles[2]),np.cos(angles[2]),0],
             [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
  return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
  """ Randomly jitter points. jittering is per point.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, jittered batch of point clouds
  """
  B, N, C = batch_data.shape
  assert(clip > 0)
  jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
  jittered_data += batch_data
  return jittered_data

def shift_point_cloud(batch_data, shift_range=0.1):
  """ Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
  """
  B, N, C = batch_data.shape
  shifts = np.random.uniform(-shift_range, shift_range, (B,5))
  for batch_index in range(B):
    batch_data[batch_index,:,:] += shifts[batch_index,:]
  return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
  """ Randomly scale the point cloud. Scale is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, scaled batch of point clouds
  """
  B, N, C = batch_data.shape
  scales = np.random.uniform(scale_low, scale_high, B)
  for batch_index in range(B):
    batch_data[batch_index,:,:] *= scales[batch_index]
  return batch_data

def getDataFiles(list_filename):
  return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
  f = h5py.File(h5_filename)
  data = f['data'][:]
  label = f['label'][:]
  return (data, label)

def loadDataFile(filename):
  return load_h5(filename)

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict['data'], dict['labels']

def raw_images_to_tensor(data, is_aug=False):
  n = data.shape[0]
  im = raw_images_to_image_tensor(data, is_aug)
  coor = np.meshgrid(range(32), range(32))
  x = np.repeat(coor[0][:, :, np.newaxis], n, axis=2).astype('float')
  x = (x.transpose(2, 0, 1)-16.0)*2.0
  y = np.repeat(coor[1][:, :, np.newaxis], n, axis=2).astype('float')
  y = (y.transpose(2, 0, 1)-16.0)*2.0
  alldata = np.concatenate((im, x[:,:,:,np.newaxis], y[:,:,:,np.newaxis]), axis=3)
  alldata = alldata.reshape(n,1024,5)
  return alldata

def raw_images_to_image_tensor(data, is_aug=False):
  n = data.shape[0]
  im = data.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype('uint8')

  if is_aug:
    im = seq.augment_images(im)

  im = (im.astype('float')-128.0)/128.0

  return im



def load_h5_data_label_seg(h5_filename):
  f = h5py.File(h5_filename)
  data = f['data'][:] # (2048, 2048, 3)
  label = f['label'][:] # (2048, 1)
  seg = f['pid'][:] # (2048, 2048)
  return (data, label, seg)