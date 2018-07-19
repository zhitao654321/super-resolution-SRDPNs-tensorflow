import os
import glob

import scipy.misc
import scipy.ndimage
import scipy.signal
import numpy as np

import tensorflow as tf
try:
  xrange
except:
  xrange = range

w_l,h_l,c_l = 0,0,0
FLAGS = tf.app.flags.FLAGS
i=1


def fillzero(input,w,h):

    if w > h:
        pad = int((w - h) /2)
        zeros = [[[0.]*3 for y in range(pad)] for x in range(w)]
        input = np.hstack((zeros,input,zeros))

    if h > w:
        pad = int((h - w) /2)
        zeros = [[[0.]*3 for y in range(h)] for x in range(pad)]
        input = np.vstack((zeros,input,zeros))

    return input


def preprocess(path,scale):

    with tf.Session() as sess:

       image = imread(path)
       w,h,_ = np.shape(image)
       if not FLAGS.is_train and max(w,h) > 400:
           r =  400./max(w,h)
           image = scipy.misc.imresize(image,size =(int(r*w),int(r*h)))
           print("image",np.shape(image))

       label_ = modcrop(image, scale)
       input_ = scipy.ndimage.interpolation.zoom(label_, [(1./scale), (1./scale), 1.], prefilter=True,order=1)
       input_ = scipy.ndimage.interpolation.zoom(input_, [(scale/1.), (scale/1.), 1.], prefilter=True,order=1)

       label_ =label_ / 255.
       input_ = input_ / 255.

       if not FLAGS.is_train:

          image_path = os.path.join(os.getcwd(), FLAGS.sample_dir)
          path = os.path.join(image_path, "ORIGNAL.png")
          imsave(input_, path)
          path = os.path.join(image_path, "GROUNDTRUTH.png")
          imsave(label_, path)
          global w_l,h_l,c_l
          w_l,h_l,c_l = np.shape(input_)
          input_ = fillzero(input_, w_l, h_l)
          w_l,h_l,c_l = np.shape(label_)
          label_ = fillzero(label_, w_l, h_l)

       else :

          global i
          print('read the image '+str(i)+' :', input_.shape)
          i+=1

       return input_, label_

def prepare_data(dataset,config):

  if FLAGS.is_train:

    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    types = ["*.bmp","*.jpg","*.png","gif"]
    data = []
    for type in types:
        data.extend(glob.glob(os.path.join(data_dir, type)))
  else:

    data = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)),config.testimg)

  return data

def imread(path):
  return  scipy.misc.imread(path, mode='RGB').astype(np.float)


def modcrop(image, scale=3):

  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image


def input_setup(config):

  """
    Read image files and make their sub-images.
  """

  if config.is_train:
    data = prepare_data(dataset="Train",config = config)
  else:
    data = prepare_data(dataset="Test",config = config)

  sub_input_sequence = []
  sub_label_sequence = []
  padding = abs(config.image_size - config.label_size) / 2

  if config.is_train :
    for i in xrange(len(data)):
      input_, label_ = preprocess(data[i],config.scale)

      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        h, w = input_.shape

      for x in range(0, h-config.image_size+1, config.stride):
        for y in range(0, w-config.image_size+1, config.stride):
          sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [32 x 32]
          sub_label = label_[x+int(padding):x+int(padding)+config.label_size, y+int(padding):y+int(padding)+config.label_size] # [22 x 22]

          sub_input = sub_input.reshape([config.image_size, config.image_size, 3])
          sub_label = sub_label.reshape([config.label_size, config.label_size, 3])

          sub_input_sequence.append(sub_input)
          sub_label_sequence.append(sub_label)

  else:

    input_, label_ = preprocess(data,config.scale)

    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      h, w = input_.shape

    nx = ny = 0
    for x in range(0, h-config.image_size+1, config.stride):
      nx += 1; ny = 0
      for y in range(0, w-config.image_size+1, config.stride):
        ny += 1
        if x>2 and y > 2:
            sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [32 x 32]
            sub_label = label_[x+int(padding):x+int(padding)+config.label_size, y+int(padding):y+int(padding)+config.label_size] # [22 x 22]
        else:
            sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
            sub_label = label_[x+int(padding):x+int(padding)+config.label_size, y+int(padding):y+int(padding)+config.label_size] # [22 x 22]

        sub_input = sub_input.reshape([config.image_size, config.image_size, 3])
        sub_label = sub_label.reshape([config.label_size, config.label_size, 3])

        sub_input_sequence.append(sub_input)
        sub_label_sequence.append(sub_label)


  arrdata = np.asarray(sub_input_sequence) # [?, 32, 32, 3]
  arrlabel = np.asarray(sub_label_sequence) # [?, 22, 22, 3]

  if not config.is_train:
    return nx, ny ,arrdata,arrlabel
  else:
    return arrdata,arrlabel


def imsave(image, path):
  scipy.misc.imsave(path, image)



def crop(images):

    pw, ph, _ = np.shape(images)
    global w_l, h_l, c_l

    if w_l > h_l:
        depad = int((ph - h_l)/2)+2
        images = images[:,depad:-depad]
    elif w_l < h_l:
        depad = int((pw - w_l)/2)+2
        images = images[depad:-depad,:]
    else:
        images = images[:,:,:]

    return images


def merge(images, size):

  h, w = images.shape[1], images.shape[2]
  img = np.zeros(((h-8)*size[0], (w-8)*size[1], 3))
  for idx, image in enumerate(images):
    i = idx % size[0]
    j = idx // size[1]
    img[j*(h-8):j*(h-8)+h-8, i*(w-8):i*(w-8)+w-8, :] = image[4:-4,4:-4]

  return img

'''
def merge(images, size):

  h, w = images.shape[1], images.shape[2]
  img = np.zeros(((h)*size[0], (w)*size[1], 3))
  for idx, image in enumerate(images):
    i = idx % size[0]
    j = idx // size[1]
    img[j*(h):j*(h)+h, i*(w):i*(w)+w, :] = image[:,:]

  return img
'''