import tensorflow as tf
import random

IMAGE_SIZE=32
import os
from scipy.ndimage import rotate
from scipy.misc import face
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
import cv2
import requests
import numpy as np 

# Phong to va cat anh
def clipped_zoom(img):
    zoom_factor = random.randint(10, 20)/10
    h, w = img.shape[:2]
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Gioi han vung phong ta trong mang dau vao
    zh = int(np.round(h / zoom_factor))
    zw = int(np.round(w / zoom_factor))
    top = (h - zh) // 2
    left = (w - zw) // 2

    out = zoom(img[top:top+zh, left:left+zw], zoom_tuple)

    trim_top = ((out.shape[0] - h) // 2)
    trim_left = ((out.shape[1] - w) // 2)
    out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    

    return out
# Dich anh
def translate_image(img):
    rows,cols,_ = img.shape  
    trans_range = random.randint(-20, +20)
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    return img

# Lam net
def sharpen_image(img):
    gaussian = cv2.GaussianBlur(img, (5,5), 30.0)
    return cv2.addWeighted(img, 2, gaussian, -1, 0)


# Tinh toan bien doi hinh anh tuyen tinh ing*s+m
def lin_img(img,s=1.0,m=0.0):
    img2=cv2.multiply(img, np.array([s]))
    return cv2.add(img2, np.array([m]))
# Bien doi ve anh Binary
def contr_img(img):
    contr = 1-random.randint(-20, +20)/float(100)
    fade=127.0*(1-contr)
    return lin_img(img, contr, fade)

# Quay anh
def rotate_images(img):
    angle = random.randint(-90, 90)
    ang_rot = angle
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    return img


# def add_salt_pepper_noise(img):
#     # Need to produce a copy as to not modify the original image
#     dice = random.randint(0, 100)
    
#     if(dice<30):
#         row, col, _ = img.shape
#         salt_vs_pepper = 0.20
#         amount = 0.030
#         num_salt = np.ceil(amount * img.size * salt_vs_pepper)
#         num_pepper = np.ceil(amount * img.size * (1.0 - salt_vs_pepper))

#         #print(num_salt)
#         # Add Salt noise
#         coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
#         img[coords[0], coords[1], :] = 255

#         # Add Pepper noise
#         coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
#         img[coords[0], coords[1], :] = 0
#     return img

# Cat anh trung tam
def image_center_crop(img):
    h, w = img.shape[0], img.shape[1]
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    if h > w:
        diff = h - w
        pad_top = diff - diff // 2
        pad_bottom = diff // 2
    else:
        diff = w - h
        pad_left = diff - diff // 2
        pad_right = diff // 2
    return img[pad_top:h-pad_bottom, pad_left:w-pad_right, :]
# Giai ma anh tu bo dem
def decode_image_from_buf(buf):
    img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
# Cat anh
def crop(img, input_shape):
    img = image_center_crop(img) 
    img = cv2.resize(img, input_shape)  
    return img
# Ap dung mo hinh cho hinh anh tho kieu byte
def apply_model_to_image_raw_bytes(raw):
    img = decode_image_from_buf(raw)
    readable_img = crop(img, (256, 256))
    img = crop(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img, readable_img
    

