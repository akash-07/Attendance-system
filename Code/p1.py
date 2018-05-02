import face_recognition as fr
import skimage
from scipy.misc import imresize
import os
from face import Face
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

IMAGE_BASE_PATH = '/home/deeplearning/Desktop/Attendance-System/frames_vid3'
BASE_IMAGE_DIR = '/home/deeplearning/Desktop/Attendance-System/Images/tmp_bins'

def getImages(image_dir):
    '''
    takes in image_dir as input

    returns a list of tuple:
        (image_id, numpy array of image)
    '''
    images = []
    img_names = os.listdir(image_dir)
    if(image_dir[-1] == '/'):
        IMAGE_BASE_DIR = image_dir
    else:
        IMAGE_BASE_DIR = image_dir + '/'
    for name in img_names:
        img_path = IMAGE_BASE_DIR + name
        img = fr.load_image_file(img_path)
        #img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)
        img_name = name
        img_tuple = (img_name, img)
        images.append(img_tuple)
    return images

def getFaces(images):
        '''
        images :: tuple(name, numpy array)
        n :: ground_truth number of faces in an image
        '''
        faces = []
        for img in images:
            print("Working on new image")
            fls = fr.face_locations(img[1], model = 'hog')
            encs = fr.face_encodings(img[1], fls)
            if len(fls)==0:
                print("GOT NO FACES IN IMAGE")
            print(len(encs), len(fls))
            b = 1
            for (fl, enc) in zip(fls, encs):
                face = Face()
                face.bound_box = fl
                face.img_name = img[0]
                face.box_no = b
                b += 1
                face.enc = enc
                faces.append(face)
                x1, y2, x2, y2 = fl
                #print(x1,y1,x2,y2)
                #if len(fls)>10:
                #    cv2.rectangle(img[1], (y1,x1), (y2,x2), (0, 0, 255), 5)
            #if len(fls)>10:
            #    plt.imshow(img[1])
            #    plt.show()
        return faces

def get_similarity(fid1,fid2,faces):
    f1,f2 = faces[fid1],faces[fid2]
    if f1.img_name==f2.img_name:
        return 0
    return (1- fr.face_distance(np.array(f1.enc.reshape(1, 128)),np.array(f2.enc.reshape(1, 128)))[0])

def print_face(n):
    img = fr.load_image_file(IMAGE_BASE_PATH+'/'+faces[n].img_name)
    x1, y2, x2, y2 = faces[n].bound_box
    print(x1, x2, y1, y2)
    plt.imshow(img[x1: x2, y1: y2])
    plt.show()

def get_sim_mat(faces):
    sim_mat = [0 for i in range(len(faces))]
    sim_mat = [sim_mat.copy() for i in range(len(faces))]

    for i in range(len(faces)):
        for j in range(i+1,len(faces)):
            sim_mat[i][j] = sim_mat[j][i] = get_similarity(i,j,faces)

    return sim_mat

bin_images = getImages(BASE_IMAGE_DIR)
frame_images = getImages(IMAGE_BASE_PATH)
images = bin_images+frame_images
faces = getFaces(images)
sim_mat = get_sim_mat(faces)
people = [i for i in range(len(bin_images))]
