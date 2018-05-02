# This code is to study the effect of threshold
# on face detection

import os
import dlib
import face_recognition as fr
import cv2
import matplotlib.pyplot as plt
from face import Face
from sklearn.cluster import KMeans
import numpy as np

def getImages(image_dir):
    images = []
    img_names = os.listdir(image_dir)
    if(image_dir[-1] == '/'):
        IMAGE_BASE_DIR = image_dir
    else:
        IMAGE_BASE_DIR = image_dir + '/'
    for name in img_names:
        img_path = IMAGE_BASE_DIR + name
        img = fr.load_image_file(img_path)
        img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)
        img_name = name
        images.append(img)
    return images

def getFaces(images):
    encs = []
    faces = []
    for n, img in enumerate(images):
        print('Working on image [{}]'.format(n))
        dets, scores, idx = detector.run(img, 1, -0.5)
        print('{} faces detected'.format(len(dets)))
        faces_n = []
        boxes = []
        encs_n = []
        for i, d in enumerate(dets):
            face = Face()
            x1 = d.top(); y2 = d.right(); x2 = d.bottom(); y1 = d.left();
            face.bound_box = (x1, y1, x2, y2)
            boxes.append((x1, y2, x2, y1))
            face.img_id = n
            face.myFace = img[x1:x2, y1:y2]
            faces_n.append(face)
            cv2.rectangle(img, (y1,x1), (y2,x2), (0, 0, 255), 5)
            print('{:.02f}'.format(scores[i]), idx[i], x1, y1, x2, y2)
        encs_n = fr.face_encodings(img, boxes)
        for i, e in enumerate(encs_n):
            faces_n[i].enc = e
            faces.append(faces_n[i])
            encs.append(e)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    return faces, np.array(encs)

def getBins(encs, n_clusters = 8):
    kmeans = KMeans(n_clusters = 16, random_state = 0).fit(encs)
    bins = [set() for i in range(n_clusters)]
    for i, c in enumerate(kmeans.labels_):
        bins[c].add(i)
    return bins


def writeDataset(DIR_PATH):
    if not os.path.exists(DIR_PATH):
        os.mkdir(DIR_PATH)
    for bin_num, b in enumerate(bins):
        BIN_PATH = DIR_PATH + '/bin' + str(bin_num) + '/'
        if not os.path.exists(BIN_PATH):
            os.mkdir(BIN_PATH)
        for i, fid in enumerate(b):
            IMAGE_PATH = BIN_PATH + 'image' + str(i) + '.jpg'
            cv2.imwrite(IMAGE_PATH, faces[fid].myFace)

IMAGE_DIR = '/home/deeplearning/Desktop/Attendance-System/clip_data_4'
#IMAGE_DIR = '/home/deeplearning/Desktop/Attendance-System/frames_vid1'
DIR_PATH = '/home/deeplearning/Desktop/Attendance-System/knn_data'
detector = dlib.get_frontal_face_detector()
images = getImages(IMAGE_DIR)
faces, encs = getFaces(images)
bins = getBins(encs, 16)
print(bins)
#writeDataset(DIR_PATH)
