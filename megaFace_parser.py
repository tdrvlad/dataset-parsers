import os 
import glob
import json
from shutil import copy
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray

NO_PERSONS = 150
MAX_IMG_PER_PERSON = 50
MIN_IMG_PER_PERSON = 15

dataset_dir = '../../../../../../hdd/identities_0'

def extract_face(filename, required_size=(240, 240)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    try:
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)

        return image
    except:
        return None


new_dataset_dir = 'classification_datasets/megaFace'
if not os.path.exists(new_dataset_dir):
    os.mkdir(new_dataset_dir)

if not os.path.exists(os.path.join(new_dataset_dir, 'train')):
    os.mkdir(os.path.join(new_dataset_dir, 'train'))

if not os.path.exists(os.path.join(new_dataset_dir, 'test')):
    os.mkdir(os.path.join(new_dataset_dir, 'test'))

sub_dirs = glob.glob(os.path.join(dataset_dir, '*'))

sub_dirs = sub_dirs[:min(len(sub_dirs), NO_PERSONS)]
i = 0 
for directory in sub_dirs:
    print(directory)
    files = glob.glob(os.path.join(directory,'*'))
    files = files[:min(len(files), MAX_IMG_PER_PERSON)]

    if len(files) > MIN_IMG_PER_PERSON:
        train_person_dir = os.path.join(new_dataset_dir, 'train', 'person_{}'.format(i))
        test_person_dir = os.path.join(new_dataset_dir, 'test', 'person_{}'.format(i))
        i += 1

        if not os.path.exists(train_person_dir):
            os.mkdir(train_person_dir)
        
        if not os.path.exists(test_person_dir):
            os.mkdir(test_person_dir)

        split_rate = 0.85
        split = int(split_rate * len(files))
        print(split)
        j=0
        for f in files[:split]:
            image_face = extract_face(f)
            if not image_face is None:
                image_face.save(os.path.join(train_person_dir, 'img_{}.png'.format(j)))
                j += 1
        
        j=0
        for f in files[split:]:
            
            image_face = extract_face(f)
            if not image_face is None:     
                image_face.save(os.path.join(test_person_dir, 'img_{}.png'.format(j)))
                j += 1

       

    else:
        print('Not enough images')  
       


    