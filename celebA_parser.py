import os 
import glob
import json
import shutil
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray

identities_file = 'identity_CelebA.txt'
raw_dataset_dir = 'img_align_celeba_png'
processed_dataset_dir = 'CelebA_Faces'


MIN_IMG_PER_PERSON = 10
TRAIN_EVAL_SPLIT = 0.85

if os.path.exists(processed_dataset_dir):
    shutil.rmtree(processed_dataset_dir)
os.mkdir(processed_dataset_dir)
os.mkdir(os.path.join(processed_dataset_dir, 'train'))
os.mkdir(os.path.join(processed_dataset_dir, 'test'))


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

identities_lines = open(identities_file, 'r').readlines()
persons = {}

for line in identities_lines:
    split_line = line.split()
    img_file = split_line[0].split('.')[0] + '.png' 
    person_id = split_line[1].rstrip('\n')

    if persons.get(person_id) is None:
        persons[person_id] = []
    
    persons[person_id].append(img_file)

print('Found {} unique persons.'.format(len(persons.keys())))

for person_id, image_files in persons.items():

    if len(image_files) > MIN_IMG_PER_PERSON:

        train_person_dir = os.path.join(processed_dataset_dir, 'train', 'person_{}'.format(person_id))
        test_person_dir = os.path.join(processed_dataset_dir, 'test', 'person_{}'.format(person_id))

        if not os.path.exists(train_person_dir):
            os.mkdir(train_person_dir)
        if not os.path.exists(test_person_dir):
            os.mkdir(test_person_dir)

        split = int(TRAIN_EVAL_SPLIT * len(image_files))

        j=0
        for f in image_files[:split]:
            image_face = extract_face(os.path.join(raw_dataset_dir,f))
            if not image_face is None:
                image_face.save(os.path.join(train_person_dir, 'img_{}.png'.format(j)))
                j += 1
        
        j=0
        for f in image_files[split:]:
            image_face = extract_face(os.path.join(raw_dataset_dir,f))
            if not image_face is None:     
                image_face.save(os.path.join(test_person_dir, 'img_{}.png'.format(j)))
                j += 1
    else:
        print('Not enough images')  
 


    