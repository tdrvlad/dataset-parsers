import scipy.io
import os
import glob
from PIL import Image, ImageOps, ImageDraw
import random
import numpy as np

image_w = 240
image_h = 240

converted_dataset_dir = 'classification_datasets/Comp_Cars_Brand_Models'
converted_dataset_train_dir = os.path.join(converted_dataset_dir, 'train')
converted_dataset_test_dir = os.path.join(converted_dataset_dir, 'test')

rel_path = '../../../compCars'
label_map_file = os.path.join(rel_path,'data','misc','make_model_name.mat')
make_label_field = 'make_names'
model_label_field = 'model_names'
label_map = scipy.io.loadmat(label_map_file)

annotations_dir = os.path.join(rel_path, 'data', 'label')
dataset_dir = os.path.join(rel_path, 'data', 'image')

target_makes = ['Aston Martin', 'BWM', 'Cadillac', 'Dacia', 'Chevy', 'Dodge', 'Ferrari', 'FIAT', 'GMC', 'Jaguar', 'LAND-ROVER', 'Lexus','MAZDA','MINI','Mustang', 'Nissan', 'Opel', 'Porsche', 'Skoda', 'Toyota', 'Volkswagen']
#target_makes = ['Dacia']

model_distinction = True
grayscale = False
unknown_class_rate = 0.1

#-1 - uncertain, 1 - front, 2 - rear, 3 - side, 4 - front-side, 5 - rear-side
target_viewpoints = [1, 4, 2, 3, 5]

if not os.path.exists(converted_dataset_dir):
    os.mkdir(converted_dataset_dir)

if not os.path.exists(converted_dataset_train_dir):
    os.mkdir(converted_dataset_train_dir)

if not os.path.exists(converted_dataset_test_dir):
    os.mkdir(converted_dataset_test_dir)

make_dirs = glob.glob(os.path.join(dataset_dir, '*'))

for make_dir in make_dirs:

    make_id = os.path.basename(make_dir)
    make_label = label_map.get(make_label_field)[int(make_id)-1]

    make_label = make_label[0][0]

    if make_label in target_makes:

        print('Processing make with name: {}'.format(make_label))
        model_dirs = glob.glob(os.path.join(make_dir,'*'))
        
        for model_dir in model_dirs:
            
            model_id = os.path.basename(model_dir)
            model_label = label_map.get(model_label_field)[int(model_id)-1]
            
            model_label = model_label[0][0]
            print(model_label)
            print('Processing {} {}.'.format(make_label, model_label))

            year_dirs = glob.glob(os.path.join(model_dir, '*'))

            for year_dir in year_dirs:

                year = os.path.basename(year_dir)
                print('Processing {} {} {}.'.format(make_label, model_label, year))
                                
                if model_distinction:
                    category_dir = make_label + ' ' + model_label
                else:
                    category_dir = make_label
                instance_train_dir = os.path.join(converted_dataset_train_dir, category_dir)
                instance_test_dir = os.path.join(converted_dataset_test_dir, category_dir)


                if not os.path.exists(instance_train_dir):
                    os.mkdir(instance_train_dir)

                if not os.path.exists(instance_test_dir):
                    os.mkdir(instance_test_dir)

                image_files = glob.glob(os.path.join(year_dir, '*'))
                
                for image_file in image_files:

                    image_id = os.path.basename(image_file).split('.')[0]

                    annotation_file = os.path.join(annotations_dir, make_id, model_id, year, image_id + '.txt')
                    with open(annotation_file) as f:
                        lines = f.readlines()
                    
                    perspective = int(str(lines[0]))
                    
                    bbox = str(lines[2]).split(' ')
                    bbox_x1 = int(bbox[0])
                    bbox_y1 = int(bbox[1])
                    bbox_x2 = int(bbox[2]) 
                    bbox_y2 = int(bbox[3])

                    image = Image.open(image_file)
                    cropped_image = image.crop((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
                    
                    w, h = cropped_image.size
                    ratio = w / h
                    
                    scale = max(w/image_w, h/image_h)
                    scaled_w = int(image_w * scale)
                    scaled_h = int(image_h * scale)

                    if grayscale:
                        mono_image = np.array(ImageOps.grayscale(cropped_image))
                        cropped_image = np.array(cropped_image)
                    
                        cropped_image[:,:,0] = mono_image
                        cropped_image[:,:,1] = mono_image
                        cropped_image[:,:,2] = mono_image

                        cropped_image = Image.fromarray(cropped_image)

                    if perspective in target_viewpoints:
                        
                        padded_image = ImageOps.pad(cropped_image, (scaled_w, scaled_h), color=0).resize((image_w, image_h))
                        if random.random() < 0.9:
                            padded_image.save(os.path.join(instance_train_dir, os.path.basename(image_file)))
                        else:
                            padded_image.save(os.path.join(instance_test_dir, os.path.basename(image_file)))
                            
    else:
        instance_train_dir = os.path.join(converted_dataset_train_dir, 'Unknown')
        instance_test_dir = os.path.join(converted_dataset_test_dir, 'Unknown')

        model_dirs = glob.glob(os.path.join(make_dir,'*'))
        
        for model_dir in model_dirs:
            
            model_id = os.path.basename(model_dir)
            model_label = label_map.get(model_label_field)[int(model_id)-1]
            
            model_label = model_label[0][0]
    
            year_dirs = glob.glob(os.path.join(model_dir, '*'))

            for year_dir in year_dirs:

                year = os.path.basename(year_dir)
                               
                instance_train_dir = os.path.join(converted_dataset_train_dir, 'Unknown')
                instance_test_dir = os.path.join(converted_dataset_test_dir, 'Unknown')

                if not os.path.exists(instance_train_dir):
                    os.mkdir(instance_train_dir)

                if not os.path.exists(instance_test_dir):
                    os.mkdir(instance_test_dir)

                image_files = glob.glob(os.path.join(year_dir, '*'))
                
                for image_file in image_files:
                    
                    if np.random.uniform() < unknown_class_rate:
                        print('Adding instance from {} as Unknown.'.format(make_label))

                        image_id = os.path.basename(image_file).split('.')[0]

                        annotation_file = os.path.join(annotations_dir, make_id, model_id, year, image_id + '.txt')
                        with open(annotation_file) as f:
                            lines = f.readlines()
                        
                        perspective = int(str(lines[0]))
                        
                        bbox = str(lines[2]).split(' ')
                        bbox_x1 = int(bbox[0])
                        bbox_y1 = int(bbox[1])
                        bbox_x2 = int(bbox[2]) 
                        bbox_y2 = int(bbox[3])

                        image = Image.open(image_file)
                        cropped_image = image.crop((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
                        
                        w, h = cropped_image.size
                        ratio = w / h
                        
                        scale = max(w/image_w, h/image_h)
                        scaled_w = int(image_w * scale)
                        scaled_h = int(image_h * scale)

                        if grayscale:
                            mono_image = np.array(ImageOps.grayscale(cropped_image))
                            cropped_image = np.array(cropped_image)
                        
                            cropped_image[:,:,0] = mono_image
                            cropped_image[:,:,1] = mono_image
                            cropped_image[:,:,2] = mono_image

                            cropped_image = Image.fromarray(cropped_image)

                        if perspective in target_viewpoints:
                            
                            padded_image = ImageOps.pad(cropped_image, (scaled_w, scaled_h), color=0).resize((image_w, image_h))
                            if random.random() < 0.9:
                                padded_image.save(os.path.join(instance_train_dir, os.path.basename(image_file)))
                            else:
                                padded_image.save(os.path.join(instance_test_dir, os.path.basename(image_file)))
    
