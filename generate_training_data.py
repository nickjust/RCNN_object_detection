import os
import cv2
import PIL.Image as Image
import xml.etree.ElementTree as ET

# --- OUTSOURCED FUNCTIONS ----------------------------------------------------

def read_xml_pascal(xml_file: str):    
    # INPUT:    XML-File with Pascal annotation of labelled bounding boxes
    # OUTPUT:   Filename, bounding box coordinates
    
    xml         = ET.parse(xml_file)
    root        = xml.getroot()    
    list_bboxes = []
    
    for bboxes in root.iter("object"):
        filename = root.find("filename").text
        xmin, ymin, xmax, ymax = None, None, None, None
        
        for bbox in bboxes.findall("bndbox"):
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))  
            ymax = int(float(bbox.find("ymax").text))              
            
        list_bboxes.append([xmin, ymin, xmax, ymax])      
        
    return filename, list_bboxes



def compute_iou(boxA, boxB):
    # INPUT:    two bounding boxes [x_min, y_min, x_max, y_max]
    # OUTPUT:   iou = intersection over union    
    #Source of IOU function: Adrian Rosebrock from pyimagesearch.com
    
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])                                                  
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
     # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)     
     # compute the area of both the prediction and ground-truth rectangles                 
    boxAArea  = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)              
    boxBArea  = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)    
    
    # compute the intersection over union by taking the intersection area and
    # dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)                  
    return iou



def image_preprocessing(image):
    # converts from BGR to RGB color notation
    image_post      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                   
    return image_post



def resize_image(image, size):
    # Resizes image to a given width and height
    image   = cv2.resize(image, size)
    image   = Image.fromarray(image)
    return image






# --- MAIN FUNCTION -----------------------------------------------------------

def generate_training_data(path_base="base_data/", path_training="training_data/"):
    # Generates training data out of a set of base images and specified bounding boxes
    # to separate between positive and negative examples
    
    path_positive   = path_training + "positive/" # Set directories
    path_negative   = path_training + "negative/"
    
    n_positive      = 0   # Set counters
    n_negative      = 0
    
    # Loop over all files in base directory
    for i, file in enumerate(os.listdir(path_base)):
        if ".png" in file:  # Pick .png image files
            xml_bbox_file           = path_base + file.split(".")[0] + ".xml"
             # Read bounding box coordinates (Pascal annotation) [x_min, y_min, x_max, y_max]
            filename, list_bboxes   = read_xml_pascal(xml_bbox_file)           
            
            progress        = str("{:5.1f}%".format((i + 2) * 100 / len(os.listdir(path_base))))
            print("-- [" + progress + "]", filename) # print current file and progress
            
            selsearch       = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            image           = cv2.imread(path_base + file)
            image           = image_preprocessing(image) # apply image preprocessing
            image_copy      = image.copy()
            
            selsearch.setBaseImage(image)  # Process Selective Search Algorithm
            selsearch.switchToSelectiveSearchFast()
            # bounding boxes [x_min, y_min, width, height]
            results         = selsearch.process()                               
            
            count_positive  = 0
            count_negative  = 0
            count_total     = 0
            
            for bbox in results:                                                
                 # bounding box transform to [x_min, y_min, x_max, y_max]
                bbox_coord  = [bbox[0], bbox[1], (bbox[0] + bbox[2]), (bbox[1] + bbox[3])]        
                 # roi = region of interest [y_min, y_max, x_min, x_max]
                image_roi   = image_copy[bbox_coord[1]:bbox_coord[3], bbox_coord[0]:bbox_coord[2]]     
                  
                # compute IOU with bounding boxes (1st of list of bounding boxes)
                                                                                
                iou         = compute_iou(bbox_coord, list_bboxes[0])                                   
                
                if iou > 0.7 and count_positive < 24:     # CASE positive
                    image_roi_use   = resize_image(image_roi, (128, 128))
                    image_roi_use.save(path_positive + "pos_" + str(n_positive) + ".png")
                    n_positive      += 1
                    count_positive  += 1
                        
                elif iou < 0.3 and count_negative < 2:     # CASE negative
                    image_roi_use   = resize_image(image_roi, (128, 128))
                    image_roi_use.save(path_negative + "neg_" + str(n_negative) + ".png")
                    n_negative      += 1
                    count_negative  += 1
                    
                if count_total > 70:                                                                    
                    break
                
                count_total += 1
            
            
            
            
            
            
            
            
            
            
            
            
            