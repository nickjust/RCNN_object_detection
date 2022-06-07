import time
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# --- MAIN FUNCTION -----------------------------------------------------------  
        
def classify(path_input="input/", input_name="test.png", path_model="model/", model_name="model_1"):
    image = path_input + input_name
    
    print("-- IMAGE:", image)
    print("-- MODEL:", path_model + model_name)
    
    if not os.path.isfile(image):   
        # Check if image file exists
        print("Image file does not exist!")
        return False
    else:
        image = cv2.imread(image)
        # convert from BGR to RGB color notation
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                 
        
    
    try:
        model = load_model(path_model + model_name + ".h5")  # Try to load model
    except:
        print(" Model does not exist!")
        return False
    
    # Run RCNN, returns image with resulting bounding box     
    time_start      = time.time()
    image_result    = rcnn(image, model)                           
    time_end        = time.time()

    print("-- RCNN classification took {:.4f} seconds".format(time_end - time_start))
    
    
    
    return image_result


# --- OUTSOURCED FUNCTIONS ----------------------------------------------------

def non_max_suppression_fast(boxes, overlapThreshold, probs):
    if len(boxes) == 0:                             # CASE no boxes
        return []
    
    pick = []
    x1 = boxes[:,0]                                 # coordinates of bounding box
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)            # area of bounding boxes
    idxs = np.argsort(probs)                        # sort by scores
    
    boxes_final = []
    while len(idxs) > 0:
        last    = len(idxs) - 1
        i       = idxs[last]
        
        # append last value of the index list
        pick.append(i)                                      
        
        # find largest coordinates for start of bounding box
        # and smallest for end of bounding box (most precise)
        xx1     = np.maximum(x1[i], x1[idxs[:last]])        
        yy1     = np.maximum(y1[i], y1[idxs[:last]])        
        xx2     = np.minimum(x2[i], x2[idxs[:last]])
        yy2     = np.minimum(y2[i], y2[idxs[:last]])
        
        width   = np.maximum(0, xx2 - xx1 + 1)              # width & height of bounding box
        height  = np.maximum(0, yy2 - yy1 + 1)
    
        overlap = (width * height) / area[idxs[:last]]      # ratio of overlap
        idxs    = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThreshold)[0])))
        
        boxes_final.append(boxes[pick])
        
    return boxes[pick]



def rcnn(image, model):
    selsearch       = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selsearch.setBaseImage(image)    
    #selsearch.switchToSelectiveSearchFast()
    selsearch.switchToSelectiveSearchQuality()
    
     # measure time for region propsal
    time_start      = time.time()
    results         = selsearch.process()                                                      
    time_end        = time.time()
    
    print("-- Selective Search took {:.4f} seconds".format(time_end - time_start))
    print("-- There are {} region proposals".format(len(results)))
    
    # images for result displaying
    # red: all propsed boxes, green: resulting box
    image_red       = image.copy()   
    image_green     = image.copy() 
    boxes_positive  = []
    probs           = []
    
     # loop over found bounding boxes
    for bbox in results:                                                                       
        x1  = bbox[0]
        y1  = bbox[1]
        x2  = bbox[0] + bbox[2]
        y2  = bbox[1] + bbox[3]
        
        roi     = image.copy()[y1:y2, x1:x2]
        roi     = cv2.resize(roi, (128, 128))
        roi_use = roi.reshape((1, 128, 128, 3))        
        class_predicted = (model.predict(roi_use) > 0.5).astype("int32")[0][0]
        
        if class_predicted == 1:
            prob = model.predict(roi_use)[0][0]
            if prob > 0.98:
                boxes_positive.append([x1, y1, x2, y2])
                probs.append(prob)
                # red boxes (all proposed bounding boxes)
                cv2.rectangle(image_red, (x1, y1), (x2, y2), (255, 0, 0), 5)                    
                
    cleaned_boxes   = non_max_suppression_fast(np.array(boxes_positive), 0.1, probs)
    boxes_total     = 0
    
    for clean_box in cleaned_boxes:
        clean_x1 = clean_box[0]
        clean_y1 = clean_box[1]
        clean_x2 = clean_box[2]
        clean_y2 = clean_box[3]
        boxes_total += 1
         # green box (resulting bounding box)
        cv2.rectangle(image_green, (clean_x1, clean_y1), (clean_x2, clean_y2), (0, 255, 0), 3) 
    
        
    plt.imshow(image_red)
    plt.show()
    plt.imshow(image_green)
    plt.show()
    
    return image_green
