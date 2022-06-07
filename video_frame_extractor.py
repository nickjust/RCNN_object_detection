# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 23:04:59 2021

@author: Oli Kue, Hochschule Trier
"""

import cv2


def FrameExtract(source=0, rate=30, limit=0, name="frame", path_output="FrameExtract/", mirror=False, frmt=".jpg", size=0):
    # EXTRACTS FRAMES EVERY <rate> FROM <source>
    # -------------------------------------------------------------------
    # source        <str> Video file or <int> 0 for webcam
    # rate          rate to extract frames
    # limit         maximum number of extracted frames
    # name          prefix for resulting image files
    # path_output   folder for resulting image files
    # mirror        flip image horizontally
    # frmt          image format of extracted frames
    # size          resize image (width, height), 0: keep original
    
    try:
        video = cv2.VideoCapture(source)    # Load video file
    except:
        print("No valid video file!")
        return False
    
    i       = 0
    count   = 0
    output  = []
    while(video.isOpened()):
        ret, frame = video.read()           # read video frame by frame
        if ret == False:
            break
        if mirror:                          # flip image if desired
            frame = cv2.flip(frame, 1)
        # pick each <rate>th frame and save as .jpg file
        if (i % rate == 0):                                     
            if size != 0:
                # resize (width, height) if desired
                frame = cv2.resize(frame, size)                 
            cv2.imwrite(path_output + name + "_" + str(i) + frmt, frame)
            output.append(name + "_" + str(i) + frmt)
            count += 1
            if limit != 0 and count >= limit: # maximum number of frames reached
                break
                
        i += 1
        
    video.release()                          # release the capture
    cv2.destroyAllWindows()
    
    return output
