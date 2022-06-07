import os
import cv2
from datetime import datetime


# Set directories
path_input      = "input/"
path_output     = "output/"
path_model      = "model/"
path_base       = "base_data/"
path_training   = "training_data/"

text_help       = "Select a task to execute:\n  - [x] extract video frames\n  - [g] generate training data\n  - [t] train model\n  - [d] detection\n  - [e] end"
model_name      = ""


# Create non-existing directories
if not os.path.exists(path_input):
    os.mkdir(path_input)
if not os.path.exists(path_output):
    os.mkdir(path_output)
if not os.path.exists(path_model):
    os.mkdir(path_model)
if not os.path.exists(path_base):
    os.mkdir(path_base)
if not os.path.exists(path_training):
    os.mkdir(path_training)
if not os.path.exists(path_training + "/positive"):
    os.mkdir(path_training + "/positive")
if not os.path.exists(path_training + "/negative"):
    os.mkdir(path_training + "/negative")






# --- OUTSOURCED FUNCTIONS ----------------------------------------------------

def image_BGR2RGB(image):
    # Converts from BGR to RGB color notation
    image_post      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_post



def save_result(data, typ="image", input_name="output", path_output="output/"):
    # Saves result image or sequence to an image or video file in output folder
    now         = datetime.now()
    timestamp   = now.strftime("%Y%m%d_%H%M%S")

    if typ == "image":
        cv2.imwrite(path_output + timestamp + "_" + input_name.split(".")[0] + ".png", image_BGR2RGB(data))
        print('-- Result image saved under "' + path_output + '"')
    
    if typ == "video":
        height, width, layers = data[0].shape
        size = (width, height)
        output = cv2.VideoWriter(path_output + timestamp + "_" + input_name.split(".")[0] + ".mp4", cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
        for frame in data:
            output.write(image_BGR2RGB(frame))
        output.release()
        print('-- Result video saved under "' + path_output + '"')
        





# --- MAIN FUNCTION -----------------------------------------------------------  

print(50*"=")
print("======= RCNN from training to classifying ========")
print(50*"=" + "\n")
print("\n" + text_help)


while True:
    task = input("Execute task: ")
    print(50*"-")
    
    
    # e - ends this program
    if task == "e":
        print("Good bye - see you!")
        break
    
            
        
    # extracts frames of a video to generate training images
    # video must be located in root folder
    # extracted frames are stored in base data folder
    elif task == "x":
        print('Video file must be located in root folder.')
        input_name = input(">> Video file:   ")
        print("Specify the frame extraction rate and frame limit.")
        frate       = int(input(">> Extract rate: "))
        flimit      = int(input(">> Frame limit:  "))
        fwidth      = int(input(">> Frame width:  "))
        fheight     = int(input(">> Frame height: "))
        frmt        = ".jpg"
        size        = (fwidth, fheight)
        
        print("EXTRACTING FRAMES...")
        import video_frame_extractor as vfe
        inputlist   = vfe.FrameExtract(source=path_input+input_name, name=input_name.split(".")[0] + "_frame", rate=frate, limit=flimit, path_output=path_base, frmt=frmt, size=size)
        print("EXTRACTION COMPLETE!\n")
        
    
    
    # generates training data out of the images stored in base data folder
    # saves positive and negative examples in training data folder
    elif task == "g":        
        import generate_training_data as gd
        
        print("GENERATING DATA...")
        gd.generate_training_data(path_base=path_base, path_training=path_training)
        print("GENERATING DATA COMPLETE!\n")
        
    
    
    # trains the model
    # model file is saved in model folder
    elif task == "t":
        import train_model as tm
        
        # variable stores the name of the used CNN architecture 
        model_name= "model_1"
        
        # name of the model selected by the user 
        print("Select a name to save the CNN model.")
        model_save  = input(">> Save model:   ")
        
        test_size   = 0.15
        epochs      = 7
        
        print("TRAINING MODEL...")
        tm.train_model(path_model=path_model, model_name=model_name, model_save=model_save, test_size=test_size, epochs=epochs)
        print("TRAINING MODEL COMPLETE!\n")
        
        
        
    # classify an input with a trained model
    # input can be image or video file
    # output is displayed and stored as image in folder output
    elif task == "d":
        import rcnn
        
        if model_name == "":
            print("Select an existing model.")
            model_name      = input(">> Model name:   ")
        else:
            print("To keep current model (" + model_name + ") press <Enter> or select another model.")
            model_name_new  = input(">> Model name:   ")
            if model_name_new != "":
                model_name  = model_name_new
                
        print("Select the Image or Video in which the object has to be detected.")
        input_name = input(">> Input Image or Video:  ")
        
        if any(i in input_name for i in [".jpg", ".jpeg", ".png"]):
            type = "image"
            print("   [type: image]")
            
        elif any(i in input_name for i in [".mp4", ".avi", ".mov"]):
            type = "video"
            print("   [type: video]")
          
        else:
            print("No supported file type!")
            continue
        
        
        if type == "image":
            print("CLASSIFYING IMAGE...")
            image_result = rcnn.classify(path_input=path_input, input_name=input_name, path_model=path_model, model_name=model_name)
            save_result(image_result, typ="image", input_name=input_name, path_output=path_output)
            print("IMAGE CLASSIFIED!\n")
        
        if type == "video":
            print("\nSpecify the frame extraction rate and frame limit.")
            frate       = int(input(">> Extract rate: "))
            flimit      = int(input(">> Frame limit:  "))
            frmt        = ".jpg"
            size        = 0
            
            import video_frame_extractor as vfe
            inputlist   = vfe.FrameExtract(source=path_input+input_name, name=input_name.split(".")[0] + "_frame", rate=frate, limit=flimit, path_output=path_input, frmt=frmt, size=size)
            
            outputlist  = []
            print("CLASSIFYING VIDEO FRAMES...")
            for i, frame in enumerate(inputlist):
                print("--------- Frame", i+1, "of", len(inputlist))
                image_result = rcnn.classify(path_input=path_input, input_name=frame, path_model=path_model, model_name=model_name)     
                outputlist.append(image_result)
            save_result(outputlist, typ="video", input_name=input_name, path_output=path_output)
            print("FRAMES CLASSIFIED!\n")
       
        
    else:
        print("Oops! Typo? Try again...")
        
    print(50*"-")
     
        
print("\n" + 50*"=")