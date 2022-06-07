import cv2
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
import models
import matplotlib.pyplot as plt

# --- MAIN FUNCTION -----------------------------------------------------------

def train_model(path_model="model/", model_name="model_1", model_save="model_1", test_size=0.15, epochs=7):
    X, y        = load_training_data()

    # split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)          
    
    if model_name == "model_1":
        model       = models.model_1(input_shape=(128, 128, 3))
        
        # Show model structure
        model.summary()                                                                     
    else:
        print("Model does not exist!")
        return False
    
    optimizer   = Adam(lr=0.0005)
    # Compile the model
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])    
    
    # fitting/training the model
    history     = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs)  
    
    # plot accuracy/loss for training/validation set as a function of the epochs
    plot_history(history)
    
    evaluate    = model.evaluate(X_test, y_test) # evaluate network on test set
    print("\nEVALUATION ON TESTSET: Loss =", evaluate[0], "- Metric values =", evaluate[1:])
    
    print("\nCLASSIFICATION REPORT ON TESTSET!\n")
    y_prediction_test = (model.predict(X_test) > 0.5).astype("int32")
    target_names = ['Auto', 'Kein Auto']
    print(classification_report(y_test, y_prediction_test, target_names=target_names))
      
    model.save(path_model + model_save + ".h5")


# --- OUTSOURCED FUNCTIONS ----------------------------------------------------

def load_training_data(limit_negative=2300):
    # INPUT:    limit of negative examples
    # OUTPUT    [[images], [pos/neg]]
    # pos: 1, neg: 0
    
    data            = load_files("training_data")
    filenames       = data["filenames"]
    
    X, y = [], []
    
    count_negative  = 0
    for name in filenames:
        if "neg" in name:                               # CASE negative
            count_negative += 1
            if count_negative < limit_negative: 
                image = cv2.imread(name)
                X.append(image)
                y.append(0)
                
        elif "pos" in name:                             # CASE positive
            image = cv2.imread(name)
            X.append(image)
            y.append(1)
            
    return np.array(X), np.array(y)


def plot_history(history):

    fig, axs = plt.subplots(2,figsize=(8,5))

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train_accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Evaluierung")
    
    # create loss subplot
    axs[1].plot(history.history["loss"], label="train_loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss Evaluierung")
   
    axs[0].grid()
    axs[1].grid()
    fig.tight_layout()
    plt.show()
    fig.savefig('Modell_Accuracy_and_Loss.pdf')   # save the figure to file






       
            
            
            
            
            
            
            
            
            
            
            