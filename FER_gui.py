from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as tkFileDialog
import cv2
from keras.models import model_from_json

outdir = ["Angry.png","Disgust.jpg","Fear.jpg","Happy.jpg","Sad.jpg","Surprise.jpg","Neutral.jpg"]

# load json and create model
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("fer.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))


def select_image():
    # grab a reference to the image panels
    global panelA, panelB
    # image
    path = tkFileDialog.askopenfilename()
    if len(path) > 0:
        # edges in it
        image = cv2.imread(path)
        # emotion = FER.py()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gif = cv2.imread(outdir[3])
        gif = cv2.cvtColor(gif, cv2.COLOR_BGR2RGB)

        # convert the images to PIL format...
        image = Image.fromarray(image)
        gif = Image.fromarray(gif)

        #ImageTk format
        image = ImageTk.PhotoImage(image)
        gif = ImageTk.PhotoImage(gif)
        # if the panels are None, initialize them
        if panelA is None or panelB is None:
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)

            # while the second panel will store the edge map
            panelB = Label(image=gif)
            panelB.image = gif
            panelB.pack(side="right", padx=10, pady=10)

        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=gif)
            panelA.image = image
            panelB.image = gif


root = Tk()
panelA = None
panelB = None

btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

root.mainloop()