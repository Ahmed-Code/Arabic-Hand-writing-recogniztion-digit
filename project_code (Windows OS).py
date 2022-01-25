import tkinter as tk
from PIL import Image,ImageGrab,ImageTk

import cv2 as cv
import matplotlib.pyplot as plt
from sklearn import svm, metrics

import pandas as pd
import numpy as np

import win32gui
from tkinter import filedialog , PhotoImage , Label

#reading the dataset from csv files and split the data to training and testing 
X_train = np.array(pd.read_csv('csvTrainImages 60k x 784.csv'))
y_train = np.array(pd.read_csv('csvTrainLabel 60k x 1.csv'))

X_test = np.array(pd.read_csv('csvTestImages 10k x 784.csv'))
y_test = np.array(pd.read_csv('csvTestLabel 10k x 1.csv'))

#transpose the images to be in the right way
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)
X_train = np.array([element.transpose() for element in X_train])
X_test = np.array([element.transpose() for element in X_test])


#output examples of the train data in the GUI
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image , label in zip(axes, X_train , y_train):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title('Training: %i' % label)


#reshape the the trining images to fit it in support vector machine method 
n1 = len(X_train)
X_train = X_train.reshape(n1,-1)

clf = svm.SVC()
print(X_train)
clf.fit(X_train , y_train)
#predict the testing data
n2 = len(X_test)
X_test = X_test.reshape(n2,-1)
predict = clf.predict(X_test)

#output examples of the testing data in the GUI with there prediction and true label
X_test = X_test.reshape(-1, 28, 28)
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction , true_label in zip(axes, X_test, predict , y_test):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}\nTrue value: {int(true_label)} ')

#Classification report for Support Vector Machine
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predict)}\n"
)


#Confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predict)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()


##############################################################################
# this class is for the input images and drawing board from the user
##############################################################################
class ImageGenerator:
    #consturcter for the class
    def __init__(self,parent):
        self.parent = parent
        self.b1 = "up"
        self.xold = None
        self.yold = None 
        self.drawing_area=tk.Canvas(self.parent,width=175,height=175)
        self.drawing_area.place(x=10,y=40)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
        self.button=tk.Button(self.parent,text="Predict",width=10,bg='white',command=self.save)#button for predicting the drawing and image
        self.button.place(x=25,y=225)
        self.button1=tk.Button(self.parent,text="Clear",width=10,bg='white',command=self.clear)#button for clearing the drawing board
        self.button1.place(x=105,y=225)
        self.photo=PhotoImage(file="Upload-Button-Background-PNG-Image.png")
        self.button2=tk.Button(self.parent , image=self.photo ,height="22" ,width="110" ,command=self.openFile)
        self.button2.place(x=595,y=225)

        self.Image_area=tk.Canvas(self.parent,width=175,height=175).place(x=560 ,y=40)

        tk.Label(text="",font=("Helvetica", 30) ,width=2 , height=1).place(x = 500, y = 130)
        tk.Label(text="",font=("Helvetica", 30) ,width=2 , height=1).place(x = 220, y = 130)

        tk.Label(text="by draw",font=("Helvetica", 18) ,width=10 , height=1 , background="white").place(x = 20, y = 0)
        tk.Label(text="by image",font=("Helvetica", 18) ,width=10 , height=1 , background="white").place(x = 578, y = 0)

    #this method is for opening the file explorer in the user device
    def openFile(self):
        #getting the file path for the image 
        filepath = filedialog.askopenfilename(title="Open file okay?",filetypes= (("images","*.png"),("images","*.jpg")))
        

        img = cv.imread(filepath)#get the image in the program
        dsize = (28, 28)
        output = cv.resize(img, dsize)#resizing the image to the dataset size
        cv.imwrite(f"temp.png",output)#after resizing the image we save it in temp file

        imm = Image.fromarray(img)
        resized_image = imm.resize((175,175))
        conveted_Image = ImageTk.PhotoImage(resized_image)
        self.fabelll = Label(self.parent , image = conveted_Image)
        self.fabelll.place(x=560 ,y=40)
        

        img = cv.imread("temp.png" , cv.IMREAD_GRAYSCALE)#read the temp file
        img = np.invert(np.array(img))
    
        img = img.reshape(1,-1)

        predict = clf.predict(img)

        tk.Label(text=str(predict[0]),font=("Helvetica", 30),width=2 , height=1).place(x = 500, y = 130)
        plt.imshow()#display the uploaded image by user to the GUI
        

    def save(self):
        
        HWND = self.drawing_area.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        a,b,c,d = rect
        rect=(a+4,b+4,c-4,d-4)
        im = ImageGrab.grab(rect)
        img = im.resize((28,28))
        img.save("resize.png", optimize=True, quality=40)


        img = cv.imread("resize.png", cv.IMREAD_GRAYSCALE)
        
        img = np.invert(np.array(img))

        img = img.reshape(1,-1)

        predict = clf.predict(img)

        tk.Label(text=str(predict[0]),font=("Helvetica", 30),width=2 , height=1).place(x = 220, y = 130)


    def clear(self):
        self.drawing_area.delete("all")

    def b1down(self,event):
        self.b1 = "down"

    def b1up(self,event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self,event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_oval(self.xold,self.yold,event.x,event.y,fill='black',width=18)
                
        self.xold = event.x
        self.yold = event.y


if __name__ == "__main__":
    root=tk.Tk()
    root.wm_geometry("%dx%d+%d+%d" % (800, 260, 600, 200))
    root.config(bg='white')
    ImageGenerator(root)
    root.mainloop()