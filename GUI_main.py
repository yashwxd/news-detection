
import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as ms
import cv2
import sqlite3
import os
import numpy as np
import time
#from tkvideo import tkvideo
#'''import detection_emotion_practice as validate'''
#import video_capture as value
#import lecture_details as detail_data
#import video_second as video1

#import lecture_video  as video

global fn
fn = ""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="white")
# root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("********** Fake News Detection Using ML **********")

# 43
# video_label =tk.Label(root)
# video_label.pack()
#   #read video to display on label
# player = tkvideo("acci.mp4", video_label,loop = 1, size = (w, h))
# player.play()
# ++++++++++++++++++++++++++++++++++++++++++++
####For background Image
image2 = Image.open('img3.jpg')
image2 = image2.resize((w, h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=70)  # , relwidth=1, relheight=1)

label_l1 = tk.Label(root, text="*****___Fake News Detection Using ML____*****",font=("Times New Roman", 35, 'bold'),
                    background="#009ACD", fg="black", width=50, height=2)
label_l1.place(x=0, y=0)



img=ImageTk.PhotoImage(Image.open("img4.jpg"))

img2=ImageTk.PhotoImage(Image.open("img3.jpg"))

img3=ImageTk.PhotoImage(Image.open("img1.jpg"))


logo_label=tk.Label()
logo_label.place(x=0,y=100)

x = 1

# function to change to next image
def move():
 	global x
 	if x == 4:
	   x = 1
 	if x == 1:
	   logo_label.config(image=img)
 	elif x == 2:
	  logo_label.config(image=img2)
 	elif x == 3:
	  logo_label.config(image=img3)
 	x = x+1
 	root.after(2000, move)

# calling the function
move()
################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# def cap_video():
    
#     video1.upload()
#     #from subprocess import call
#     #call(['python','video_second.py'])

def reg():
    from subprocess import call
    call(["python","registration.py"])

def log():
    from subprocess import call
    call(["python","login.py"])
 
        
    
    
def window():
  root.destroy()
label_l1 = tk.Label(root,font=("Times New Roman", 5, 'bold'),
                    background="#009ACD", fg="black", width=1800, height=2)
label_l1.place(x=0, y=850)

button1 = tk.Button(root, text="LOGIN", command=log, width=14, height=1,font=('times', 20, ' bold '), bg="#8B0A50", fg="white")
button1.place(x=20, y=190)

button2 = tk.Button(root, text="REGISTER",command=reg,width=14, height=1,font=('times', 20, ' bold '), bg="#8B0A50", fg="white")
button2.place(x=20, y=300)

button3 = tk.Button(root, text="EXIT",command=window,width=14, height=1,font=('times', 20, ' bold '), bg="red", fg="black")
button3.place(x=20, y=450)
root.mainloop()