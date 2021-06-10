import PIL
from PIL import Image,ImageTk
# import pytesseract
import cv2
from tkinter import *
width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = Tk()
root.bind('<Escape>', lambda e: root.quit())
root.geometry("840x480")

lmain = Label(root)
lmain.pack()
# lmain.place(x=0,y=0)

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    img = img.resize((840,480),PIL.Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(image=img)
    #replace image
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)
show_frame()
print('ads')
root.mainloop()