# create window
from tkinter import *
import PIL

from PIL import Image,ImageTk
import tkinter.ttk as exTk

import cv2

class Demo1:
    def __init__(self, w):
        #create window
        self.w = 1920
        self.h = 1080
        #IMGAE
        self.canvas_bg = Canvas(w, width=1920, height=1080,bd=0,bg='red',highlightthickness=0,relief='ridge')
        self.canvas_bg.place(x=0,y=0)
        self.ib = ImageTk.PhotoImage(file="/home/bao/Desktop/DoAnTotNghiep/main/image/tech.png")
        self.img_bg =self.canvas_bg.create_image(0,0,anchor = 'nw',image = self.ib)

        self.canvas_img = Canvas(w, width=self.w*0.7, height = self.h*0.7,bd=3,relief='ridge')
        self.im = ImageTk.PhotoImage(file="/home/bao/Desktop/DoAnTotNghiep/main/image/a.png")
        self.image =self.canvas_img.create_image(0,0,anchor = 'nw',image = self.im)
        self.canvas_img.place(relx = 0.15,rely=0.24)
        #TEXT
        self.lable_height = exTk.Label(w,text="Height :",
                                    foreground='white',
                                    background='#081559',
                                    font="Times 20",
                                    relief=None,
                                    borderwidth=5,
                                    anchor='e',
                                    justify=CENTER)
        self.lable_bm = exTk.Label(w,text="Bust measurements:",
                                    background='#081559',
                                    foreground='white',
                                    font="Times 20",
                                    relief=None,
                                    borderwidth=5,
                                    anchor='e',
                                    justify=CENTER)
        self.lable_wm = exTk.Label(w,text="Waist measurements:",
                                    background='#081559',
                                    foreground='white',
                                    font="Times 20",
                                    relief=None,
                                    borderwidth=5,
                                    anchor='e',
                                    justify=CENTER)
        self.lable_hm = exTk.Label(w,text="Hip measurements:",
                                    background='#081559',
                                    foreground='white',
                                    font="Times 20",
                                    relief=None,
                                    borderwidth=5,
                                    anchor='e',
                                    justify=CENTER)
        # VALUE
        self.value_height = exTk.Label(w,
                                    background='#081559',
                                    font="Times 20",
                                    foreground='white',
                                    relief=None,
                                    borderwidth=5,
                                    anchor=CENTER,
                                    justify=CENTER)
        self.value_bm = exTk.Label(w,
                                    background='#081559',
                                    font="Times 20",
                                    foreground='white',
                                    relief=None,
                                    borderwidth=5,
                                    anchor=CENTER,
                                    justify=CENTER)
        self.value_wm = exTk.Label(w,
                                    background='#081559',
                                    font="Times 20",
                                    foreground='white',
                                    relief=None,
                                    borderwidth=5,
                                    anchor=CENTER,
                                    justify=CENTER)
        self.value_hm = exTk.Label(w,
                                    background='#081559',
                                    font="Times 20",
                                    foreground='white',
                                    relief=None,
                                    borderwidth=5,
                                    anchor=CENTER,
                                    justify=CENTER)
        self.num = exTk.Label(w,text="None",
                                background='#081559',
                                font="Times 20",
                                foreground='white',
                                relief=None,
                                borderwidth=5,
                                anchor=CENTER,
                                justify=CENTER)
        #position
        self.lable_height.place(height=50,width=250,x=300,y=20)
        self.lable_bm.place(height=50,width=250,x=300,y=70)
        self.lable_wm.place(height=50,width=250,x=300,y=120)
        self.lable_hm.place(height=50,width=250,x=300,y=170)

        self.value_height.place(height=50,width=100,x=550,y=20)
        self.value_bm.place(height=50,width=100,x=550,y=70)
        self.value_wm.place(height=50,width=100,x=550,y=120)
        self.value_hm.place(height=50,width=100,x=550,y=170)

        self.num.place(height=50,width=100,x=900,y=170)
# def replace_image()

def main(win,app,cap):
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    # lmain.imgtk = imgtk
    app.canvas_img.itemconfigure(app.image ,image=imgtk)
    win.after(1, main) 

win =Tk()
win.title("window")
win.geometry("1920x1080")
win.resizable(width=False,height=False)
app = Demo1(win)

cap = cv2.VideoCapture(0)
main(win,app,cap)
win.mainloop()
