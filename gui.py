from tkinter import *
from PIL import ImageFilter,Image
from tkinter import filedialog, messagebox
import os
import psutil
import time
import subprocess
import cv2
import fnmatch


main_win = Tk()
# main_win.iconbitmap('3/favicon.ico')


def show_entry_fields():
    print(" Age: %s\n Veg-NonVeg:%s\n Weight%s\n Hight%s\n" % (e1.get(), e2.get(),e3.get(), e4.get()))

def Weight_Loss():
    print(" Age: %s\n Veg-NonVeg:%s\n Weight%s\n Hight%s\n" % (e1.get(), e2.get(),e3.get(), e4.get()))

def Weight_Gain():
    print(" Age: %s\n Veg-NonVeg:%s\n Weight%s\n Hight%s\n" % (e1.get(), e2.get(),e3.get(), e4.get()))

def Healthy():
    print(" Age: %s\n Veg-NonVeg:%s\n Weight%s\n Hight%s\n" % (e1.get(), e2.get(),e3.get(), e4.get()))
#master = tk.Tk()
Label(main_win,text="Age").grid(row=0,column=0,sticky=W,pady=4)
Label(main_win,text="veg/Non veg").grid(row=1,column=0,sticky=W,pady=4)
Label(main_win,text="Weight").grid(row=2,column=0,sticky=W,pady=4)
Label(main_win,text="Height").grid(row=3,column=0,sticky=W,pady=4)
# Label(main_win,text="Age").grid(row=2,column=0)

e1 = Entry(main_win)
e2 = Entry(main_win)
e3 = Entry(main_win)
e4 = Entry(main_win)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)

Button(main_win,text='Quit',command=main_win.quit).grid(row=5,column=0,sticky=W,pady=4)
Button(main_win,text='Weight Loss',command=Weight_Loss).grid(row=1,column=4,sticky=W,pady=4)
Button(main_win,text='Weight Gain',command=Weight_Loss).grid(row=2,column=4,sticky=W,pady=4)
Button(main_win,text='Healthy',command=Weight_Loss).grid(row=3,column=4,sticky=W,pady=4)
main_win.geometry("400x200")
main_win.wm_title("DIET RECOMMENDATION SYSTEM")

main_win.mainloop()
