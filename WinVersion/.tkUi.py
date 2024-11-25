#from tkinter import Tk, Button, Label, Entry
from customtkinter import *
import Connect_NN as cn
import os
import subprocess as sp

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#sp.call(['.\setup.bat'])

'''
r = Tk()
r.maxsize(450, 550)
r.minsize(350, 450)
r.title("WCNN Main Interface")
button = Button(r, text='Send', width=10, command=r.destroy)
button.pack()
r.mainloop()
'''

def start():
    window.destroy()
    cn.train()

def filter():
    if entryBox.get() != "":
        label.configure(text='Message blocked' if cn.talk(entryBox.get())[0][0] == 1 else f'You: {entryBox.get()}')
    else:
        label.configure("- Write something -")


window = CTk()
window.geometry("300x490")
window.title("WCNN Main Interface")
window.iconbitmap("logo.ico", "logo.ico")

tframe = CTkFrame(window, width=290, height=50)

train = CTkButton(tframe, text='Train', width=10, command=start)

tsframe = CTkFrame(window, width=290, height=417)
entryBox = CTkEntry(tsframe, placeholder_text="Enter your message here", width=200)
talk = CTkButton(tsframe, text='Send', width=10, command=filter)
label = CTkLabel(tsframe, width=180, text="Send a message for me!")

tframe.pack(padx=10, pady=5)
tsframe.pack(padx=10, pady=5)

tframe.pack_propagate(False)
tsframe.pack_propagate(False)

train.pack(pady=10)
label.pack(pady=5)
entryBox.pack(pady=5)
talk.pack(pady=5)

window.mainloop()
