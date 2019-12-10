import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import os

class ScrolledFrame(tk.Frame):
    IMAGE_HEIGHT = 100
    IMAGE_WIDTH = 150
    LABEL_HEIGHT = 20
    GAP = 25
    current_x = 0
    current_y = 0
    label_x_offset = IMAGE_WIDTH / 2
    label_y_offset = IMAGE_HEIGHT + 10
    
    def __init__(self, width, height, title):
        self.window_width = width
        self.window_height = height
        self._root = tk.Tk()
        self._root.title(title)
        self._root.geometry('{}x{}'.format(width, height))
        self._root.resizable(width=False, height=False)
        super().__init__(self._root)

        self._canvas = tk.Canvas(self)
        self._canvas.grid(row=0, column=0, sticky='news')

        self._vertical_bar = tk.Scrollbar(self, orient='vertical', command=self._canvas.yview)
        self._vertical_bar.grid(row=0, column=1, sticky='ns')
        self._canvas.configure(yscrollcommand=self._vertical_bar.set)

        self.inner = tk.Frame(self._canvas, bg='black')
        self._window = self._canvas.create_window((0, 0), window=self.inner, anchor='nw')

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.inner.bind('<Configure>', self.resize)
        self._canvas.bind('<Configure>', self.frame_width)

        self.inner_canvas_width = self.window_width
        self.inner_canvas_height = self.window_height
        self.inner_canvas = Canvas(self.inner, width=self.inner_canvas_width, 
                                   height=self.inner_canvas_height) 
        self.inner_canvas.pack() 
        
    def frame_width(self, event):
        canvas_width = event.width
        self._canvas.itemconfig(self._window, width = canvas_width)

    def resize(self, event=None): 
        self._canvas.configure(scrollregion=self._canvas.bbox('all'))

    def insert_image(self, image, label):
        self.inner_canvas.create_image(self.current_x, self.current_y, anchor=NW, image=image)
        self.inner_canvas.create_text(self.current_x + self.label_x_offset, 
                                      self.current_y + self.label_y_offset, text=label)
        self.update_cursor()
        
    # update position of current cursor after adding an image
    def update_cursor(self):
        self.current_x += self.IMAGE_WIDTH
        
        # the cursor goes to the head of a new line when it meets the end of the current line
        if self.current_x > 1000:
            self.current_x = 0
            self.current_y += self.IMAGE_HEIGHT + self.LABEL_HEIGHT
            
            # expand size of inner panel when needed
            if self.current_y + self.IMAGE_HEIGHT + self.LABEL_HEIGHT > self.inner_canvas_height:
                self.inner_canvas_height = self.current_y + self.IMAGE_HEIGHT + self.LABEL_HEIGHT
                self.inner_canvas.config(width=self.inner_canvas_width, height=self.inner_canvas_height)
            
        else:
            self.current_x += self.GAP

    def start(self):
        self._root.mainloop()   
        
        
# test
window = ScrolledFrame(width=1025, height=500, title='result')
window.pack(expand=True, fill='both')
image_list = []

current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_dataset = current_dir + '/../dataset/example/original/'

for i in range(14):
    img = ImageTk.PhotoImage(Image.open(path_to_dataset + "1.jpg").resize((150, 100), Image.ANTIALIAS))  
    image_list.append(img)
    window.insert_image(img, label = 'cow1')
    
    img = ImageTk.PhotoImage(Image.open(path_to_dataset + "2.jpg").resize((150, 100), Image.ANTIALIAS))
    image_list.append(img)    
    window.insert_image(img, label = 'cow2')
    
    img = ImageTk.PhotoImage(Image.open(path_to_dataset + "3.jpg").resize((150, 100), Image.ANTIALIAS))  
    image_list.append(img)  
    window.insert_image(img, label = ['cow3', 'grass'])
    
    img = ImageTk.PhotoImage(Image.open(path_to_dataset + "4.jpg").resize((150, 100), Image.ANTIALIAS))  
    image_list.append(img)  
    window.insert_image(img, label = ['cow4', 'grass'])

window.start()