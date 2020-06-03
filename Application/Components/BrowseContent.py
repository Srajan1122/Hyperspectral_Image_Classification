import tkinter as tk
from Application.Components.State import State
from tkinter import filedialog


class BrowseContent(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.state = State()

        self.upperframe = tk.Frame(self,
                                   background='#929292')
        self.lowerframe = tk.Frame(self,
                                   background='#929292')

        self.label = tk.Label(self.upperframe,
                              text="Upload the corrected image data and ground truth data",
                              background='#929292')

        self.browse_image = tk.Button(self.lowerframe,
                                      text='Upload image',
                                      height=2,
                                      width=20,
                                      bd=0,
                                      relief=tk.FLAT,
                                      command=self.image_browse)
        self.browse_gt = tk.Button(self.lowerframe,
                                   text='Upload ground_truth',
                                   height=2,
                                   width=20,
                                   bd=0,
                                   relief=tk.FLAT,
                                   command=self.gt_browse)

        self.label.grid(row=0, column=0, padx=5, pady=10)
        self.browse_image.grid(row=1, column=0, padx=5, pady=20, sticky='e')
        self.browse_gt.grid(row=1, column=1, padx=5, pady=20, sticky='w')

        self.upperframe.grid(row=0, column=0, sticky='nsew')
        self.lowerframe.grid(row=1, column=0, sticky='nsew')

        self.upperframe.grid_rowconfigure(0, weight=1)
        self.upperframe.grid_columnconfigure(0, weight=1)

        self.lowerframe.grid_columnconfigure((0, 1), weight=1)
        self.lowerframe.grid_rowconfigure(0, weight=1)

        self.grid_rowconfigure((0, 1), weight=1)
        self.columnconfigure(0, weight=1)

    def image_browse(self):
        self.image = filedialog.askopenfilename(initialdir='/', title='select a file', filetype=[('Matlab', ".mat")])
        self.state.set_image_file(self.image)

    def gt_browse(self):
        self.gt = filedialog.askopenfilename(initialdir='/', title='select a file', filetype=[('Matlab', ".mat")])
        self.state.set_gt_file(self.gt)
