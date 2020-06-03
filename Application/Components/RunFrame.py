import tkinter as tk
from tkinter import ttk
from Application.Components.RunModel import RunModel
import time


class RunFrame(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self['background'] = '#929292'

        self.run = tk.Button(self,
                             text='Run Model',
                             height=2,
                             width=10,
                             bd=0,
                             relief=tk.FLAT,
                             command=self.run)
        self.run.grid(row=0, column=0, padx=10, pady=20)

        self.progressbar = ttk.Progressbar(self,
                                           orient='horizontal',
                                           length='286',
                                           mode='determinate')

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def run(self):
        self.model = RunModel(self)
        self.progressbar.grid(row=1, column=0, pady=10)
        self.progressbar.start()

    def run_progressBar(self):
        self.progressbar['maximum'] = 100

        for i in range(101):
            time.sleep(0.05)
            self.progressbar['value'] = i
            self.progressbar.update()

        self.progressbar['value'] = 0

