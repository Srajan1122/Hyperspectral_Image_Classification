import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from .RunModel import RunModel
import time
from .State import State


class RunFrame(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.state = State()
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
        if self.state.image_file == '':
            messagebox.showwarning("warning", "Image is not uploaded")
            return
        if self.state.gt_file == '':
            messagebox.showwarning("warning", "Ground truth is not uploaded")
            return
        if self.state.gt_file == self.state.image_file:
            messagebox.showwarning("warning", "Check your uploaded file")
            return
        if self.state.test_set == 0:
            messagebox.showwarning("warning", "Testing size should not be zero")
            return

        try:
            if self.model.my_thread.is_alive():
                print('Please don\'t interrupt this process')
                messagebox.showwarning("warning", "Previous model is still running\nWait for it to finish")
                return
            self.model.output_frame.destroy()
        except Exception:
            pass

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

