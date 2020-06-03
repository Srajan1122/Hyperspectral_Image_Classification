import tkinter as tk
from Application.Components.State import State
from Application.Components.ToggleButton import ToggleButton


class ButtonContent(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.state = State()

        self.normalization = ToggleButton(self,
                                          text='Normalization',
                                          command=self.state.change_normalization)
        self.feature_selection = ToggleButton(self,
                                              text='Feature Selection',
                                              command=self.state.change_feature_selection)

        self.one_hot_encoding = ToggleButton(self,
                                             text='OneHotEncoding',
                                             command=self.state.change_one_hot_encoding)

        self.normalization.grid(row=0, column=0, padx=10)
        self.feature_selection.grid(row=0, column=1, padx=10)
        self.one_hot_encoding.grid(row=0, column=2, padx=10)

        self.grid_columnconfigure((0, 1, 2), weight=1)
        self.grid_rowconfigure(0, weight=1)