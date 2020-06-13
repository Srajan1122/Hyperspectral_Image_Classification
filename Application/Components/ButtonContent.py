import tkinter as tk
from .State import State
from .ToggleButton import ToggleButton
from .ModelSelection import ModelSelection


class ButtonContent(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.state = State()

        self.toggleFrame = tk.Frame(self)
        self.toggleFrame.grid(row=0, column=0, sticky='nsew')

        self.selectFrame = tk.Frame(self, background='#929292')
        self.selectFrame.grid(row=1, column=0, sticky='nsew', padx=15, pady=15)

        self.normalization = ToggleButton(self.toggleFrame,
                                          text='Normalization',
                                          command=self.state.change_normalization)
        self.feature_selection = ToggleButton(self.toggleFrame,
                                              text='Feature Selection',
                                              command=self.state.change_feature_selection)

        self.feature_extraction = ToggleButton(self.toggleFrame,
                                              text='Feature Extraction',
                                              command=self.state.change_feature_extraction)

        self.one_hot_encoding = ToggleButton(self.toggleFrame,
                                             text='OneHotEncoding',
                                             command=self.state.change_one_hot_encoding)

        self.selectModel = ModelSelection(self)

        self.normalization.grid(row=0, column=0, padx=10)
        self.feature_selection.grid(row=0, column=1, padx=10)
        self.feature_extraction.grid(row=0, column=2, padx=10)
        self.one_hot_encoding.grid(row=0, column=3, padx=10)

        self.toggleFrame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.toggleFrame.grid_rowconfigure(0, weight=1)

        self.selectFrame.grid_rowconfigure(0, weight=1)
        self.selectFrame.grid_columnconfigure((0, 1), weight=1)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

