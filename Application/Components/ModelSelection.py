import tkinter as tk
from Application.Components.State import State


class ModelSelection():
    def __init__(self, master, *args, **kwargs):
        self.state = State()
        self.master = master
        options = [
            'Artificial Neural Network',
            'Support Vector Machine',
        ]

        self.label = tk.Label(self.master.selectFrame, text='Select the model', background='#929292')
        self.label.grid(row=0, column=0, sticky='e')

        self.variable = tk.StringVar(self.master.selectFrame)
        self.variable.set(options[0])
        self.variable.trace('w', self.onChange)

        self.dropDown = tk.OptionMenu(self.master.selectFrame, self.variable, *options)
        self.dropDown.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        self.dropDown.config(background='white',
                             activebackground='white',
                             bd=0,
                             width=30,
                             relief=tk.FLAT)

    def onChange(self, *args):
        model = self.variable.get()
        self.state.change_model(model)
