import tkinter as tk


class ToggleButton(tk.Button):
    def __init__(self, master, text, *args, **kwargs):
        tk.Button.__init__(self, master, *args, **kwargs)
        self['text'] = text
        self['background'] = 'green'
        self['activebackground'] = 'green'
        self['relief'] = tk.FLAT
        self['bd'] = 0
        self['height'] = 3
        self['width'] = 15

        self.bind('<Button-1>', self.on_click)

    def on_click(self, event):
        if self['background'] == 'red':
            self.config(
                background='green',
                activebackground='green'
            )
            return
        self.config(
            background='red',
            activebackground='red'
        )
