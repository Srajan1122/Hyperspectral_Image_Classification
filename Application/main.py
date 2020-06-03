import tkinter as tk

from Application.Components.VerticalScroll import ScrollableFrame
from Application.Components.ScaleContent import ScaleContent
from Application.Components.ButtonContent import ButtonContent
from Application.Components.BrowseContent import BrowseContent
from Application.Components.RunFrame import RunFrame


class MainFrame(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self['background'] = 'white'
        self.scrollableFrame = ScrollableFrame(self)

        self.browse = BrowseContent(self.scrollableFrame.scrollable_frame)
        self.browse.grid(row=0, column=0, sticky='nsew', padx=15, pady=15)

        self.button_content = ButtonContent(self.scrollableFrame.scrollable_frame)
        self.button_content.grid(row=1, column=0, sticky='nsew')

        self.scale_content = ScaleContent(self.scrollableFrame.scrollable_frame)
        self.scale_content.grid(row=2, column=0, sticky='nsew', pady=10)

        self.runFrame = RunFrame(self.scrollableFrame.scrollable_frame)
        self.runFrame.grid(row=3, column=0, sticky='nsew', padx=15, pady=15)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Parameter Optimization with Genetic Algorithm')
    root.state('zoomed')

    mainFrame = MainFrame(root)
    mainFrame.grid(row=0, column=0, sticky='nsew')

    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    root.mainloop()
