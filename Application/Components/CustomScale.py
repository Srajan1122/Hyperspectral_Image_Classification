import tkinter as tk


class CustomScale(tk.Frame):
    def __init__(self, master, text, start, end, gap, state_function, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.state_function = state_function

        self.test_label = tk.Label(self, text=text, width=50)

        self.testing_set = tk.Scale(self,
                                    from_=start,
                                    to=end,
                                    resolution=gap,
                                    orient=tk.HORIZONTAL,
                                    bd=0,
                                    highlightthickness=0,
                                    length=200,
                                    command=self.update_state)

        self.test_label.grid(row=0, column=0, pady=10)
        self.testing_set.grid(row=0, column=1, pady=10)

    def update_state(self, value):
        self.state_function(value)
