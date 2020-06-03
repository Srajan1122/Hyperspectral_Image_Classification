import tkinter as tk
from Application.Components.CustomScale import CustomScale
from Application.Components.State import State


class ScaleContent(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.state = State()

        self.test_frame = CustomScale(self,
                                      text='Testing set',
                                      start=0,
                                      end=1,
                                      gap=0.01,
                                      state_function=self.state.set_test_size)

        self.population = CustomScale(self,
                                      text='No of Population in one generation',
                                      start=1,
                                      end=20,
                                      gap=1,
                                      state_function=self.state.set_population)

        self.generation = CustomScale(self,
                                      text='No of generation',
                                      start=1,
                                      end=20,
                                      gap=1,
                                      state_function=self.state.set_generation)

        self.crossover = CustomScale(self,
                                     text='Crossover probability',
                                     start=0,
                                     end=1,
                                     gap=0.1,
                                     state_function=self.state.set_crossover)

        self.mutation = CustomScale(self,
                                    text='Mutation probability',
                                    start=0,
                                    end=1,
                                    gap=0.1,
                                    state_function=self.state.set_mutation)

        self.test_frame.grid(row=0, column=0)
        self.population.grid(row=1, column=0)
        self.generation.grid(row=2, column=0)
        self.crossover.grid(row=3, column=0)
        self.mutation.grid(row=4, column=0)