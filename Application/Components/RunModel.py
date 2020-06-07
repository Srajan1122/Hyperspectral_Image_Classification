import numpy as np
import pandas as pd
import random
from pyeasyga import pyeasyga
from scipy import io
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import sklearn.svm
from sklearn.preprocessing import OneHotEncoder
import threading
import spectral as sp
from PIL import ImageTk, Image
# import math

import tkinter as tk
from Application.Components.State import State


class RunModel:

    def __init__(self, master):
        self.master = master
        self.state = State()
        self.population = int(self.state.population)
        self.generation = int(self.state.generation)
        self.mutation = float(self.state.mutation)
        self.crossover = float(self.state.crossover)

        self.image = np.array(self.load_data(self.state.image_file))
        self.image_shape = self.image.shape
        self.gt = np.array(self.load_data(self.state.gt_file))

        sp.save_rgb('image.jpg', self.image, [43, 21, 11])
        sp.save_rgb('gt.jpg', self.gt, colors=sp.spy_colors)
        self.display_input_image()

        resize_data = self.resize_data(self.image, self.gt)
        cleaned_data = self.drop_if_gt_zero(resize_data)
        X, y = self.feature_target(cleaned_data)

        if self.state.feature_selection:
            X = self.feature_selection(X, y)

        if self.state.one_hot_encoding:
            y = self.one_hot_encoding(y)

        if self.state.normalization:
            X = self.standardizing(X)

        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(X, y, float(self.state.test_set))

        self.model = self.genetic_algorithm(population=self.population,
                                            generation=self.generation,
                                            crossover=self.crossover,
                                            mutation=self.mutation)
        self.my_thread = threading.Thread(target=self.run_in_thread, args=(self.X_train, self.y_train, X))
        self.my_thread.start()
        # print(self.my_thread.is_alive())

    def display_input_image(self):
        self.image_1 = Image.open('image.jpg')
        self.image_1 = self.image_1.resize((250, 250), Image.ANTIALIAS)
        self.gt_1 = Image.open('gt.jpg')
        self.gt_1 = self.gt_1.resize((250, 250), Image.ANTIALIAS)
        self.image_1 = ImageTk.PhotoImage(self.image_1)
        self.gt_1 = ImageTk.PhotoImage(self.gt_1)

        self.input_image_frame = tk.Frame(self.master.master)
        self.input_image_frame.grid(row=4, column=0, sticky='nsew', pady=15, padx=15)

        self.input_image_label = tk.Label(self.input_image_frame, text='Input image')
        self.input_gt_label = tk.Label(self.input_image_frame, text='Input ground truth')
        self.image_label = tk.Label(self.input_image_frame, image=self.image_1)
        self.gt_label = tk.Label(self.input_image_frame, image=self.gt_1)

        self.input_image_label.grid(row=0, column=0, sticky='nsew')
        self.input_gt_label.grid(row=0, column=1, sticky='nsew')
        self.image_label.grid(row=1, column=0, sticky='nsew')
        self.gt_label.grid(row=1, column=1, sticky='nsew')

        self.input_image_frame.grid_columnconfigure((0, 1), weight=1)
        self.input_image_frame.grid_rowconfigure((0, 1), weight=1)

    def run_in_thread(self, X_train, y_train, X):
        self.model.run()
        accuracy, best_parameters = self.best_parameters(self.model)
        data = self.initial_data
        self.X_train = X_train
        self.y_train = y_train
        self.X = X

        self.output_frame = tk.Frame(self.master.master)
        self.output_frame.grid(row=5, column=0, sticky='nsew', pady=15, padx=15)

        if self.state.model == 'Artificial Neural Network':
            self.Mlp(best_parameters, data, self.X_train, self.y_train, self.X, accuracy)
        if self.state.model == 'Support Vector Machine':
            self.Svm(best_parameters, data, self.X_train, self.y_train, self.X, accuracy)

        self.output_image_frame = tk.Frame(self.output_frame)
        self.output_image_frame.grid(row=0, column=1, sticky='nsew')

        self.input_gt_label = tk.Label(self.output_image_frame, text='Output ground truth')
        self.gt_label = tk.Label(self.output_image_frame, image=self.predicted_gt)

        self.input_gt_label.grid(row=0, column=0, sticky='nsew')
        self.gt_label.grid(row=1, column=0, sticky='nsew')

        self.output_image_frame.grid_columnconfigure(0, weight=1)
        self.output_image_frame.grid_rowconfigure((0, 1), weight=1)

        self.output_frame.grid_columnconfigure((0, 1), weight=1)
        self.output_frame.grid_rowconfigure(0, weight=1)

        self.master.progressbar.stop()
        self.master.progressbar.grid_forget()

    def load_data(self, path):
        loaded_dataset = io.loadmat(path)
        for key, value in loaded_dataset.items():
            if isinstance(value, type(np.array([1]))):
                image = loaded_dataset[key]

        return image

    def resize_data(self, image, gt):
        image_with_gt = np.dstack((image, gt))
        final_output_data = image_with_gt.reshape(gt.size, image.shape[2] + 1)
        return final_output_data

    def drop_if_gt_zero(self, data):
        data = pd.DataFrame(data)
        self.zero_data = data.index[data.iloc[:, -1] == 0].tolist()
        data = data[data.iloc[:, -1] != 0]
        return data

    def feature_target(self, data):
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return x, y

    def feature_selection(self, feature, target):
        x = SelectKBest(f_classif, k=int(feature.shape[1] * .75)).fit_transform(feature, target)
        return x

    def one_hot_encoding(self, target):
        if self.state.model != 'Support Vector Machine':
            print('not svm')
            onehotencoder = OneHotEncoder()
            y = onehotencoder.fit_transform(np.array(target).reshape(-1, 1)).toarray()
            return y
        return target

    def standardizing(self, feature):
        x = preprocessing.scale(feature)
        return x

    def split_data(self, feature, target, t):
        X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=t, random_state=1)
        return X_train, X_test, y_train, y_test

    def initialize_population(self):

        if self.state.model == 'Artificial Neural Network':
            print('ann')
            hidden_layer_sizes = []
            for _ in range(200):
                hidden_layer = []
                for _ in range(5):
                    hidden_layer.append(random.randrange(50, 250))
                hidden_layer_sizes.append(hidden_layer)

            alpha = list(np.logspace(-5, 3, 9))
            batch_size = list(range(200, 1001, 50))
            learning_rate_init = np.logspace(-1, -5, 5)
            n_iter_no_change = list(range(10, 310, 10))

            population = []
            population.append(hidden_layer_sizes)
            population.append(alpha)
            population.append(batch_size)
            population.append(learning_rate_init)
            population.append(n_iter_no_change)

        if self.state.model == 'Support Vector Machine':
            print('svm')
            C = list(np.logspace(-3, 3, 7))
            kernel = ['linear', 'poly', 'rbf', 'sigmoid']
            degree = list(x for x in range(3, 8))
            gamma = ['scale', 'auto'] + list(np.logspace(-5, -1, 5))
            shrinking = [True, False]
            probability = [True, False]
            decision_function_shape = ['ovo', 'ovr']

            population = []
            population.append(C)
            population.append(kernel)
            population.append(degree)
            population.append(gamma)
            population.append(shrinking)
            population.append(probability)
            population.append(decision_function_shape)

        return list(population)

    def create_individual(self, data):
        choice = []
        for i in range(len(data)):
            choice.append(random.choice(list(range(len(data[i])))))
        return choice

    def fitness_function(self, individual, data):
        if self.state.model == 'Artificial Neural Network':
            print('Next Individual')
            model = MLPClassifier(
                hidden_layer_sizes=data[0][individual[0]],
                activation='relu',
                solver='adam',
                alpha=data[1][individual[1]],
                batch_size=data[2][individual[2]],
                learning_rate='constant',
                learning_rate_init=data[3][individual[3]],
                power_t=0.5,
                max_iter=30,
                shuffle=True,
                random_state=1,
                tol=0.0001,
                verbose=False,
                warm_start=False,
                momentum=0.9,
                nesterovs_momentum=True,
                early_stopping=False,
                validation_fraction=0.18,  # 0.33 0.18
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08,
                n_iter_no_change=data[4][individual[4]],
                max_fun=15000)

            model.fit(self.X_train, self.y_train)
            prediction = model.predict(self.X_test)
            score = accuracy_score(self.y_test, prediction)

        if self.state.model == "Support Vector Machine":
            print('svm')
            print('Next Individual')
            model = sklearn.svm.SVC(C=data[0][individual[0]],
                                    kernel=data[1][individual[1]],
                                    degree=data[2][individual[2]],
                                    gamma=data[3][individual[3]],
                                    coef0=0.0,
                                    shrinking=data[4][individual[4]],
                                    probability=data[5][individual[5]],
                                    tol=0.001,
                                    cache_size=200,
                                    class_weight=None,
                                    verbose=True,
                                    max_iter=-1,
                                    decision_function_shape=data[6][individual[6]],
                                    break_ties=False,
                                    random_state=None)
            model.fit(self.X_train, self.y_train)
            prediction = model.predict(self.X_test)
            score = accuracy_score(self.y_test, prediction)
        return score

    def genetic_algorithm(self, population, generation, crossover, mutation):
        self.initial_data = self.initialize_population()
        ga = pyeasyga.GeneticAlgorithm(seed_data=self.initial_data,
                                       population_size=population,
                                       generations=generation,
                                       crossover_probability=crossover,
                                       mutation_probability=mutation,
                                       elitism=True,
                                       maximise_fitness=True)
        ga.create_individual = self.create_individual
        ga.fitness_function = self.fitness_function
        return ga

    def best_parameters(self, model):
        accuracy, best_parameters = model.best_individual()[0], model.best_individual()[1]
        for index, i in enumerate(self.initial_data):
            print(i[best_parameters[index]])
        return accuracy, best_parameters

    def Mlp(self, best_parameters, data, X_train, y_train, X, accuracy):
        model = MLPClassifier(hidden_layer_sizes=tuple(data[0][best_parameters[0]]),
                              activation='relu',
                              solver='adam',
                              alpha=data[1][best_parameters[1]],
                              batch_size=data[2][best_parameters[2]],
                              learning_rate='constant',
                              learning_rate_init=data[3][best_parameters[3]],
                              power_t=0.5,
                              max_iter=10,
                              shuffle=True,
                              random_state=1,
                              tol=0.0001,
                              verbose=False,
                              warm_start=True,
                              momentum=0.9,
                              nesterovs_momentum=True,
                              early_stopping=False,
                              validation_fraction=0.18,  # 0.33 0.18
                              beta_1=0.9,
                              beta_2=0.999,
                              epsilon=1e-08,
                              n_iter_no_change=data[4][best_parameters[4]],
                              max_fun=15000)
        print('fitting')
        model.fit(X_train, y_train)
        prediction = model.predict(X)

        if len(prediction.shape) == 2:
            predicted_gt_1 = np.argmax(prediction, axis=1)
            predicted_gt_1_list = list(predicted_gt_1)
            predicted_gt_1_list = [x + 1 for x in predicted_gt_1_list]
        else:
            predicted_gt_1_list = prediction

        for i in self.zero_data:
            predicted_gt_1_list = np.insert(predicted_gt_1_list, i, 0)

        # predicted_gt_size = int(math.sqrt(predicted_gt_1_list.shape[0]))

        self.predicted_gt_1_list = predicted_gt_1_list.reshape(self.image_shape[0], self.image_shape[1])

        sp.save_rgb('predicted_gt.jpg', self.predicted_gt_1_list, colors=sp.spy_colors)

        self.predicted_gt = Image.open('predicted_gt.jpg')
        self.predicted_gt = self.predicted_gt.resize((250, 250), Image.ANTIALIAS)
        self.predicted_gt = ImageTk.PhotoImage(self.predicted_gt)

        self.output_detail_frame = tk.Frame(self.output_frame)
        self.output_detail_frame.grid(row=0, column=0, sticky='nsew')

        self.accuracy = tk.Label(self.output_detail_frame,
                                 text='Accuracy: ' + str(accuracy))

        self.hidden_layer = tk.Label(self.output_detail_frame,
                                     text='hidden_layer_sizes: ' + str(data[0][best_parameters[0]]))
        self.alpha = tk.Label(self.output_detail_frame,
                              text='alpha: ' + str(data[1][best_parameters[1]]))
        self.degree = tk.Label(self.output_detail_frame,
                               text='batch_size: ' + str(data[2][best_parameters[2]]))
        self.learning_rate_init = tk.Label(self.output_detail_frame,
                                           text='learning_rate_init: ' + str(data[3][best_parameters[3]]))
        self.n_iter_no_change = tk.Label(self.output_detail_frame,
                                         text='n_iter_no_change: ' + str(data[4][best_parameters[4]]))

        self.accuracy.grid(row=0, column=0, sticky='nsew')
        self.hidden_layer.grid(row=1, column=0, sticky='nsew')
        self.alpha.grid(row=2, column=0, sticky='nsew')
        self.degree.grid(row=3, column=0, sticky='nsew')
        self.learning_rate_init.grid(row=4, column=0, sticky='nsew')
        self.n_iter_no_change.grid(row=5, column=0, sticky='nsew')

        self.output_detail_frame.grid_rowconfigure((0, 1, 2, 3, 4, 5), weight=1)
        self.output_detail_frame.columnconfigure(0, weight=1)

    def Svm(self, best_parameters, data, X_train, y_train, X, accuracy):
        model = sklearn.svm.SVC(C=data[0][best_parameters[0]],
                                kernel=data[1][best_parameters[1]],
                                degree=data[2][best_parameters[2]],
                                gamma=data[3][best_parameters[3]],
                                coef0=0.0,
                                shrinking=data[4][best_parameters[4]],
                                probability=data[5][best_parameters[5]],
                                tol=0.001,
                                cache_size=200,
                                class_weight=None,
                                verbose=True,
                                max_iter=-1,
                                decision_function_shape=data[6][best_parameters[6]],
                                break_ties=False,
                                random_state=None)

        print('fitting')
        model.fit(X_train, y_train)
        prediction = model.predict(X)
        predicted_gt_1_list = prediction

        for i in self.zero_data:
            predicted_gt_1_list = np.insert(predicted_gt_1_list, i, 0)

        self.predicted_gt_1_list = predicted_gt_1_list.reshape(self.image_shape[0], self.image_shape[1])

        sp.save_rgb('predicted_gt.jpg', self.predicted_gt_1_list, colors=sp.spy_colors)

        self.predicted_gt = Image.open('predicted_gt.jpg')
        self.predicted_gt = self.predicted_gt.resize((250, 250), Image.ANTIALIAS)
        self.predicted_gt = ImageTk.PhotoImage(self.predicted_gt)

        self.output_detail_frame = tk.Frame(self.output_frame)
        self.output_detail_frame.grid(row=0, column=0, sticky='nsew')

        self.accuracy = tk.Label(self.output_detail_frame,
                                 text='Accuracy: ' + str(accuracy))

        self.C = tk.Label(self.output_detail_frame,
                                     text='C: ' + str(data[0][best_parameters[0]]))
        self.kernel = tk.Label(self.output_detail_frame,
                              text='kernel: ' + str(data[1][best_parameters[1]]))
        self.degree = tk.Label(self.output_detail_frame,
                               text='degree: ' + str(data[2][best_parameters[2]]))
        self.gamma = tk.Label(self.output_detail_frame,
                                           text='gamma: ' + str(data[3][best_parameters[3]]))
        self.shrinking = tk.Label(self.output_detail_frame,
                                         text='shrinking: ' + str(data[4][best_parameters[4]]))
        self.probability = tk.Label(self.output_detail_frame,
                                         text='probability: ' + str(data[5][best_parameters[5]]))
        self.decision_function_shape = tk.Label(self.output_detail_frame,
                                         text='decision_function_shape: ' + str(data[6][best_parameters[6]]))

        self.accuracy.grid(row=0, column=0, sticky='nsew')
        self.C.grid(row=1, column=0, sticky='nsew')
        self.kernel.grid(row=2, column=0, sticky='nsew')
        self.degree.grid(row=3, column=0, sticky='nsew')
        self.gamma.grid(row=4, column=0, sticky='nsew')
        self.shrinking.grid(row=5, column=0, sticky='nsew')
        self.probability.grid(row=5, column=0, sticky='nsew')
        self.decision_function_shape.grid(row=5, column=0, sticky='nsew')

        self.output_detail_frame.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7), weight=1)
        self.output_detail_frame.columnconfigure(0, weight=1)
