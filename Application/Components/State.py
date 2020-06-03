class State:
    normalization = True
    feature_selection = True
    one_hot_encoding = True
    test_set = 0
    population = 1
    generation = 1
    mutation = 0
    crossover = 0

    image_file = ''
    gt_file = ''

    @classmethod
    def change_normalization(cls):
        cls.normalization = not cls.normalization

    @classmethod
    def change_feature_selection(cls):
        cls.feature_selection = not cls.feature_selection

    @classmethod
    def change_one_hot_encoding(cls):
        cls.one_hot_encoding = not cls.one_hot_encoding

    @classmethod
    def set_image_file(cls, path):
        cls.image_file = path

    @classmethod
    def set_gt_file(cls, path):
        cls.gt_file = path

    @classmethod
    def set_test_size(cls, size):
        cls.test_set = size

    @classmethod
    def set_population(cls, size):
        cls.population = size

    @classmethod
    def set_generation(cls, size):
        cls.generation = size

    @classmethod
    def set_mutation(cls, size):
        cls.mutation = size

    @classmethod
    def set_crossover(cls, size):
        cls.crossover = size
