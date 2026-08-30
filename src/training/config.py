class TrainingConfig:

    def __init__(self):

        self.batch_size = 8

        self.num_workers = 0

        self.learning_rate = 5e-5

        self.weight_decay = 0.1

        self.num_epochs = 5

        self.eval_freq = 50

        self.eval_iter = 5

        self.allowed_max_length = 1024

        self.random_seed = 123