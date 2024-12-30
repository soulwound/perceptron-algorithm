import matplotlib
import numpy as np
from typing import List
import random
from matplotlib import pyplot as plt

matplotlib.use("QtAgg")


class Perceptron:

    weights: np.ndarray
    vocabulary: np.ndarray
    features: np.ndarray
    bias: float
    labels: np.ndarray
    sentences: List[str]
    learning_rate: float

    def __init__(self, vocabulary: List[str], sad: List[str], happy: List[str]):
        matplotlib.use("QtAgg")
        self.weights = np.array([0] * len(vocabulary))
        self.vocabulary = np.array(vocabulary)
        self.features = np.array([0] * len(vocabulary))
        self.sentences = sad
        self.labels = np.array([0] * len(sad) + [1] * len(happy))
        self.sentences += happy
        self.learning_rate = 0.02

    def perceptron_trick(self, sentence: str, last_epochs: bool):
        label = self.labels[self.sentences.index(sentence)]
        pred = self.predict(sentence)
        if last_epochs:
            print(sentence, pred, label)
        for i in range(len(self.weights)):
            self.weights[i] += (label - pred) * self.features[i] * self.learning_rate
            self.bias += (label - pred) * self.learning_rate
        return self.weights, self.bias

    def perceptron_algorithm(self, epochs=99999):
        self.weights = np.array([1.0] * len(self.vocabulary))
        self.bias = 0.0
        errors = []
        for epoch in range(epochs):
            error = self.mean_error()
            errors.append(error)
            i = random.randint(0, len(self.sentences) - 1)
            self.weights, self.bias = self.perceptron_trick(self.sentences[i], epoch>99000)
        plt.show()
        plt.scatter(range(epochs), errors, s=3)
        plt.show()



    def predict(self, sentence: str):
        return self.step(self.score(sentence))

    def score(self, sentence: str):
        self.count_features(sentence)
        return self.features.dot(self.weights) + self.bias

    # Считает, сколько раз встречаются слова в предложении
    def count_features(self, sentence: str):
        self.features = np.array([0] * len(self.vocabulary))
        for word in sentence.split():
            self.features[np.where(self.vocabulary == word)] += 1

    def error(self, sentence: str):
        pred = self.predict(sentence)
        if pred == self.labels[self.sentences.index(sentence)]:
            return 0
        else:
            return np.abs(self.score(sentence))

    def mean_error(self):
        total_error = 0
        for i in range(len(self.sentences)):
            total_error += self.error(self.sentences[i])
        return total_error/len(self.sentences)


    @staticmethod
    def step(x):
        if x >= 0:
            return 1
        else:
            return 0