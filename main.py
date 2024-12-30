from data import DataSet
from perceptron import Perceptron

data_set = DataSet()
dictionary = data_set.parse_data()

neuron = Perceptron(dictionary, data_set.sad_sentences, data_set.happy_sentences)
neuron.count_features("место прогулка чтение чтение")
neuron.perceptron_algorithm()

