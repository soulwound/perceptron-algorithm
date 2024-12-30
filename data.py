import re
from typing import List


class DataSet:

    sad_sentences: List[str]
    happy_sentences: List[str]

    @staticmethod
    def get_data(file_name: str):
        file = open(file_name, 'r', encoding='utf-8')
        return file.readlines()

    @staticmethod
    def parse_sentences_list(sentences):
        words = []
        for sentence in sentences:
            sentence = re.sub('[!@#$.,]', '', sentence)
            elem = sentence.lower()[:-1].split()
            for word in elem:
                if word not in words:
                    words.append(word)
        return words

    def parse_data(self):
        words = []
        self.sad_sentences = self.get_data('sad.txt')
        words += self.parse_sentences_list(self.sad_sentences)
        self.happy_sentences = self.get_data('happy.txt')
        words += self.parse_sentences_list(self.happy_sentences)
        return list(set(words))