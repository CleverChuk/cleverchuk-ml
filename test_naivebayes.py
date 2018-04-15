#!/usr/bin/env python
"""
Test and Training data from
https://github.com/hmcuesta/PDA_Book/
"""

import unittest
import os
import csv
from naivebayes import *

ROOT_DIR = "~"
CUR_DIR = ""
class TestNaiveBayes(unittest.TestCase):
    nb = NaiveBayes()
    def setUp(self):
        path = os.path.join("training.csv")
        with open(path) as fd:
            subjects = dict(csv.reader(fd, delimiter=","))
            features = subjects.keys()
            classes = subjects.values()
            self.nb.train(features, classes)

    def test_naivebayes_1(self):
        self.assertEqual(self.nb.classify("Available on Term Life - Free")[0], "spam")

    def test_naivebayes_2(self):
        path = os.path.join("test.csv")
        with open(path) as fd:
            count = correct = 0
            subjects = csv.reader(fd)
            for subject in subjects:
                count += 1
                clas = self.nb.classify(subject[0])

                if clas[0] == subject[1]:
                    correct += 1

            # print("Efficiency {0}% of {1}%".format(
            #     round(correct/float(count)*100, 2), 100.0))

            self.assertEqual(correct/float(count), 1.0)


if __name__ == '__main__':
    unittest.main()
