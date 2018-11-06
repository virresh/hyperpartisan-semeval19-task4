#!/usr/bin/env python

"""Random baseline for the PAN19 hyperpartisan news detection task"""
# Version: 2018-09-18

# Parameters:
# --inputDataset=<directory>
#   Directory that contains the articles XML file with the articles for which a prediction should be made.
# --outputDir=<directory>
#   Directory to which the predictions will be written. Will be created if it does not exist.

from __future__ import division

import os
import getopt
import sys
import xml.sax
import random

# from library.sentiment import SentimentAnalysis
import pandas as pd

random.seed(42)
runOutputFileName = "test_data.txt"

"""Global Variables"""
num_articles = 0
char_len = 0

def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["inputDataset=", "outputDir="]
        opts, _ = getopt.getopt(sys.argv[1:], "d:o:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    outputDir = "undefined"

    for opt, arg in opts:
        if opt in ("-d", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
        else:
            assert False, "Unknown option."
    if inputDataset == "undefined":
        sys.exit("Input dataset, the directory that contains the articles XML file, is undefined. Use option -d or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The input dataset folder does not exist (%s)." % inputDataset)

    if outputDir == "undefined":
        sys.exit("Output path, the directory into which the predictions should be written, is undefined. Use option -o or --outputDir.")
    elif not os.path.exists(outputDir):
        os.mkdir(outputDir)

    return (inputDataset, outputDir)


########## SAX ##########

class HyperpartisanNewsRandomPredictor(xml.sax.ContentHandler):
    def __init__(self, outFile, fname, label):
        xml.sax.ContentHandler.__init__(self)
        self.outFile = outFile
        self.articleId = fname
        self.article_content = ''
        self.article_title = ''
        self.label = label
        # self.sentiment_analyser = SentimentAnalysis(filename='library/SentiWordNet.txt')
        # self.outFile.write('articleId' + "," + 'content_length' + "," + 'content_polarity' + "," + 'title_length' + "," + 'title_polarity' + "," + 'article_date' + "\n")

    def startElement(self, name, attrs):
        self.current_element = name

    def endElement(self, name):
        global num_articles, char_len
        if name=='article':
            num_articles += 1
            # char_len += len(self.article_content + self.article_title)
            self.article_content = (self.article_content+" "+self.article_title).lower().replace('\n',' ')
            # print(self.article_content)
            # polarity_content = self.sentiment_analyser.score(self.article_content)
            # polarity_title = self.sentiment_analyser.score(self.article_title)
            self.outFile.write(self.articleId + " " + str(self.label) + " " + self.article_content + "\n")

    def characters(self, content):
        if self.current_element == "mainText":
            self.article_content += content
        elif self.current_element == "title":
            self.article_title += content


########## MAIN ##########


def main(inputDataset, outputDir):
    """Main method of this module."""
    global num_articles, char_len
    df = pd.read_csv('../BuzzFeedData/overview.csv')
    df = df[['XML', 'orientation']]
    df['orientation'] = df['orientation'].apply(lambda x: 0 if x == 'mainstream' else 1)
    df['XML'] = df['XML'].apply(lambda x: int(x[:-4]))

    with open(outputDir + "/" + runOutputFileName, 'w') as outFile:
        num_articles=0
        # outFile.write('articleId' + "," + 'content_length' + "," + 'content_polarity' + "," + 'title_length' + "," + 'title_polarity' + "\n")
        for file in os.listdir(inputDataset):
            if file.endswith(".xml"):
                char_len = 0
                with open(inputDataset + "/" + file) as inputRunFile:
                    label = df[df['XML']==int(file[:-4])]['orientation'].iloc[0]
                    xml.sax.parse(inputRunFile, HyperpartisanNewsRandomPredictor(outFile, file[:-4], label))
        print('\tNumber of Articles == ', num_articles)
        print('\tAvg char length == ', char_len/num_articles)

    print("The predictions have been written to the output folder.")

if __name__ == '__main__':
    main(*parse_options())
