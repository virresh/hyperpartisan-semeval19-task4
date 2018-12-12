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
        long_options = ["inputDataset=", "outputDir=", "truthDir="]
        opts, _ = getopt.getopt(sys.argv[1:], "d:o:t:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    outputDir = "undefined"
    truthDir = "undefined"

    for opt, arg in opts:
        if opt in ("-d", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
        elif opt in ("-t", "--truthDir"):
            truthDir = arg
        else:
            assert False, "Unknown option."
    if inputDataset == "undefined":
        sys.exit("Input dataset, the directory that contains the articles XML file, is undefined. Use option -d or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The input dataset folder does not exist (%s)." % inputDataset)

    if truthDir == "undefined":
        sys.exit("Ground Truth , the directory that contains the ground truth XML file, is undefined. Use option -d or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The ground truth folder does not exist (%s)." % inputDataset)

    if outputDir == "undefined":
        sys.exit("Output path, the directory into which the predictions should be written, is undefined. Use option -o or --outputDir.")
    elif not os.path.exists(outputDir):
        os.mkdir(outputDir)

    return (inputDataset, outputDir, truthDir)


####### Ground Truth Parser #######

groundTruth = {}
class HyperpartisanNewsGroundTruthHandler(xml.sax.ContentHandler):
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)

    def startElement(self, name, attrs):
        if name == "article":
            articleId = attrs.getValue("id")
            hyperpartisan = attrs.getValue("hyperpartisan")
            groundTruth[articleId] = hyperpartisan

########## SAX ##########

class HyperpartisanNewsDataExtractor(xml.sax.ContentHandler):
    def __init__(self, outFile, labels):
        xml.sax.ContentHandler.__init__(self)
        self.outFile = outFile
        self.articleId = None
        self.article_title = ''
        self.article_content = ''
        self.labels = labels
        # self.outFile.write('articleId' + "," + 'hyperpartisan' + "," + 'text' + "\n")

    def startElement(self, name, attrs):
        if name=='article':
            self.current_element = name
            self.articleId = attrs.getValue("id")
            self.label = 1 if self.labels[self.articleId] == 'true' else 0
            self.article_title = attrs.getValue("title")
            self.article_content = ''

    def endElement(self, name):
        global num_articles, char_len
        if name=='article':
            num_articles += 1
            self.article_content = (self.article_content+" "+self.article_title).lower().replace('\n',' ')
            # print(self.article_content)
            self.outFile.write(self.articleId + " " + str(self.label) + " " + self.article_content + "\n")

    def characters(self, content):
        self.article_content += content


########## MAIN ##########


def main(inputDataset, outputDir, truthDir):
    """Main method of this module."""
    global num_articles, char_len

    for file in os.listdir(truthDir):
        if file.endswith(".xml"):
            with open(truthDir + "/" + file) as inputRunFile:
                xml.sax.parse(inputRunFile, HyperpartisanNewsGroundTruthHandler())

    with open(outputDir + "/" + runOutputFileName, 'w') as outFile:
        num_articles=0
        for file in os.listdir(inputDataset):
            if file.endswith(".xml"):
                print('Processing', file)
                with open(inputDataset + "/" + file) as inputRunFile:
                    xml.sax.parse(inputRunFile, HyperpartisanNewsDataExtractor(outFile, groundTruth))

    print("The predictions have been written to the output folder.")

if __name__ == '__main__':
    main(*parse_options())
