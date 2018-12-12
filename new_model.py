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

from library.sentiment import SentimentAnalysis
from library.use_model import predict_func

random.seed(42)
runOutputFileName = "prediction.txt"

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

class HyperpartisanPredictor(xml.sax.ContentHandler):
    def __init__(self, outFile):
        xml.sax.ContentHandler.__init__(self)
        self.outFile = outFile
        self.article_content = ''

    def startElement(self, name, attrs):
        self.current_element = name
        if name == "article":
            self.article_content = ''
            if "published-at" in attrs.keys():
                self.article_date = attrs.getValue("published-at")
            else:
                self.article_date = None
            self.article_title = attrs.getValue("title")
            self.articleId = attrs.getValue("id") # id of the article for which hyperpartisanship should be predicted

            # output format per line: "<article id> <prediction>[ <confidence>]"
            #   - prediction is either "true" (hyperpartisan) or "false" (not hyperpartisan)
            #   - confidence is an optional value to describe the confidence of the predictor in the prediction---the higher, the more confident

    def endElement(self, name):
        global num_articles, char_len
        if name == "article":
            char_len += len(self.article_content)
            num_articles += 1
            prediction = predict_func(self.article_title+' '+self.article_content)
            self.outFile.write(self.articleId + " " + prediction + "\n")

    def characters(self, content):
        self.article_content += content


########## MAIN ##########


def main(inputDataset, outputDir):
    """Main method of this module."""
    global num_articles, char_len

    with open(outputDir + "/" + runOutputFileName, 'w') as outFile:
        for file in os.listdir(inputDataset):
            if file.endswith(".xml"):
                print('Processing', file)
                with open(inputDataset + "/" + file) as inputRunFile:
                    xml.sax.parse(inputRunFile, HyperpartisanPredictor(outFile))
    print("The predictions have been written to the output folder.")


if __name__ == '__main__':
    main(*parse_options())

