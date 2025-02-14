#!/usr/bin/env python

"""Extract information from the articles about time."""
# Version: 2018-09-21

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

random.seed(42)
runOutputFileName = "truth.txt"

# Assign Left = -10
# Assign Left Center = -5
# Assign Least = 0
# Assign Right Center = +5
# Assign Right = +10

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
    def __init__(self, outFile):
        xml.sax.ContentHandler.__init__(self)
        self.outFile = outFile

    def startElement(self, name, attrs):
        self.current_element = name
        if name == "article":

            articleId = attrs.getValue("id")

            if attrs.getValue("hyperpartisan") == 'true':
                hyperpartisan = 1
            else:
                hyperpartisan = 0
            bias = attrs.getValue('bias')
            orientation = None

            if bias == 'least':
                orientation = 0
            elif bias == 'left':
                orientation = -10
            elif bias == 'right':
                orientation =  10
            elif bias == 'left-center':
                orientation = -5
            elif bias == 'right-center':
                orientation =  5

            self.outFile.write(str(articleId) + ", " + str(hyperpartisan) + ", " + str(orientation) + "\n")

            # output format per line: "<article id> <prediction>[ <confidence>]"
            #   - prediction is either "true" (hyperpartisan) or "false" (not hyperpartisan)
            #   - confidence is an optional value to describe the confidence of the predictor in the prediction---the higher, the more confident

########## MAIN ##########


def main(inputDataset, outputDir):
    """Main method of this module."""

    with open(outputDir + "/" + runOutputFileName, 'w') as outFile:
        outFile.write('articleId' + "," + 'hyperpartisan' + "," + 'orientation' + "\n")
        for file in os.listdir(inputDataset):
            if file.endswith(".xml"):
                with open(inputDataset + "/" + file) as inputRunFile:
                    xml.sax.parse(inputRunFile, HyperpartisanNewsRandomPredictor(outFile))
                print(file, 'done.')

    print("The predictions have been written to the output folder.")


if __name__ == '__main__':
    main(*parse_options())

