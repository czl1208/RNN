from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import re

def removeSymbols(sentence):
	sentence = re.sub(r'[^\w]', ' ', sentence)
	sentence = sentence.replace("_", "")
	sentence = ' '.join(sentence.split())
	return sentence

ppt = open("ppt.dat", "r")
processed = open("processed_ppt.dat", "w")
text = ppt.readlines()
print(text)
for line in text:
	processedline = removeSymbols(line)
	if len(processedline) != 0:
		processedline = processedline +'\n'
		processed.write(processedline)
ppt.close()
processed.close()
newfile = open("processed_ppt.dat", "r")
print(newfile.readlines())
#processed = open("processed_ppt.dat", "w")
