import zipfile
import re
import xml.etree.ElementTree as ET

with zipfile.ZipFile('Presentation1.zip') as myZip:
    print(myZip.namelist())
    files = [myZip.open(f) for f in myZip.namelist() if re.search('slide[0-9]+.xml$', f)]
    trees = [ET.parse(f) for f in files]
    for f in myZip.namelist():
        if re.search('slide[0-9]+.xml$', f):
            pass
    for tree in trees:
        root = tree.getroot()
        #print(root.tag)
        for child in root:
            for c in child:
                for c1 in c:
                    for c2 in c1:
                        for c3 in c2:
                            for c4 in c3:
                                for c5 in c4:
                                    print(c5.tag, c5.text, c5.attrib)
                                    for c6 in c5:
                                        print(c6.tag, c6.text, c6.attrib)
                                        for c7 in c6:
                                            print(c7.tag, c7.text, c7.attrib)

