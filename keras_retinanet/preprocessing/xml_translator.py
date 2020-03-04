import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def _translate_class(s):
    if s == 'white':
        result = 'helmet'
    elif s == 'red':
        result = 'helmet'
    elif s == 'yellow':
        result = 'helmet'
    elif s == 'blue':
        result = 'helmet'
    elif s == 'none':
        result = 'no_helmet'
    else:
        print('unknown class -- skip example!')
        result = 'ERROR'

    return result


def translate_XMLtoCSV(path, label):
    f = open('../../tests/test-data/hardHat/ImageSets/Main/' + label + '_sorted.txt', 'r')
    l = list(f)
    l = [s[:-1] for s in l]
    l = sorted(l)
    xmlList = []
    for xmlFile in sorted(glob.glob(path + '/*.xml')):
        tree = ET.parse(xmlFile)
        root = tree.getroot()
        name = root.find('filename').text[:-4]
        if name in l:
            fileName = './JPEGImages/' + name + '.jpg'

            for member in root.findall('object'):
                classID = _translate_class(member[0].text)
                if classID == 'ERROR':
                    continue
                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)

                # combine results in 'value'
                value = (fileName, xmin, ymin, xmax, ymax, classID)
                xmlList.append(value)
    # columnName = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    xmlDF = pd.DataFrame(xmlList)
    return xmlDF

def main():
    labels = ['trainval', 'test']
    for label in labels:
        imagePath = '../../tests/test-data/hardHat/' + 'Annotations'
        xmlDF = translate_XMLtoCSV(imagePath, label)
        xmlDF.to_csv('../../tests/test-data/hardHat/' + label + '.csv', index=None, header=False)
        print('successfully converted {} xml to csv'.format(label))

main()