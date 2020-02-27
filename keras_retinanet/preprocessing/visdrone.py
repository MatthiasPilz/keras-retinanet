"""
Copyright 2020-2021 Matthias Pilz (https://github.com/MatthiasPilz)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

""" This file is ran independently to bring the visdrone dataset into the format that can be 
    loaded via the csv_generator
"""

import os
import csv

vd_classes = {
    'ignored'           : 0,
    'pedestrian'        : 1,
    'people'            : 2,
    'bicycle'           : 3,
    'car'               : 4,
    'van'               : 5,
    'truck'             : 6,
    'tricycle'          : 7,
    'awning-tricycle'   : 8,
    'bus'               : 9,
    'motor'             : 10,
    'others'            : 11
}

vd_classesFocussed = {
    'car'               : 4,
    'van'               : 5,
    'truck'             : 6,
    'bus'               : 9,
    'others'            : 11
}

vd_classesFocussedInversed = {
    4   : 'car',
    5   : 'van',
    6   : 'truck',
    9   : 'bus',
    11  : 'others'
}


def _translate_boudingBoxes(
        xLeft: int,
        yLeft: int,
        width: int,
        height: int
):
    """ translate the annotation of the visdrone dataset into the one used in csv_generator

    Args
        xleft       : The x coordinate of the top-left corner of the bounding box
        yleft       : The y coordinate of the top-left corner of the bounding box
        width       : The width in pixels of the bounding box
        height      : The height in pixels of the bounding box

    Returns
        The coordinates of the top-left corner as well as the coordinates of the bottom-right corner of the
        bounding box
    """
    x1 = xLeft
    y1 = yLeft
    x2 = x1 + width
    y2 = y1 + height

    if y2 == y1:
        y2 += 1
    if x2 == x1:
        x2 += 1

    return x1, y1, x2, y2


def create_csvAnnotation(
        fileLocation,
        csvFileName = "visDroneAnnotation"
):
    """ go through annotations of the visdrone dataset and create respective csv file
        visdrone annotations have the following format (per line):
        <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
            0           1           2            3           4           5               6           7

    Args
        fileLocation    : The folder that contains the annotations

    Returns
        list of images that are usable, i.e. include suitable objects
    """

    with open(csvFileName+'.csv', mode='w') as csvFile:
        csvWriter = csv.writer(csvFile, delimiter=',')
        fileList = set()
        for fileNameComplete in os.listdir(fileLocation):
            fileName = os.path.splitext(fileNameComplete)[0]
            fileName = './images/' + fileName + '.jpg'
            with open(os.path.join(fileLocation, fileNameComplete), 'r') as f:
                line = f.readline()
                while line:
                    visAnnotation = line.split(',')
                    currentClass = int(visAnnotation[5])

                    # only consider those of the 'interesting' classes:
                    if currentClass in vd_classesFocussedInversed.keys():
                        fileList.add(fileName)
                        x1, y1, x2, y2 = _translate_boudingBoxes( int(visAnnotation[0]),
                                                                  int(visAnnotation[1]),
                                                                  int(visAnnotation[2]),
                                                                  int(visAnnotation[3]) )
                        csvWriter.writerow([fileName, str(x1), str(y1), str(x2), str(y2), vd_classesFocussedInversed[currentClass]])

                    line = f.readline()

    return fileList


########################################################################################################################
#   MAIN FUNCTION                                                                                                      #
########################################################################################################################
def main():
    temp = ['val', 'train', 'test-dev']
    for t in temp:
        fileLocation = '../../tests/test-data/VisDrone2019-DET-' + t + '/annotations'
        csvFileName = '../../tests/test-data/VisDrone2019-DET-' + t + '/VisDrone2019-DET-' + t + 'Annotation'

        files = create_csvAnnotation(fileLocation, csvFileName)

        print('There are {} unique images in the {}-set'.format(len(files), t))


if __name__ == '__main__':
    main()
