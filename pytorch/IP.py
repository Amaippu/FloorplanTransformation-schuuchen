#from gurobipy import *
from pulp import *
import cv2
import numpy as np
import sys
import csv
import copy
from utils import *
from skimage import measure
from skimage.draw import polygon

GAPS = {'wall_extraction': 5, 'door_extraction': 5, 'icon_extraction': 5, 'wall_neighbor': 5, 'door_neighbor': 5, 'icon_neighbor': 5, 'wall_conflict': 5, 'door_conflict': 5, 'icon_conflict': 5, 'wall_icon_neighbor': 5, 'wall_icon_conflict': 5, 'wall_door_neighbor': 5, 'door_point_conflict': 5}
DISTANCES = {'wall_icon': 5, 'point': 5, 'wall': 10, 'door': 5, 'icon': 5}
LENGTH_THRESHOLDS = {'wall': 5, 'door': 5, 'icon': 5}


junctionWeight = 100
augmentedJunctionWeight = 50
labelWeight = 1

wallWeight = 10
doorWeight = 10
iconWeight = 10

#wallTypeWeight = 10
#doorTypeWeight = 10
iconTypeWeight = 10

wallLineWidth = 3
doorLineWidth = 2
#doorExposureWeight = 0


NUM_WALL_TYPES = 1
NUM_DOOR_TYPES = 2
#NUM_LABELS = NUM_WALL_TYPES + NUM_DOOR_TYPES + NUM_ICONS + NUM_ROOMS + 1
NUM_LABELS = NUM_ICONS + NUM_ROOMS

WALL_LABEL_OFFSET = NUM_ROOMS + 1
DOOR_LABEL_OFFSET = NUM_ICONS + 1
ICON_LABEL_OFFSET = 0
ROOM_LABEL_OFFSET = NUM_ICONS


colorMap = ColorPalette(NUM_CORNERS).getColorMap()

width = 256
height = 256
maxDim = max(width, height)
sizes = np.array([width, height])

ORIENTATION_RANGES = getOrientationRanges(width, height)

iconNames = getIconNames()
iconNameNumberMap = dict(zip(iconNames, range(len(iconNames))))
iconNumberNameMap = dict(zip(range(len(iconNames)), iconNames))

def get_pt(i):
  if i < 4:
    return 0
  elif i < 28:
    return 1
  elif i < 52:
    return 2
  else:
    return 3

def get_ori(i):
  if i < 4:
    return i
  elif i < 28:
    return i-4
  elif i < 52:
    return i-28
  else:
    return i-52

## Extract corners from corner heatmp predictions
def extractCorners(heatmaps, threshold, gap, cornerType = 'wall', augment=False, gt=False):
  if gt:
    orientationPoints = heatmaps
  else:
    orientationPoints = extractCornersFromHeatmaps(heatmaps, threshold)
    pass

  if cornerType == 'wall':
    cornerOrientations = []
    for orientations in POINT_ORIENTATIONS:
      cornerOrientations += orientations
      continue
  elif cornerType == 'door':
    cornerOrientations = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)]#POINT_ORIENTATIONS[0]
  else:
    cornerOrientations = [(0, 3), (0, 1), (1, 2), (2, 3)]#POINT_ORIENTATIONS[1]
    pass
  #print(orientationPoints)
  if augment:
    orientationMap = {}
    for pointType, orientationOrientations in enumerate(POINT_ORIENTATIONS):
      for orientation, orientations in enumerate(orientationOrientations):
        orientationMap[orientations] = orientation
        continue
      continue
    #print(orientationMap)
    for orientationIndex, corners in enumerate(orientationPoints):
      if len(corners) > 3:
        continue #skip aug
      #pointType = orientationIndex // 4
      pointType = get_pt(orientationIndex)

      if pointType in [2]:
        #orientation = orientationIndex % 4
        orientation = get_ori(orientationIndex)
        orientations = POINT_ORIENTATIONS[pointType][orientation]
        for i in range(len(orientations)):
          newOrientations = list(orientations)
          newOrientations.remove(orientations[i])
          newOrientations = tuple(sorted(newOrientations))
          if not newOrientations in orientationMap:
            continue
          newOrientation = orientationMap[newOrientations]
          for corner in corners:
            #print('-'*20, orientations, orientation, corner)
            orientationPoints[4 + newOrientation].append(corner + (True, ))
            continue
          continue
      elif pointType in [1]:
        orientation = get_ori(orientationIndex)
        orientations = POINT_ORIENTATIONS[pointType][orientation]
        for orientation in range(8):
          if orientation in orientations:
            continue
          newOrientations = list(orientations)
          newOrientations.append(orientation)
          newOrientations = tuple(sorted(newOrientations))
          if not newOrientations in orientationMap:
            continue
          newOrientation = orientationMap[newOrientations]
          for corner in corners:
            #print('-'*20, orientations, orientation, corner)
            orientationPoints[28 + newOrientation].append(corner + (True, ))
            continue
          continue
        pass
      continue
    pass
  #print(orientationPoints)
  pointOffset = 0
  pointOffsets = []
  points = []
  pointOrientationLinesMap = []
  for orientationIndex, corners in enumerate(orientationPoints):
    pointOffsets.append(pointOffset)
    orientations = cornerOrientations[orientationIndex]
    #print('-'*20, orientations)
    for point in corners:
      orientationLines = {}
      for orientation in orientations:
        orientationLines[orientation] = []
        continue
      pointOrientationLinesMap.append(orientationLines)
      continue

    pointOffset += len(corners)
    aug = []
    if cornerType == 'wall':
      points += [[corner[0][0], corner[0][1], get_pt(orientationIndex), get_ori(orientationIndex)] for corner in corners]

      aug += [[corner[0][0], corner[0][1], get_pt(orientationIndex), get_ori(orientationIndex)] for corner in corners if corner[-1] == True]
      #print(orientationIndex, aug)
    elif cornerType == 'door':
      points += [[corner[0][0], corner[0][1], 0, orientationIndex] for corner in corners]
    else:
      points += [[corner[0][0], corner[0][1], 1, orientationIndex] for corner in corners]
      pass
    continue

  augmentedPointMask = {}

  lines = []
  pointNeighbors = [[] for point in points]
  for orientationIndex, corners in enumerate(orientationPoints):
    #print(orientationIndex, corners)
    orientations = cornerOrientations[orientationIndex]
    for orientation in orientations:
      #if orientation not in [1, 2]: continue
      if orientation < 4:
          oppositeOrientation = (orientation + 2) % 4
      elif orientation < 6:
          oppositeOrientation = orientation + 2
      else:
          oppositeOrientation = orientation - 2
      
      lineDim = -1
      if orientation in [0, 2]:
        lineDim = 0
      elif orientation in [1, 3]:
        lineDim = 1
        pass

      for cornerIndex, corner in enumerate(corners):
        pointIndex = pointOffsets[orientationIndex] + cornerIndex
        #print(corner)
        if len(corner) > 3:
          augmentedPointMask[pointIndex] = True
          pass
        if orientation < 4:
          ranges = copy.deepcopy(ORIENTATION_RANGES[orientation])
          ranges[lineDim] = min(ranges[lineDim], corner[0][lineDim])
          ranges[lineDim + 2] = max(ranges[lineDim + 2], corner[0][lineDim])
          ranges[1 - lineDim] = min(ranges[1 - lineDim], corner[1][1 - lineDim] - gap)
          ranges[1 - lineDim + 2] = max(ranges[1 - lineDim + 2], corner[2][1 - lineDim] + gap)

        for oppositeOrientationIndex, oppositeCorners in enumerate(orientationPoints):
          if oppositeOrientation not in cornerOrientations[oppositeOrientationIndex]:
            continue
          for oppositeCornerIndex, oppositeCorner in enumerate(oppositeCorners):
            if orientationIndex == oppositeOrientationIndex and oppositeCornerIndex == cornerIndex:
              continue

            oppositePointIndex = pointOffsets[oppositeOrientationIndex] + oppositeCornerIndex
            if orientation < 4:
              if oppositeCorner[0][lineDim] < ranges[lineDim] or oppositeCorner[0][lineDim] > ranges[lineDim + 2] or ranges[1 - lineDim] > oppositeCorner[2][1 - lineDim] or ranges[1 - lineDim + 2] < oppositeCorner[1][1 - lineDim]:
                continue
              if abs(oppositeCorner[0][lineDim] - corner[0][lineDim]) < LENGTH_THRESHOLDS[cornerType]: continue
            else:
              #if orientation == 4 and (True not in corner and True not in oppositeCorner): print(oppositeCorner[0][0], corner[1][0]-gap, oppositeCorner[0][1], corner[1][1]-gap)
              #if orientation == 6 and (True not in corner and True not in oppositeCorner): print(oppositeCorner[0][0], corner[2][0]+gap, oppositeCorner[0][1], corner[2][1]+gap)
              #if orientation == 5 and (True not in corner and True not in oppositeCorner): print(oppositeCorner[0][0], corner[1][0]-gap, oppositeCorner[0][1], corner[2][1]+gap); print(corner, oppositeCorner)
              '''
              if orientation == 4 and not (oppositeCorner[0][0] < corner[1][0]-gap and oppositeCorner[0][1] < corner[1][1]-gap): continue
              if orientation == 5 and not (oppositeCorner[0][0] < corner[1][0]-gap and oppositeCorner[0][1] > corner[2][1]+gap): continue
              if orientation == 6 and not (oppositeCorner[0][0] > corner[2][0]+gap and oppositeCorner[0][1] > corner[2][1]+gap): continue
              if orientation == 7 and not (oppositeCorner[0][0] > corner[2][0]+gap and oppositeCorner[0][1] < corner[1][1]-gap): continue
              '''
              if orientation == 4 and not (oppositeCorner[0][0] < corner[1][0] and oppositeCorner[0][1] < corner[1][1]): continue
              if orientation == 5 and not (oppositeCorner[0][0] < corner[1][0] and oppositeCorner[0][1] > corner[2][1]): continue
              if orientation == 6 and not (oppositeCorner[0][0] > corner[2][0] and oppositeCorner[0][1] > corner[2][1]): continue
              if orientation == 7 and not (oppositeCorner[0][0] > corner[2][0] and oppositeCorner[0][1] < corner[1][1]): continue

              if abs(oppositeCorner[0][0] - corner[0][0]) < LENGTH_THRESHOLDS[cornerType] or abs(oppositeCorner[0][1] - corner[0][1]) < LENGTH_THRESHOLDS[cornerType]: continue

              if True not in corner and True not in oppositeCorner: print('detected slant lines: ', cornerType, corner, oppositeCorner, orientation, oppositeOrientation)

            lineIndex = len(lines)
            pointOrientationLinesMap[pointIndex][orientation].append(lineIndex)
            pointOrientationLinesMap[oppositePointIndex][oppositeOrientation].append(lineIndex)
            pointNeighbors[pointIndex].append(oppositePointIndex)
            pointNeighbors[oppositePointIndex].append(pointIndex)

            lines.append((pointIndex, oppositePointIndex))
            continue
          continue
        continue
      continue
    continue
  return points, lines, pointOrientationLinesMap, pointNeighbors, augmentedPointMask


## Corner type augmentation to enrich the candidate set (e.g., a T-shape corner can be treated as a L-shape corner)
def augmentPoints(points, decreasingTypes = [2], increasingTypes = [1]):
  print('augmentPoints'); exit(1)
  orientationMap = {}
  for pointType, orientationOrientations in enumerate(POINT_ORIENTATIONS):
    for orientation, orientations in enumerate(orientationOrientations):
      orientationMap[orientations] = orientation
      continue
    continue

  newPoints = []
  for pointIndex, point in enumerate(points):
    if point[2] not in decreasingTypes:
      continue
    orientations = POINT_ORIENTATIONS[point[2]][point[3]]
    for i in range(len(orientations)):
      newOrientations = list(orientations)
      newOrientations.remove(orientations[i])
      newOrientations = tuple(newOrientations)
      if not newOrientations in orientationMap:
        continue
      newOrientation = orientationMap[newOrientations]
      newPoints.append([point[0], point[1], point[2] - 1, newOrientation])
      continue
    continue

  for pointIndex, point in enumerate(points):
    if point[2] not in increasingTypes:
      continue
    orientations = POINT_ORIENTATIONS[point[2]][point[3]]
    for orientation in range(4):
      if orientation in orientations:
        continue

      if orientation < 4:
          oppositeOrientation = (orientation + 2) % 4
      elif orientation < 6:
          oppositeOrientation = orientation + 2
      else:
          oppositeOrientation = orientation - 2

      ranges = copy.deepcopy(ORIENTATION_RANGES[orientation])
      lineDim = -1
      if orientation == 0 or orientation == 2:
        lineDim = 1
      else:
        lineDim = 0
        pass
      deltas = [0, 0]

      if lineDim == 1:
        deltas[0] = gap
      else:
        deltas[1] = gap
        pass

      for c in range(2):
        ranges[c] = min(ranges[c], point[c] - deltas[c])
        ranges[c + 2] = max(ranges[c + 2], point[c] + deltas[c])
        continue

      hasNeighbor = False
      for neighborPointIndex, neighborPoint in enumerate(points):
        if neighborPointIndex == pointIndex:
          continue

        neighborOrientations = POINT_ORIENTATIONS[neighborPoint[2]][neighborPoint[3]]
        if oppositeOrientation not in neighborOrientations:
          continue

        inRange = True
        for c in range(2):
          if neighborPoint[c] < ranges[c] or neighborPoint[c] > ranges[c + 2]:
            inRange = False
            break
          continue

        if not inRange or abs(neighborPoint[lineDim] - point[lineDim]) < max(abs(neighborPoint[1 - lineDim] - point[1 - lineDim]), 1):
          continue

        hasNeighbor = True
        break

      if not hasNeighbor:
        continue

      newOrientations = list(orientations)
      newOrientations.append(orientation)
      newOrientations = tuple(newOrientations)
      if not newOrientations in orientationMap:
        continue
      newOrientation = orientationMap[newOrientations]
      newPoints.append([point[0], point[1], point[2] + 1, newOrientation])
      continue
    continue
  return points + newPoints


## Remove invalid walls as preprocessing
def filterWalls(wallPoints, wallLines):
  orientationMap = {}
  for pointType, orientationOrientations in enumerate(POINT_ORIENTATIONS):
    for orientation, orientations in enumerate(orientationOrientations):
      orientationMap[orientations] = orientation
      continue
    continue

  #print(POINT_ORIENTATIONS)
  while True:
    pointOrientationNeighborsMap = {}
    for line in wallLines:
      #print('-'*10, (wallPoints[line[0]], wallPoints[line[1]]))
      lineDim = calcLineDim(wallPoints, line)
      #if lineDim > 1: print('-*'*10, 368); exit(1)
      for c, pointIndex in enumerate(line):
        if lineDim == 0:
          if c == 0:
            orientation = 2#1
          else:
            orientation = 0#3
        elif lineDim == 1:
          if c == 0:
            orientation = 1#2
          else:
            orientation = 3#0
        elif lineDim == 2:
          if c == 0:
            orientation = 6#1
          else:
            orientation = 4#3
        else:
          if c == 0:
            orientation = 7#2
          else:
            orientation = 5#0

        if pointIndex not in pointOrientationNeighborsMap:
          pointOrientationNeighborsMap[pointIndex] = {}
          pass
        if orientation not in pointOrientationNeighborsMap[pointIndex]:
          pointOrientationNeighborsMap[pointIndex][orientation] = []
          pass
        pointOrientationNeighborsMap[pointIndex][orientation].append(line[1 - c])
        #print(pointOrientationNeighborsMap); exit(1)
        continue
      continue

    invalidPointMask = {}
    for pointIndex, point in enumerate(wallPoints):
      if pointIndex not in pointOrientationNeighborsMap:
        invalidPointMask[pointIndex] = True
        continue
      orientationNeighborMap = pointOrientationNeighborsMap[pointIndex]
      try:
        orientations = POINT_ORIENTATIONS[point[2]][point[3]]
      except:
        #print('-'*20, point[2], point[3])
        raise
      if len(orientationNeighborMap) < len(orientations):
        if len(orientationNeighborMap) >= 2 and tuple(orientationNeighborMap.keys()) in orientationMap:
          newOrientation = orientationMap[tuple(orientationNeighborMap.keys())]
          wallPoints[pointIndex][2] = len(orientationNeighborMap) - 1
          wallPoints[pointIndex][3] = newOrientation
          #print(orientationNeighborMap)
          #print('new', len(orientationNeighborMap), newOrientation)
          continue
        invalidPointMask[pointIndex] = True
        pass
      continue

    if len(invalidPointMask) == 0:
      break

    newWallPoints = []
    pointIndexMap = {}
    for pointIndex, point in enumerate(wallPoints):
      if pointIndex not in invalidPointMask:
        pointIndexMap[pointIndex] = len(newWallPoints)
        newWallPoints.append(point)
        pass
      continue

    wallPoints = newWallPoints

    newWallLines = []
    for lineIndex, line in enumerate(wallLines):
      if line[0] in pointIndexMap and line[1] in pointIndexMap:
        newLine = (pointIndexMap[line[0]], pointIndexMap[line[1]])
        newWallLines.append(newLine)
        pass
      continue
    wallLines = newWallLines
    continue

  pointOrientationLinesMap = [{} for _ in range(len(wallPoints))]
  pointNeighbors = [[] for _ in range(len(wallPoints))]

  for lineIndex, line in enumerate(wallLines):
    lineDim = calcLineDim(wallPoints, line)
    #if lineDim > 1: print('-*'*10, 445); exit(1)
    for c, pointIndex in enumerate(line):
      if lineDim == 0:
        if wallPoints[pointIndex][lineDim] < wallPoints[line[1 - c]][lineDim]:
          orientation = 2
        else:
          orientation = 0
          pass
      elif lineDim == 1:
        if wallPoints[pointIndex][lineDim] < wallPoints[line[1 - c]][lineDim]:
          orientation = 1
        else:
          orientation = 3
      elif lineDim == 2:
        if wallPoints[pointIndex][0] < wallPoints[line[1 - c]][0] and wallPoints[pointIndex][1] < wallPoints[line[1 - c]][1]:
          orientation = 6
        else:
          orientation = 4
          pass
      else:
        if wallPoints[pointIndex][0] < wallPoints[line[1 - c]][0] and wallPoints[pointIndex][1] > wallPoints[line[1 - c]][1]:
          orientation = 7
        else:
          orientation = 5

      if orientation not in pointOrientationLinesMap[pointIndex]:
        pointOrientationLinesMap[pointIndex][orientation] = []
        pass
      pointOrientationLinesMap[pointIndex][orientation].append(lineIndex)
      pointNeighbors[pointIndex].append(line[1 - c])
      continue
    continue
  return wallPoints, wallLines, pointOrientationLinesMap, pointNeighbors

## Write wall points to result file
def writePoints(points, pointLabels, output_prefix='test/'):
  with open(output_prefix + 'points_out.txt', 'w') as points_file:
    for point in points:
      points_file.write(str(point[0] + 1) + '\t' + str(point[1] + 1) + '\t')
      points_file.write(str(point[0] + 1) + '\t' + str(point[1] + 1) + '\t')
      points_file.write('point\t')
      points_file.write(str(point[2] + 1) + '\t' + str(point[3] + 1) + '\n')

  points_file.close()

  with open(output_prefix + 'point_labels.txt', 'w') as point_label_file:
    for point in pointLabels:
      point_label_file.write(str(point[0]) + '\t' + str(point[1]) + '\t' + str(point[2]) + '\t' + str(point[3]) + '\t' + str(point[4]) + '\t' + str(point[5]) + '\t' + str(point[6]) + '\t' + str(point[7]) + '\n')
  point_label_file.close()

## Write doors to result file
def writeDoors(points, lines, doorTypes, output_prefix='test/'):
  with open(output_prefix + 'doors_out.txt', 'w') as doors_file:
    for lineIndex, line in enumerate(lines):
      point_1 = points[line[0]]
      point_2 = points[line[1]]

      doors_file.write(str(point_1[0] + 1) + '\t' + str(point_1[1] + 1) + '\t')
      doors_file.write(str(point_2[0] + 1) + '\t' + str(point_2[1] + 1) + '\t')
      doors_file.write('door\t')
      doors_file.write(str(doorTypes[lineIndex] + 1) + '\t1\n')
    doors_file.close()

## Write icons to result file    
def writeIcons(points, icons, iconTypes, output_prefix='test/'):
  with open(output_prefix + 'icons_out.txt', 'w') as icons_file:
    for iconIndex, icon in enumerate(icons):
      point_1 = points[icon[0]]
      point_2 = points[icon[1]]
      point_3 = points[icon[2]]
      point_4 = points[icon[3]]

      x_1 = int(round((point_1[0] + point_3[0]) // 2)) + 1
      x_2 = int(round((point_2[0] + point_4[0]) // 2)) + 1
      y_1 = int(round((point_1[1] + point_2[1]) // 2)) + 1
      y_2 = int(round((point_3[1] + point_4[1]) // 2)) + 1

      icons_file.write(str(x_1) + '\t' + str(y_1) + '\t')
      icons_file.write(str(x_2) + '\t' + str(y_2) + '\t')
      icons_file.write(iconNumberNameMap[iconTypes[iconIndex]] + '\t')
      #icons_file.write(str(iconNumberStyleMap[iconTypes[iconIndex]]) + '\t')
      icons_file.write('1\t')
      icons_file.write('1\n')
    icons_file.close()


## Adjust wall corner locations to align with each other after optimization
def adjustPoints(points, lines):
  lineNeighbors = []
  for lineIndex, line in enumerate(lines):
    lineDim = calcLineDim(points, line)
    #if lineDim > 1: print('-*'*10, 531); exit(1)
    neighbors = []
    for neighborLineIndex, neighborLine in enumerate(lines):
      if neighborLineIndex <= lineIndex:
        continue
      #neighborLineDim = calcLineDim(points, neighborLine)
      point_1 = points[neighborLine[0]]
      point_2 = points[neighborLine[1]]
      lineDimNeighbor = calcLineDim(points, neighborLine)

      if lineDimNeighbor != lineDim:
        continue
      if neighborLine[0] != line[0] and neighborLine[0] != line[1] and neighborLine[1] != line[0] and neighborLine[1] != line[1]:
        continue
      neighbors.append(neighborLineIndex)
      continue
    lineNeighbors.append(neighbors)
    continue

  visitedLines = {}
  for lineIndex in range(len(lines)):
    if lineIndex in visitedLines:
      continue
    lineGroup = [lineIndex]
    while True:
      newLineGroup = lineGroup
      hasChange = False
      for line in lineGroup:
        neighbors = lineNeighbors[line]
        for neighbor in neighbors:
          if neighbor not in newLineGroup:
            newLineGroup.append(neighbor)
            hasChange = True
            pass
          continue
        continue
      if not hasChange:
        break
      lineGroup = newLineGroup
      continue

    for line in lineGroup:
      visitedLines[line] = True
      continue

    #print([[points[pointIndex] for pointIndex in lines[lineIndex]] for lineIndex in lineGroup], calcLineDim(points, lines[lineGroup[0]]))

    pointGroup = []
    for line in lineGroup:
      for index in range(2):
        pointIndex = lines[line][index]
        if pointIndex not in pointGroup:
          pointGroup.append(pointIndex)
          pass
        continue
      continue

    xy = np.concatenate([np.array([points[pointIndex][:2] for pointIndex in lines[lineIndex]]) for lineIndex in lineGroup], axis=0)
    mins = list(xy.min(0))
    maxs = list(xy.max(0))

    # for slant lines
    lineDim = calcLineDim(points, lines[lineGroup[0]])
    if lineDim == 2:
      #print([lines[l] for l in lineGroup])
      print('adjust slant line: ', lineDim, xy, mins, maxs)
      for p in pointGroup:
        #print(points[p][:2], points[p]); exit(1)
        if points[p][:2] == mins or points[p][:2] == maxs: continue
        points[p][:2] = project_p_toline([points[p][0], points[p][1]], [mins, maxs])
    elif lineDim == 3:
      print('adjust slant line: ', lineDim, xy, mins, maxs)
      #print([lines[l] for l in lineGroup]); print(lineDim, xy, mins, maxs)
      for p in pointGroup:
        #print(points[p][:2], points[p]); exit(1)
        if points[p][0] == mins[0] or points[p][1] == mins[1] or points[p][0] == maxs[0] or points[p][1] == maxs[1]: continue
        points[p][:2] = project_p_toline([points[p][0], points[p][1]], [[mins[0], maxs[1]], [maxs[0], mins[1]]])
    else:
      # for Manhattan lines
      '''
      if maxs[0] - mins[0] > maxs[1] - mins[1]:
        lineDim = 0
      else:
        lineDim = 1
        pass
      '''
      fixedValue = 0
      for point in pointGroup:
        fixedValue += points[point][1 - lineDim]
        continue
      fixedValue /= len(pointGroup)

      for point in pointGroup:
        points[point][1 - lineDim] = fixedValue
        continue
      continue
  return

## Merge two close points after optimization
def mergePoints(points, lines):
  validPointMask = {}
  for line in lines:
    validPointMask[line[0]] = True
    validPointMask[line[1]] = True
    continue

  orientationMap = {}
  for pointType, orientationOrientations in enumerate(POINT_ORIENTATIONS):
    for orientation, orientations in enumerate(orientationOrientations):
      orientationMap[orientations] = (pointType, orientation)
      continue
    continue

  for pointIndex_1, point_1 in enumerate(points):
    if pointIndex_1 not in validPointMask:
      continue
    for pointIndex_2, point_2 in enumerate(points):
      if pointIndex_2 <= pointIndex_1:
        continue
      if pointIndex_2 not in validPointMask:
        continue
      if pointDistance(point_1[:2], point_2[:2]) <= DISTANCES['point']:
        orientations = list(POINT_ORIENTATIONS[point_1[2]][point_1[3]] + POINT_ORIENTATIONS[point_2[2]][point_2[3]])
        if len([line for line in lines if pointIndex_1 in line and pointIndex_2 in line]) > 0:
          '''
          if abs(point_1[0] - point_2[0]) > abs(point_1[1] - point_2[1]):
            orientations.remove(1)
            orientations.remove(3)
          else:
            orientations.remove(0)
            orientations.remove(2)
          '''
          lineDim = calcLineDim_(point_1, point_2)
          if lineDim == 0:
            orientations = [ori for ori in orientations if ori == 0 or ori == 2]
          elif lineDim == 1:
            orientations = [ori for ori in orientations if ori == 1 or ori == 3]
          elif lineDim == 2:
            orientations = [ori for ori in orientations if ori == 4 or ori == 6]
          else:
            orientations = [ori for ori in orientations if ori == 5 or ori == 7]
            pass
          pass
        orientations = tuple(set(orientations))
        if orientations not in orientationMap:
          for lineIndex, line in enumerate(lines):
            if pointIndex_1 in line and pointIndex_2 in line:
              lines[lineIndex] = (-1, -1)
              pass
            continue
          lineIndices_1 = [(lineIndex, tuple(set(line) - set((pointIndex_1, )))[0]) for lineIndex, line in enumerate(lines) if pointIndex_1 in line and pointIndex_2 not in line]
          lineIndices_2 = [(lineIndex, tuple(set(line) - set((pointIndex_2, )))[0]) for lineIndex, line in enumerate(lines) if pointIndex_2 in line and pointIndex_1 not in line]
          if len(lineIndices_1) == 1 and len(lineIndices_2) == 1:
            lineIndex_1, index_1 = lineIndices_1[0]
            lineIndex_2, index_2 = lineIndices_2[0]
            lines[lineIndex_1] = (index_1, index_2)
            lines[lineIndex_2] = (-1, -1)
            pass
          continue

        pointInfo = orientationMap[orientations]
        newPoint = [(point_1[0] + point_2[0]) // 2, (point_1[1] + point_2[1]) // 2, pointInfo[0], pointInfo[1]]
        points[pointIndex_1] = newPoint
        for lineIndex, line in enumerate(lines):
          if pointIndex_2 == line[0]:
            lines[lineIndex] = (pointIndex_1, line[1])
            pass
          if pointIndex_2 == line[1]:
            lines[lineIndex] = (line[0], pointIndex_1)
            pass
          continue
        pass
      continue
    continue
  return

## Adjust door corner locations to align with each other after optimization
def adjustDoorPoints(doorPoints, doorLines, wallPoints, wallLines, doorWallMap):
  for doorLineIndex, doorLine in enumerate(doorLines):
    lineDim = calcLineDim(doorPoints, doorLine, True)
    wallLine = wallLines[doorWallMap[doorLineIndex]]
    wallPoint_1 = wallPoints[wallLine[0]]
    wallPoint_2 = wallPoints[wallLine[1]]

    if lineDim > 1:
      dp0, dp1 = doorPoints[doorLine[0]][:2], doorPoints[doorLine[1]][:2]
      wl = [wallPoint_1[:2], wallPoint_2[:2]]
      doorPoints[doorLine[0]][:2] = project_p_toline(dp0, wl)
      doorPoints[doorLine[1]][:2] = project_p_toline(dp1, wl)
      
    else:
      fixedValue = (wallPoint_1[1 - lineDim] + wallPoint_2[1 - lineDim]) // 2
      for endPointIndex in range(2):
        doorPoints[doorLine[endPointIndex]][1 - lineDim] = fixedValue
        continue
    continue

## Generate icon candidates
def findIconsFromLines(iconPoints, iconLines):
  icons = []
  pointOrientationNeighborsMap = {}
  for line in iconLines:
    lineDim = calcLineDim(iconPoints, line)
    if lineDim > 1: print('-*'*10, 697); exit(1)
    for c, pointIndex in enumerate(line):
      if lineDim == 0:
        if c == 0:
          orientation = 2
        else:
          orientation = 0
      else:
        if c == 0:
          orientation = 1
        else:
          orientation = 3
          pass
        pass

      if pointIndex not in pointOrientationNeighborsMap:
        pointOrientationNeighborsMap[pointIndex] = {}
        pass
      if orientation not in pointOrientationNeighborsMap[pointIndex]:
        pointOrientationNeighborsMap[pointIndex][orientation] = []
        pass
      pointOrientationNeighborsMap[pointIndex][orientation].append(line[1 - c])
      continue
    continue

  for pointIndex, orientationNeighborMap in pointOrientationNeighborsMap.items():
    if 1 not in orientationNeighborMap or 2 not in orientationNeighborMap:
      continue
    for neighborIndex_1 in orientationNeighborMap[1]:
      if 2 not in pointOrientationNeighborsMap[neighborIndex_1]:
        continue
      lastCornerCandiates = pointOrientationNeighborsMap[neighborIndex_1][2]
      for neighborIndex_2 in orientationNeighborMap[2]:
        if 1 not in pointOrientationNeighborsMap[neighborIndex_2]:
          continue
        for lastCornerIndex in pointOrientationNeighborsMap[neighborIndex_2][1]:
          if lastCornerIndex not in lastCornerCandiates:
            continue

          point_1 = iconPoints[pointIndex]
          point_2 = iconPoints[neighborIndex_1]
          point_3 = iconPoints[neighborIndex_2]
          point_4 = iconPoints[lastCornerIndex]

          x_1 = int((point_1[0] + point_3[0]) // 2)
          x_2 = int((point_2[0] + point_4[0]) // 2)
          y_1 = int((point_1[1] + point_2[1]) // 2)
          y_2 = int((point_3[1] + point_4[1]) // 2)

          #if x_2 <= x_1 or y_2 <= y_1:
          #continue
          if (x_2 - x_1 + 1) * (y_2 - y_1 + 1) <= LENGTH_THRESHOLDS['icon'] * LENGTH_THRESHOLDS['icon']:
            continue

          icons.append((pointIndex, neighborIndex_1, neighborIndex_2, lastCornerIndex))
          continue
        continue
      continue
    continue
  return icons


## Find two wall lines facing each other and accumuate semantic information in between
def findLineNeighbors(points, lines, labelVotesMap, labelVotesMap1, gap):
  lineNeighbors = [[{}, {}] for lineIndex in range(len(lines))]
  for lineIndex, line in enumerate(lines):
    #assert all(0 < points[line[i]][j] < 256 for i in range(2) for j in range(2)), print('-!--' * 20)
    lineDim = calcLineDim(points, line)
    for neighborLineIndex, neighborLine in enumerate(lines):
      if neighborLineIndex <= lineIndex:
        continue
      neighborLineDim = calcLineDim(points, neighborLine)
      if lineDim != neighborLineDim:
        continue

      if lineDim > 1:
        lp1 = [points[line[0]][:2], points[line[1]][:2]]
        lp2 = [points[neighborLine[0]][:2], points[neighborLine[1]][:2]]
        if lp1[0] == lp2[0] or lp1[0] == lp2[1] or lp1[1] == lp2[0] or lp1[1] == lp2[1]: continue

        p1_to_l2 = [project_p_toline(p, lp2) for p in lp1]
        p2_to_l1 = [project_p_toline(p, lp1) for p in lp2]

        if p2p(lp1[0], p1_to_l2[0]) < gap: continue
        if all(pInLine(p, lp2) for p in p1_to_l2):
          if p2p(lp1[0], lp1[1]) < gap: continue
          if lineDim == 2:
            if lp1[0][0] < p1_to_l2[0][0]:
              region = [lp1[0], lp1[1], p1_to_l2[1], p1_to_l2[0], 0]
              lineNeighbors[lineIndex][0][neighborLineIndex] = region
              lineNeighbors[neighborLineIndex][1][lineIndex] = region
              #print(region, lp1, lp2, lineDim, '------824-----')
            else:
              region = [p1_to_l2[0], p1_to_l2[1], lp1[1], lp1[0], 1]
              lineNeighbors[lineIndex][1][neighborLineIndex] = region
              lineNeighbors[neighborLineIndex][0][lineIndex] = region
              #print(region, lp1, lp2, lineDim, '------827-----')
          else:
            if lp1[0][0] < p1_to_l2[0][0]:
              region = [lp1[1], lp1[0], p1_to_l2[0], p1_to_l2[1], 2]
              lineNeighbors[lineIndex][1][neighborLineIndex] = region
              lineNeighbors[neighborLineIndex][0][lineIndex] = region
              #print(region, lp1, lp2, lineDim, '------833-----')
            else:
              region = [p1_to_l2[1], p1_to_l2[0], lp1[0], lp1[1], 3]
              lineNeighbors[lineIndex][0][neighborLineIndex] = region
              lineNeighbors[neighborLineIndex][1][lineIndex] = region
              #print(region, lp1, lp2, lineDim, '------838-----')

        elif pInLine(p1_to_l2[0], lp2):
          if p2p(p1_to_l2[0], lp2[1]) < gap: continue
          if lineDim == 2:
            if lp1[0][0] < p1_to_l2[0][0]:
              region = [lp1[0], p2_to_l1[1], lp2[1], p1_to_l2[0], 4]
              lineNeighbors[lineIndex][0][neighborLineIndex] = region
              lineNeighbors[neighborLineIndex][1][lineIndex] = region
              #print(region, lp1, lp2, lineDim, '------847-----')
            else:
              region = [p1_to_l2[0], lp2[1], p2_to_l1[1], lp1[0], 5]
              lineNeighbors[lineIndex][1][neighborLineIndex] = region
              lineNeighbors[neighborLineIndex][0][lineIndex] = region
              #print(region, lp1, lp2, lineDim, '------852-----')
          else:
            if lp1[0][0] < p1_to_l2[0][0]:
              region = [p2_to_l1[1], lp1[0], p1_to_l2[0], lp2[1], 6]
              lineNeighbors[lineIndex][1][neighborLineIndex] = region
              lineNeighbors[neighborLineIndex][0][lineIndex] = region
              #print(region, lp1, lp2, lineDim, '------858-----')
            else:
              region = [lp2[1], p1_to_l2[0], lp1[0], p2_to_l1[1], 7]
              lineNeighbors[lineIndex][0][neighborLineIndex] = region
              lineNeighbors[neighborLineIndex][1][lineIndex] = region
              #print(region, lp1, lp2, lineDim, '------863-----')

        elif pInLine(p1_to_l2[1], lp2):
           if p2p(p1_to_l2[1], lp2[0]) < gap: continue
           if lineDim == 2:
             if lp1[1][0] < p1_to_l2[1][0]:
               region = [p2_to_l1[0], lp1[1], p1_to_l2[1], lp2[0], 8]
               lineNeighbors[lineIndex][0][neighborLineIndex] = region
               lineNeighbors[neighborLineIndex][1][lineIndex] = region
               #print(region, lp1, lp2, lineDim, '------872-----')
             else:
               region = [lp2[0], p1_to_l2[1], lp1[1], p2_to_l1[0], 9]
               lineNeighbors[lineIndex][1][neighborLineIndex] = region
               lineNeighbors[neighborLineIndex][0][lineIndex] = region
               #print(region, lp1, lp2, lineDim, '------877-----')
           else:
             if lp1[1][0] < p1_to_l2[1][0]:
               region = [lp1[1], p2_to_l1[0], lp2[0], p1_to_l2[1], 10]
               lineNeighbors[lineIndex][1][neighborLineIndex] = region
               lineNeighbors[neighborLineIndex][0][lineIndex] = region
               #print(region, lp1, lp2, lineDim, '------883-----')
             else:
               region = [p1_to_l2[1], lp2[0], p2_to_l1[0], lp1[1], 11]
               lineNeighbors[lineIndex][0][neighborLineIndex] = region
               lineNeighbors[neighborLineIndex][1][lineIndex] = region
               #print(region, lp1, lp2, lineDim, '------888-----')

        continue

      minValue = max(points[line[0]][lineDim], points[neighborLine[0]][lineDim])
      maxValue = min(points[line[1]][lineDim], points[neighborLine[1]][lineDim])
      if maxValue - minValue < gap:
        continue
      fixedValue_1 = points[line[0]][1 - lineDim]
      fixedValue_2 = points[neighborLine[0]][1 - lineDim]

      minValue = int(minValue)
      maxValue = int(maxValue)
      fixedValue_1 = int(fixedValue_1)
      fixedValue_2 = int(fixedValue_2)

      if abs(fixedValue_2 - fixedValue_1) < gap:
        continue
      if lineDim == 0:
        if fixedValue_1 < fixedValue_2:
          region = ((minValue, fixedValue_1), (maxValue, fixedValue_2))
          lineNeighbors[lineIndex][1][neighborLineIndex] = region
          lineNeighbors[neighborLineIndex][0][lineIndex] = region
        else:
          region = ((minValue, fixedValue_2), (maxValue, fixedValue_1))
          lineNeighbors[lineIndex][0][neighborLineIndex] = region
          lineNeighbors[neighborLineIndex][1][lineIndex] = region
      else:
        if fixedValue_1 < fixedValue_2:
          region = ((fixedValue_1, minValue), (fixedValue_2, maxValue))
          lineNeighbors[lineIndex][0][neighborLineIndex] = region
          lineNeighbors[neighborLineIndex][1][lineIndex] = region
        else:
          region = ((fixedValue_2, minValue), (fixedValue_1, maxValue))
          lineNeighbors[lineIndex][1][neighborLineIndex] = region
          lineNeighbors[neighborLineIndex][0][lineIndex] = region
          pass
        pass
      continue
    continue

  # remove neighbor pairs which are separated by another line
  while True:
    hasChange = False
    for lineIndex, neighbors in enumerate(lineNeighbors):
      lineDim = calcLineDim(points, lines[lineIndex])
      #if lineDim > 1: continue#print('-*'*10, 915); exit(1)
      if lineDim == 3:
        for neighbor_1, region_1 in neighbors[1].items():
          for neighbor_2, _ in neighbors[0].items():
            if neighbor_2 not in lineNeighbors[neighbor_1][0]:
              continue
            region_2 = lineNeighbors[neighbor_1][0][neighbor_2]
            r2_p0, r2_p1 = region_2[0], region_2[1]
            r1_p0, r1_p1 = region_1[0], region_1[1]
            moved_r2_p1, moved_r2_p0 = move_p_inline(r2_p1, [r2_p1, r2_p0]), move_p_inline(r2_p0, [r2_p0, r2_p1])
            p_r2_p1, p_r2_p0 = project_p_toline(moved_r2_p1, [r1_p0, r1_p1]), project_p_toline(moved_r2_p0, [r1_p0, r1_p1])
            if p_r2_p1[0] > r1_p1[0] and p_r2_p0[0] < r1_p0[0]:
              #print('-*'*10, 918); exit(1)
              lineNeighbors[neighbor_1][0].pop(neighbor_2)
              lineNeighbors[neighbor_2][1].pop(neighbor_1)
              hasChange = True
              pass
            continue
          continue
        continue
       
      for neighbor_1, region_1 in neighbors[1].items():
        for neighbor_2, _ in neighbors[0].items():
          if neighbor_2 not in lineNeighbors[neighbor_1][0]:
            continue
          region_2 = lineNeighbors[neighbor_1][0][neighbor_2]

          if lineDim == 2:
            r2_p3, r2_p2 = region_2[3], region_2[2]
            r1_p3, r1_p2 = region_1[3], region_1[2]
            moved_r2_p3, moved_r2_p2 = move_p_inline(r2_p3, [r2_p3, r2_p2]), move_p_inline(r2_p2, [r2_p2, r2_p3])
            p_r2_p3, p_r2_p2 = project_p_toline(moved_r2_p3, [r1_p3, r1_p2]), project_p_toline(moved_r2_p2, [r1_p3, r1_p2])

          if (lineDim < 2 and region_1[0][lineDim] < region_2[0][lineDim] + gap and region_1[1][lineDim] > region_2[1][lineDim] - gap) or (lineDim == 2 and p_r2_p3[0] > r1_p3[0] and p_r2_p2[0] < r1_p2[0]):
            lineNeighbors[neighbor_1][0].pop(neighbor_2)
            lineNeighbors[neighbor_2][1].pop(neighbor_1)
            hasChange = True
            pass
          continue
        continue
      continue
    if not hasChange:
      break


  for lineIndex, directionNeighbors in enumerate(lineNeighbors):
    lineDim = calcLineDim(points, lines[lineIndex])
    for direction, neighbors in enumerate(directionNeighbors):
      for neighbor, region in neighbors.items():
        if lineDim > 1:
          region = np.array(region[:4])
          rs, cs = polygon(region[:, 1]+1, region[:, 0]+1)
          rs[rs > 255] = 255
          cs[cs > 255] = 255
          labelVotes = np.array([labelVotesMap[i][rs, cs].sum() for i in range(labelVotesMap.shape[0])])
        else:
          #labelVotes = labelVotesMap1[:, region[1][1], region[1][0]] + labelVotesMap1[:, region[0][1], region[0][0]] - labelVotesMap1[:, region[0][1], region[1][0]] - labelVotesMap1[:, region[1][1], region[0][0]]
          region = labelVotesMap[:, region[0][1]+1:region[1][1]+1, region[0][0]+1:region[1][0]+1]
          labelVotes = np.sum(np.sum(region, axis=1), axis=1)
          
        neighbors[neighbor] = labelVotes
        continue
      continue
    continue
  return lineNeighbors


## Find neighboring wall line/icon pairs
def findRectangleLineNeighbors(rectanglePoints, rectangles, linePoints, lines, lineNeighbors, gap, distanceThreshold):
  rectangleLineNeighbors = [{} for rectangleIndex in range(len(rectangles))]
  minDistanceLineNeighbors = {}
  for rectangleIndex, rectangle in enumerate(rectangles):
    for lineIndex, line in enumerate(lines):
      lineDim = calcLineDim(linePoints, line)
      #if lineDim > 1: print('-*'*10, 850); exit(1)
      if lineDim > 1: continue
      #print([rectanglePoints[rectangle[i]] for i in range(4)]); exit(1)

      minValue = max(rectanglePoints[rectangle[0]][lineDim], rectanglePoints[rectangle[2 - lineDim]][lineDim], linePoints[line[0]][lineDim])
      maxValue = min(rectanglePoints[rectangle[1 + lineDim]][lineDim], rectanglePoints[rectangle[3]][lineDim], linePoints[line[1]][lineDim])

      if maxValue - minValue < gap:
        continue

      rectangleFixedValue_1 = (rectanglePoints[rectangle[0]][1 - lineDim] + rectanglePoints[rectangle[1 + lineDim]][1 - lineDim]) // 2
      rectangleFixedValue_2 = (rectanglePoints[rectangle[2 - lineDim]][1 - lineDim] + rectanglePoints[rectangle[3]][1 - lineDim]) // 2
      lineFixedValue = (linePoints[line[0]][1 - lineDim] + linePoints[line[1]][1 - lineDim]) // 2

      if lineFixedValue < rectangleFixedValue_2 - gap and lineFixedValue > rectangleFixedValue_1 + gap:
        continue

      if lineFixedValue <= rectangleFixedValue_1 + gap:
        index = lineDim * 2 + 0
        distance = rectangleFixedValue_1 - lineFixedValue
        if index not in minDistanceLineNeighbors or distance < minDistanceLineNeighbors[index][1]:
          minDistanceLineNeighbors[index] = (lineIndex, distance, 1 - lineDim)
      else:
        index = lineDim * 2 + 1
        distance = lineFixedValue - rectangleFixedValue_2
        if index not in minDistanceLineNeighbors or distance < minDistanceLineNeighbors[index][1]:
          minDistanceLineNeighbors[index] = (lineIndex, distance, lineDim)

      if lineFixedValue < rectangleFixedValue_1 - distanceThreshold or lineFixedValue > rectangleFixedValue_2 + distanceThreshold:
        continue

      if lineFixedValue <= rectangleFixedValue_1 + gap:
        if lineDim == 0:
          rectangleLineNeighbors[rectangleIndex][lineIndex] = 1
        else:
          rectangleLineNeighbors[rectangleIndex][lineIndex] = 0
          pass
        pass
      else:
        if lineDim == 0:
          rectangleLineNeighbors[rectangleIndex][lineIndex] = 0
        else:
          rectangleLineNeighbors[rectangleIndex][lineIndex] = 1
          pass
        pass

      continue
    if len(rectangleLineNeighbors[rectangleIndex]) == 0 or True:
      for index, lineNeighbor in minDistanceLineNeighbors.items():
        rectangleLineNeighbors[rectangleIndex][lineNeighbor[0]] = lineNeighbor[2]
        continue
      pass
    continue

  return rectangleLineNeighbors

## Find the door line to wall line map
def findLineMap(points, lines, points_2, lines_2, gap):
  lineMap = [{} for lineIndex in range(len(lines))]
  for lineIndex, line in enumerate(lines):
    lineDim = calcLineDim(points, line, True)
    for neighborLineIndex, neighborLine in enumerate(lines_2):
      neighborLineDim = calcLineDim(points_2, neighborLine)
      if lineDim != neighborLineDim:
        continue

      if lineDim > 1: 
        p0, p1 = points[line[0]][:2], points[line[1]][:2]
        ln = [points_2[neighborLine[0]][:2], points_2[neighborLine[1]][:2]]
        #print(p0, p1, ln)
        if p2line(p0, ln) > gap:
          continue
        pp0, pp1 = project_p_toline(p0, ln), project_p_toline(p1, ln)
        if (pInLine(pp0, ln) and pInLine(pp1, ln)) and p2p(pp0, pp1) >= gap:
          ratio = 1.0
        elif (pInLine(pp0, ln) and (not pInLine(pp1, ln))) and p2p(pp0, ln[1]) >= gap:
          ratio = (p2p(pp0, ln[1]) + 1.0) / (p2p(p0, p1) + 1.0)
        elif ((not pInLine(pp0, ln)) and pInLine(pp1, ln)) and p2p(pp1, ln[0]) >= gap:
          ratio = (p2p(pp1, ln[0]) + 1.0) / (p2p(p0, p1) + 1.0)
        else:
          continue
        #print(ratio, '-------------'); exit(1)
        lineMap[lineIndex][neighborLineIndex] = ratio
        continue

      minValue = max(points[line[0]][lineDim], points_2[neighborLine[0]][lineDim])
      maxValue = min(points[line[1]][lineDim], points_2[neighborLine[1]][lineDim])
      if maxValue - minValue < gap:
        continue
      fixedValue_1 = (points[line[0]][1 - lineDim] + points[line[1]][1 - lineDim]) // 2
      fixedValue_2 = (points_2[neighborLine[0]][1 - lineDim] + points_2[neighborLine[1]][1 - lineDim]) // 2

      if abs(fixedValue_2 - fixedValue_1) > gap: #dsc
        continue

      lineMinValue = points[line[0]][lineDim]
      lineMaxValue = points[line[1]][lineDim]
      ratio = float(maxValue - minValue + 1) / (lineMaxValue - lineMinValue + 1)

      lineMap[lineIndex][neighborLineIndex] = ratio
      continue
    continue

  return lineMap


## Find the one-to-one door line to wall line map after optimization
def findLineMapSingle(points, lines, points_2, lines_2, gap):
  lineMap = []
  for lineIndex, line in enumerate(lines):
    lineDim = calcLineDim(points, line, True)
    minDistance = max(width, height)
    minDistanceLineIndex = -1
    for neighborLineIndex, neighborLine in enumerate(lines_2):
      neighborLineDim = calcLineDim(points_2, neighborLine)
      if lineDim != neighborLineDim:
        continue

      if lineDim > 1:
        dp0, dp1 = points[line[0]][:2], points[line[1]][:2]
        wp0, wp1 = points_2[neighborLine[0]][:2], points_2[neighborLine[1]][:2]

        if p2line(dp0, [wp0, wp1]) < gap:
          continue

        pdp0, pdp1 = project_p_toline(dp0, [wp0, wp1]), project_p_toline(dp1, [wp0, wp1])
        pwp0, pwp1 = project_p_toline(wp0, [dp0, dp1]), project_p_toline(wp1, [dp0, dp1])

        in_pdp0, in_pdp1 = pInLine(pdp0, [wp0, wp1]), pInLine(pdp1, [wp0, wp1])
        in_pwp0, in_pwp1 = pInLine(pwp0, [dp0, dp1]), pInLine(pwp1, [dp0, dp1])

        if (in_pdp0 and in_pdp1 and p2p(dp0, dp1) < gap) or (in_pwp0 and in_pwp1 and p2p(wp0, wp1) < gap):
          continue
        elif (in_pdp0 and not in_pdp1) and p2p(pdp0, wp1) < gap:
          continue
        elif (not in_pdp0 and in_pdp1) and p2p(wp0, pdp1) < gap:
          continue
        else:
          continue

        distance = line2line([dp0, dp1], [wp0, wp1])
      else:
        minValue = max(points[line[0]][lineDim], points_2[neighborLine[0]][lineDim])
        maxValue = min(points[line[1]][lineDim], points_2[neighborLine[1]][lineDim])
        if maxValue - minValue < gap:
          continue
        fixedValue_1 = (points[line[0]][1 - lineDim] + points[line[1]][1 - lineDim]) // 2
        fixedValue_2 = (points_2[neighborLine[0]][1 - lineDim] + points_2[neighborLine[1]][1 - lineDim]) // 2

        distance = abs(fixedValue_2 - fixedValue_1)

      if distance < minDistance:
        minDistance = distance
        minDistanceLineIndex = neighborLineIndex
        pass
      continue

    #if abs(fixedValue_2 - fixedValue_1) > gap:
    #continue
    #print((lineIndex, minDistance, minDistanceLineIndex))
    lineMap.append(minDistanceLineIndex)
    continue

  return lineMap


## Find conflicting line pairs
def findConflictLinePairs(points, lines, gap, distanceThreshold, considerEndPoints=False, door=False):
  conflictLinePairs = []
  for lineIndex_1, line_1 in enumerate(lines):
    lineDim_1 = calcLineDim(points, line_1, door)
    point_1 = points[line_1[0]]
    point_2 = points[line_1[1]]
    lp_1 = [point_1[:2], point_2[:2]]

    fixedValue_1 = int(round((point_1[1 - lineDim_1] + point_2[1 - lineDim_1]) // 2))
    minValue_1 = int(min(point_1[lineDim_1], point_2[lineDim_1]))
    maxValue_1 = int(max(point_1[lineDim_1], point_2[lineDim_1]))

    for lineIndex_2, line_2 in enumerate(lines):
      if lineIndex_2 <= lineIndex_1:
        continue

      lineDim_2 = calcLineDim(points, line_2, door)
      point_1 = points[line_2[0]]
      point_2 = points[line_2[1]]
      lp_2 = [point_1[:2], point_2[:2]]
      
      if lineDim_2 == lineDim_1:
        if line_1[0] == line_2[0] or line_1[1] == line_2[1]:
          #if lineDim_1 < 2: continue
          conflictLinePairs.append((lineIndex_1, lineIndex_2))
          continue
        elif line_1[0] == line_2[1] or line_1[1] == line_2[0]:
          continue
        pass
      else:
        if (line_1[0] in line_2 or line_1[1] in line_2):
          continue
        pass

      if considerEndPoints:
        if min([pointDistance(points[line_1[0]], points[line_2[0]]), pointDistance(points[line_1[0]], points[line_2[1]]), pointDistance(points[line_1[1]], points[line_2[0]]), pointDistance(points[line_1[1]], points[line_2[1]])]) <= gap:
          conflictLinePairs.append((lineIndex_1, lineIndex_2))
          continue
        pass

      fixedValue_2 = int(round((point_1[1 - lineDim_2] + point_2[1 - lineDim_2]) // 2))
      minValue_2 = int(min(point_1[lineDim_2], point_2[lineDim_2]))
      maxValue_2 = int(max(point_1[lineDim_2], point_2[lineDim_2]))

      if lineDim_1 == lineDim_2:
        if lineDim_1 < 2 and (abs(fixedValue_2 - fixedValue_1) >= distanceThreshold or minValue_1 > maxValue_2 - gap or minValue_2 > maxValue_1 - gap): continue

        moved_p1, moved_p2 = move_p_inline(lp_1[0], lp_1, gap), move_p_inline(lp_2[0], lp_2, gap)
        if lineDim_1 == 2 and (line2line(lp_1, lp_2) >= distanceThreshold or (moved_p1[0] > lp_2[1][0] and moved_p1[1] > lp_2[1][1]) or (moved_p2[0] > lp_1[1][0] and moved_p2[1] > lp_1[1][1])): continue
        if lineDim_1 == 3 and (line2line(lp_1, lp_2) >= distanceThreshold or (moved_p1[0] > lp_2[1][0] and moved_p1[1] < lp_2[1][1]) or (moved_p2[0] > lp_1[1][0] and moved_p2[1] < lp_1[1][1])): continue

        if lineDim_1 < 2: continue
        conflictLinePairs.append((lineIndex_1, lineIndex_2))
      else:
        if lineDim_1 < 2 and lineDim_2 < 2 and (minValue_1 > fixedValue_2 - gap or maxValue_1 < fixedValue_2 + gap or minValue_2 > fixedValue_1 - gap or maxValue_2 < fixedValue_1 + gap): continue

        #p = interLine(lp_1, lp_2)
        #if any((not pInLine(p, l)) for l in [lp_1, lp_2]): continue
        #if any(p2p(p, lp) < gap for l in [lp_1, lp_2] for lp in l): continue
        if lineDim_1 > 1 or lineDim_2 > 1: continue
        conflictLinePairs.append((lineIndex_1, lineIndex_2))
        pass
      continue
    continue
  #return []
  return conflictLinePairs


## Find conflicting line/icon pairs
def findConflictRectanglePairs(points, rectangles, gap):
  conflictRectanglePairs = []
  for rectangleIndex_1, rectangle_1 in enumerate(rectangles):
    for rectangleIndex_2, rectangle_2 in enumerate(rectangles):
      if rectangleIndex_2 <= rectangleIndex_1:
        continue

      conflict = False
      for cornerIndex in range(4):
        if rectangle_1[cornerIndex] == rectangle_2[cornerIndex]:
          conflictRectanglePairs.append((rectangleIndex_1, rectangleIndex_2))
          conflict = True
          break
        continue

      if conflict:
        continue

      minX = max((points[rectangle_1[0]][0] + points[rectangle_1[2]][0]) // 2, (points[rectangle_2[0]][0] + points[rectangle_2[2]][0]) // 2)
      maxX = min((points[rectangle_1[1]][0] + points[rectangle_1[3]][0]) // 2, (points[rectangle_2[1]][0] + points[rectangle_2[3]][0]) // 2)
      if minX > maxX - gap:
        continue
      minY = max((points[rectangle_1[0]][1] + points[rectangle_1[1]][1]) // 2, (points[rectangle_2[0]][1] + points[rectangle_2[1]][1]) // 2)
      maxY = min((points[rectangle_1[2]][1] + points[rectangle_1[3]][1]) // 2, (points[rectangle_2[2]][1] + points[rectangle_2[3]][1]) // 2)
      if minY > maxY - gap:
        continue
      conflictRectanglePairs.append((rectangleIndex_1, rectangleIndex_2))
      continue
    continue

  return conflictRectanglePairs


## Find conflicting icon pairs
def findConflictRectangleLinePairs(rectanglePoints, rectangles, linePoints, lines, gap):
  conflictRectangleLinePairs = []
  for rectangleIndex, rectangle in enumerate(rectangles):
    for lineIndex, line in enumerate(lines):
      lineDim = calcLineDim(linePoints, line)
      #if lineDim > 1: print('-*'*10, 1073); exit(1)
      if lineDim > 1: continue
      if lineDim == 0:
        minX = max(rectanglePoints[rectangle[0]][0], rectanglePoints[rectangle[2]][0], linePoints[line[0]][0])
        maxX = min(rectanglePoints[rectangle[1]][0], rectanglePoints[rectangle[3]][0], linePoints[line[1]][0])
        if minX > maxX - gap:
          continue
        if max(rectanglePoints[rectangle[0]][1], rectanglePoints[rectangle[1]][1]) + gap > min(linePoints[line[0]][1], linePoints[line[1]][1]):
          continue
        if min(rectanglePoints[rectangle[2]][1], rectanglePoints[rectangle[3]][1]) - gap < max(linePoints[line[0]][1], linePoints[line[1]][1]):
          continue

      elif lineDim == 1:
        minY = max(rectanglePoints[rectangle[0]][1], rectanglePoints[rectangle[1]][1], linePoints[line[0]][1])
        maxY = min(rectanglePoints[rectangle[2]][1], rectanglePoints[rectangle[3]][1], linePoints[line[1]][1])
        if minY > maxY - gap:
          continue
        if max(rectanglePoints[rectangle[0]][0], rectanglePoints[rectangle[2]][0]) + gap > min(linePoints[line[0]][0], linePoints[line[1]][0]):
          continue
        if min(rectanglePoints[rectangle[1]][0], rectanglePoints[rectangle[3]][0]) - gap < max(linePoints[line[0]][0], linePoints[line[1]][0]):
          continue

      conflictRectangleLinePairs.append((rectangleIndex, lineIndex))
      continue
    continue

  return conflictRectangleLinePairs

## Find point to line map
def findLinePointMap(points, lines, points_2, gap):
  lineMap = [[] for lineIndex in range(len(lines))]
  for lineIndex, line in enumerate(lines):
    lineDim = calcLineDim(points, line, True)
    if lineDim > 1:
      dp0, dp1 = points[line[0]], points[line[1]]
      l = [dp0, dp1]
      for neighborPointIndex, neighborPoint in enumerate(points_2):
        if p2line(neighborPoint, l) > gap: continue
        pnp = project_p_toline(neighborPoint, l)
        if not (pInLine(pnp, l) and p2p(pnp, dp0) >= gap and p2p(pnp, dp1) >= gap): continue
      lineMap[lineIndex].append(neighborPointIndex)
      continue
    fixedValue = (points[line[0]][1 - lineDim] + points[line[1]][1 - lineDim]) // 2
    for neighborPointIndex, neighborPoint in enumerate(points_2):
      if neighborPoint[lineDim] < points[line[0]][lineDim] + gap or neighborPoint[lineDim] > points[line[1]][lineDim] - gap:
        continue

      if abs((neighborPoint[1 - lineDim] + neighborPoint[1 - lineDim]) // 2 - fixedValue) > gap:
        continue

      lineMap[lineIndex].append(neighborPointIndex)
      continue
    continue
  return lineMap

## Generate primitive candidates from heatmaps
def findCandidatesFromHeatmaps(iconHeatmaps, iconPointOffset, doorPointOffset):
  newIcons = []
  newIconPoints = []
  newDoorLines = []
  newDoorPoints = []
  for iconIndex in range(1, NUM_ICONS + 2):
    heatmap = iconHeatmaps[:, :, iconIndex] > 0.5
    kernel = np.ones((3, 3), dtype=np.uint8)
    heatmap = cv2.dilate(cv2.erode(heatmap.astype(np.uint8), kernel), kernel)
    regions = measure.label(heatmap, background=0)
    for regionIndex in range(regions.min() + 1, regions.max() + 1):
      regionMask = regions == regionIndex
      ys, xs = regionMask.nonzero()
      minX, maxX = xs.min(), xs.max()
      minY, maxY = ys.min(), ys.max()
      if iconIndex <= NUM_ICONS:
        if maxX - minX < GAPS['icon_extraction'] or maxY - minY < GAPS['icon_extraction']:
          continue
        mask = regionMask[minY:maxY + 1, minX:maxX + 1]
        sizeX, sizeY = maxX - minX + 1, maxY - minY + 1
        sumX = mask.sum(0)

        for x in range(sizeX):
          if sumX[x] * 2 >= sizeY:
            break
          minX += 1
          continue

        for x in range(sizeX - 1, -1, -1):
          if sumX[x] * 2 >= sizeY:
            break
          maxX -= 1
          continue


        sumY = mask.sum(1)
        for y in range(sizeY):
          if sumY[y] * 2 >= sizeX:
            break
          minY += 1
          continue

        for y in range(sizeY - 1, -1, -1):
          if sumY[y] * 2 >= sizeX:
            break
          maxY -= 1
          continue
        if (maxY - minY + 1) * (maxX - minX + 1) <= LENGTH_THRESHOLDS['icon'] * LENGTH_THRESHOLDS['icon'] * 2:
          continue
        newIconPoints += [[minX, minY, 1, 2], [maxX, minY, 1, 3], [minX, maxY, 1, 1], [maxX, maxY, 1, 0]]
        newIcons.append((iconPointOffset, iconPointOffset + 1, iconPointOffset + 2, iconPointOffset + 3))
        iconPointOffset += 4
      else:
        sizeX, sizeY = maxX - minX + 1, maxY - minY + 1
        if sizeX >= LENGTH_THRESHOLDS['door'] and sizeY * 2 <= sizeX:
          newDoorPoints += [[minX, (minY + maxY) // 2, 0, 1], [maxX, (minY + maxY) // 2, 0, 3]]
          newDoorLines.append((doorPointOffset, doorPointOffset + 1))
          doorPointOffset += 2
        elif sizeY >= LENGTH_THRESHOLDS['door'] and sizeX * 2 <= sizeY:
          newDoorPoints += [[(minX + maxX) // 2, minY, 0, 2], [(minX + maxX) // 2, maxY, 0, 0]]
          newDoorLines.append((doorPointOffset, doorPointOffset + 1))
          doorPointOffset += 2
        elif sizeX >= LENGTH_THRESHOLDS['door'] and sizeY >= LENGTH_THRESHOLDS['door']:
          mask = regionMask[minY:maxY + 1, minX:maxX + 1]
          sumX = mask.sum(0)
          minOffset, maxOffset = 0, 0
          for x in range(sizeX):
            if sumX[x] * 2 >= sizeY:
              break
            minOffset += 1
            continue

          for x in range(sizeX - 1, -1, -1):
            if sumX[x] * 2 >= sizeY:
              break
            maxOffset += 1
            continue

          if (sizeX - minOffset - maxOffset) * 2 <= sizeY and sizeX - minOffset - maxOffset > 0:
            newDoorPoints += [[(minX + minOffset + maxX - maxOffset) // 2, minY, 0, 2], [(minX + minOffset + maxX - maxOffset) // 2, maxY, 0, 0]]
            newDoorLines.append((doorPointOffset, doorPointOffset + 1))
            doorPointOffset += 2
            pass

          sumY = mask.sum(1)
          minOffset, maxOffset = 0, 0
          for y in range(sizeY):
            if sumY[y] * 2 >= sizeX:
              break
            minOffset += 1
            continue

          for y in range(sizeY - 1, -1, -1):
            if sumY[y] * 2 >= sizeX:
              break
            maxOffset += 1
            continue

          if (sizeY - minOffset - maxOffset) * 2 <= sizeX and sizeY - minOffset - maxOffset > 0:
            newDoorPoints += [[minX, (minY + minOffset + maxY - maxOffset) // 2, 0, 1], [maxX, (minY + minOffset + maxY - maxOffset) // 2, 0, 3]]
            newDoorLines.append((doorPointOffset, doorPointOffset + 1))
            doorPointOffset += 2
            pass
          pass
        pass
      continue
    continue
  return newIcons, newIconPoints, newDoorLines, newDoorPoints

## Sort lines so that the first point always has smaller x or y
def sortLines(points, lines, door=False):
  for lineIndex, line in enumerate(lines):
    lineDim = calcLineDim(points, line, door)
    #if lineDim > 1: print('-*'*10, 1233); exit(1)
    if lineDim < 2:
      if points[line[0]][lineDim] > points[line[1]][lineDim]:
        lines[lineIndex] = (line[1], line[0])
    elif lineDim == 2:
      if not (points[line[0]][0] < points[line[1]][0] and points[line[0]][1] < points[line[1]][1]):
        lines[lineIndex] = (line[1], line[0])
    else:
      if not (points[line[0]][0] < points[line[1]][0] and points[line[0]][1] > points[line[1]][1]):
        lines[lineIndex] = (line[1], line[0])

## Reconstruct a floorplan via IP optimization
def reconstructFloorplan(
    wallCornerHeatmaps, doorCornerHeatmaps, iconCornerHeatmaps, iconHeatmaps, roomHeatmaps, 
    output_prefix='test/', 
    densityImage=None, 
    gt_dict=None, 
    gt=False, 
    gap=-1, 
    distanceThreshold=-1, 
    lengthThreshold=-1, 
    debug_prefix='test', 
    heatmapValueThresholdWall=None, 
    heatmapValueThresholdDoor=None, 
    heatmapValueThresholdIcon=None, 
    enableAugmentation=False):
  print('reconstruct')

  wallPoints = []
  iconPoints = []
  doorPoints = []
  
  numWallPoints = 100
  numDoorPoints = 100
  numIconPoints = 100
  if heatmapValueThresholdWall is None:
    heatmapValueThresholdWall = 0.5
  
  if heatmapValueThresholdDoor is None:
    heatmapValueThresholdDoor = 0.5
  
  if heatmapValueThresholdIcon is None:
    heatmapValueThresholdIcon = 0.5
    
  gap = 3
  if gap > 0:
    for k in GAPS:
      GAPS[k] = gap
      continue
    pass
  if distanceThreshold > 0:
    for k in DISTANCES:
      DISTANCES[k] = distanceThreshold
      continue
    pass
  if lengthThreshold > 0:
    for k in LENGTH_THRESHOLDS:
      LENGTH_THRESHOLDS[k] = lengthThreshold
      continue
    pass

  wallPoints, wallLines, wallPointOrientationLinesMap, wallPointNeighbors, augmentedPointMask = extractCorners(wallCornerHeatmaps, heatmapValueThresholdWall, gap=GAPS['wall_extraction'], augment=enableAugmentation, gt=gt)
  doorPoints, doorLines, doorPointOrientationLinesMap, doorPointNeighbors, _ = extractCorners(doorCornerHeatmaps, heatmapValueThresholdDoor, gap=GAPS['door_extraction'], cornerType='door', gt=gt)
  iconPoints, iconLines, iconPointOrientationLinesMap, iconPointNeighbors, _ = extractCorners(iconCornerHeatmaps, heatmapValueThresholdIcon, gap=GAPS['icon_extraction'], cornerType='icon', gt=gt)
  if not gt:
    for pointIndex, point in enumerate(wallPoints):
      #print((pointIndex, np.array(point[:2]).astype(np.int32).tolist(), point[2], point[3]))
      continue

    wallPoints, wallLines, wallPointOrientationLinesMap, wallPointNeighbors = filterWalls(wallPoints, wallLines)
    pass


  sortLines(doorPoints, doorLines, True)
  sortLines(wallPoints, wallLines)

  print('the number of points', len(wallPoints), len(doorPoints), len(iconPoints))
  print('the number of lines', len(wallLines), len(doorLines), len(iconLines))


  drawPoints(os.path.join(debug_prefix, "points.png"), width, height, wallPoints, densityImage, pointSize=3)
  drawPointsSeparately(os.path.join(debug_prefix, 'points'), width, height, wallPoints, densityImage, pointSize=3)
  drawLines(os.path.join(debug_prefix, 'lines.png'), width, height, wallPoints, wallLines, [], None, 1, lineColor=255)

  wallMask = drawLineMask(width, height, wallPoints, wallLines)

  labelVotesMap = np.zeros((NUM_ROOMS, height, width))
  labelVotesMap1 = np.zeros((NUM_ROOMS, height, width))
  #labelMap = np.zeros((NUM_LABELS, height, width))
  #semanticHeatmaps = np.concatenate([iconHeatmaps, roomHeatmaps], axis=2)
  #print(roomHeatmaps.shape); exit(1)
  for segmentIndex in range(NUM_ROOMS):
    segmentation_img = roomHeatmaps[:, :, segmentIndex]
    #segmentation_img = (segmentation_img > 0.5).astype(np.float)
    labelVotesMap[segmentIndex] = segmentation_img
    labelVotesMap1[segmentIndex] = segmentation_img
    #labelMap[segmentIndex] = segmentation_img
    continue

  labelVotesMap1 = np.cumsum(np.cumsum(labelVotesMap1, axis=1), axis=2)

  icons = findIconsFromLines(iconPoints, iconLines)
  
  if not gt:
    newIcons, newIconPoints, newDoorLines, newDoorPoints = findCandidatesFromHeatmaps(iconHeatmaps, len(iconPoints), len(doorPoints))

    icons += newIcons
    iconPoints += newIconPoints
    doorLines += newDoorLines
    doorPoints += newDoorPoints
    pass

  if True:
    drawLines(os.path.join(debug_prefix, 'lines.png'), width, height, wallPoints, wallLines, [], None, 2, lineColor=255)
    drawLines(os.path.join(debug_prefix, 'doors.png'), width, height, doorPoints, doorLines, [], None, 2, lineColor=255)
    drawRectangles(os.path.join(debug_prefix, 'icons.png'), width, height, iconPoints, icons, {}, 2)
    print('number of walls: ' + str(len(wallLines)))
    print('number of doors: ' + str(len(doorLines)))
    print('number of icons: ' + str(len(icons)))
    pass

  doorWallLineMap = findLineMap(doorPoints, doorLines, wallPoints, wallLines, gap=GAPS['wall_door_neighbor'])

  newDoorLines = []
  newDoorWallLineMap = []
  for lineIndex, walls in enumerate(doorWallLineMap):
    if len(walls) > 0:
      newDoorLines.append(doorLines[lineIndex])
      newDoorWallLineMap.append(walls)
      pass
    continue
  doorLines = newDoorLines
  doorWallLineMap = newDoorWallLineMap


  conflictWallLinePairs = findConflictLinePairs(wallPoints, wallLines, gap=GAPS['wall_conflict'], distanceThreshold=DISTANCES['wall'], considerEndPoints=False)

  conflictDoorLinePairs = findConflictLinePairs(doorPoints, doorLines, gap=GAPS['door_conflict'], distanceThreshold=DISTANCES['door'], door=True)
  conflictIconPairs = findConflictRectanglePairs(iconPoints, icons, gap=GAPS['icon_conflict'])

  if False:
    print(wallLines)
    os.system('mkdir ' + debug_prefix + '/lines')
    for lineIndex, line in enumerate(wallLines):
      drawLines(os.path.join(debug_prefix, 'lines/line_' + str(lineIndex) + '.png'), width, height, wallPoints, [line], [], lineColor=255)
      continue
    exit(1)
    pass


  wallLineNeighbors = findLineNeighbors(wallPoints, wallLines, labelVotesMap, labelVotesMap1, gap=GAPS['wall_neighbor'])

  iconWallLineNeighbors = findRectangleLineNeighbors(iconPoints, icons, wallPoints, wallLines, wallLineNeighbors, gap=GAPS['wall_icon_neighbor'], distanceThreshold=DISTANCES['wall_icon'])
  conflictIconWallPairs = findConflictRectangleLinePairs(iconPoints, icons, wallPoints, wallLines, gap=GAPS['wall_icon_conflict'])


  if False:
    print(conflictWallLinePairs)
    for wallIndex in [0, 17]:
      print(wallLines[wallIndex])
      print([wallPoints[pointIndex] for pointIndex in wallLines[wallIndex]])
      print(wallPointOrientationLinesMap[wallLines[wallIndex][0]])
      print(wallPointOrientationLinesMap[wallLines[wallIndex][1]])
      continue
    exit(1)
    pass


  exteriorLines = {}
  for lineIndex, neighbors in enumerate(wallLineNeighbors):
    if len(neighbors[0]) == 0 and len(neighbors[1]) > 0:
      exteriorLines[lineIndex] = 0
    elif len(neighbors[0]) > 0 and len(neighbors[1]) == 0:
      exteriorLines[lineIndex] = 1
      pass
    continue
  #print(exteriorLines)

  if False:
    filteredWallLines = []
    for lineIndex, neighbors in enumerate(wallLineNeighbors):
      if len(neighbors[0]) == 0 and len(neighbors[1]) > 0:
        print(lineIndex)
        filteredWallLines.append(wallLines[lineIndex])
        pass
      continue
    drawLines(os.path.join(debug_prefix, 'exterior_1.png'), width, height, wallPoints, filteredWallLines, lineColor=255)

    filteredWallLines = []
    for lineIndex, neighbors in enumerate(wallLineNeighbors):
      if len(neighbors[0]) > 0 and len(neighbors[1]) == 0:
        print(lineIndex)
        filteredWallLines.append(wallLines[lineIndex])
        pass
      continue
    drawLines(os.path.join(debug_prefix, 'exterior_2.png'), width, height, wallPoints, filteredWallLines, lineColor=255)
    exit(1)
    pass

  if True:
    #model = Model("JunctionFilter")
    model = LpProblem("JunctionFilter", LpMinimize)

    #add variables
    w_p = [LpVariable(cat=LpBinary, name="point_" + str(pointIndex)) for pointIndex in range(len(wallPoints))]
    w_l = [LpVariable(cat=LpBinary, name="line_" + str(lineIndex)) for lineIndex in range(len(wallLines))]

    d_l = [LpVariable(cat=LpBinary, name="door_line_" + str(lineIndex)) for lineIndex in range(len(doorLines))]

    i_r = [LpVariable(cat=LpBinary, name="icon_rectangle_" + str(lineIndex)) for lineIndex in range(len(icons))]

    i_types = []
    for iconIndex in range(len(icons)):
      i_types.append([LpVariable(cat=LpBinary, name="icon_type_" + str(iconIndex) + "_" + str(typeIndex)) for typeIndex in range(NUM_ICONS)])
      continue

    l_dir_labels = []
    for lineIndex in range(len(wallLines)):
      dir_labels = []
      for direction in range(2):
        labels = []
        for label in range(NUM_ROOMS):
          labels.append(LpVariable(cat=LpBinary, name="line_" + str(lineIndex) + "_" + str(direction) + "_" + str(label)))
        dir_labels.append(labels)
      l_dir_labels.append(dir_labels)



    #model.update()
    #obj = QuadExpr()
    obj = LpAffineExpression()
    
    if gt:
      for pointIndex in range(len(wallPoints)):
        model += (w_p[pointIndex] == 1, 'gt_point_active_' + str(pointIndex))
        continue

      pointIconMap = {}
      for iconIndex, icon in enumerate(icons):
        for pointIndex in icon:
          if pointIndex not in pointIconMap:
            pointIconMap[pointIndex] = []
            pass
          pointIconMap[pointIndex].append(iconIndex)
          continue
        continue
      for pointIndex, iconIndices in pointIconMap.items():
        break
        iconSum = LpAffineExpression()
        for iconIndex in iconIndices:
          iconSum += i_r[iconIndex]
          continue
        model += (iconSum == 1)
        continue
      pass
    ## Semantic label one hot constraints
    for lineIndex in range(len(wallLines)):
      for direction in range(2):
        labelSum = LpAffineExpression()
        for label in range(NUM_ROOMS):
          labelSum += l_dir_labels[lineIndex][direction][label]
          continue
        model += (labelSum == w_l[lineIndex], 'label_sum_' + str(lineIndex) + '_' + str(direction))
        continue
      continue

    ## Opposite room constraints
    if False:
      oppositeRoomPairs = [(1, 1), (2, 2), (4, 4), (5, 5), (7, 7), (9, 9)]
      for lineIndex in range(len(wallLines)):
        for oppositeRoomPair in oppositeRoomPairs:
          model += (l_dir_labels[lineIndex][0][oppositeRoomPair[0]] + l_dir_labels[lineIndex][0][oppositeRoomPair[1]] <= 1)
          if oppositeRoomPair[0] != oppositeRoomPair[1]:
            model += (l_dir_labels[lineIndex][0][oppositeRoomPair[1]] + l_dir_labels[lineIndex][0][oppositeRoomPair[0]] <= 1)
            pass
          continue
        continue
      pass

    ## Loop constraints
    closeRooms = {}
    for label in range(NUM_ROOMS):
      closeRooms[label] = True
      continue
    '''
    #closeRooms[1] = False
    #closeRooms[2] = False
    #closeRooms[3] = False
    #closeRooms[8] = False
    #closeRooms[9] = False
    for label in range(NUM_ROOMS):
      if not closeRooms[label]:
        continue
      for pointIndex, orientationLinesMap in enumerate(wallPointOrientationLinesMap):
        #print(orientationLinesMap); exit(1)
        for orientation, lines in orientationLinesMap.items():
          #print('orientation: ', orientation)
          direction = int(orientation in [1, 2, 6, 7])
          lineSum = LpAffineExpression()
          for lineIndex in lines:
            lineSum += l_dir_labels[lineIndex][direction][label]
            continue
          for nextOrientation in range(orientation + 1, 16):
          #for nextOrientation in range(0, 8):
            if (not (nextOrientation%8) in orientationLinesMap) or ((nextOrientation%8) == orientation):
              continue
            nextLines = orientationLinesMap[nextOrientation%8]
            nextDirection = int(nextOrientation%8 in [0, 3, 4, 5])
            nextLineSum = LpAffineExpression()
            for nextLineIndex in nextLines:
              nextLineSum += l_dir_labels[nextLineIndex][nextDirection][label]
              continue
            model += (lineSum == nextLineSum)
            break
          continue
        continue
      continue
    '''

    ## Exterior constraints
    exteriorLineSum = LpAffineExpression()
    for lineIndex in range(len(wallLines)):
      if lineIndex not in exteriorLines:
        continue
      #direction = exteriorLines[lineIndex]
      label = 0
      model += (l_dir_labels[lineIndex][0][label] + l_dir_labels[lineIndex][1][label] == w_l[lineIndex], 'exterior_wall_' + str(lineIndex))
      exteriorLineSum += w_l[lineIndex]
      continue
    model += (exteriorLineSum >= 1, 'exterior_wall_sum')


    ## Wall line room semantic objectives
    for lineIndex, directionNeighbors in enumerate(wallLineNeighbors):
      for direction, neighbors in enumerate(directionNeighbors):
        labelVotesSum = np.zeros(NUM_ROOMS)
        for neighbor, labelVotes in neighbors.items():
          #print(neighbor, labelVotes); exit(1)
          labelVotesSum += labelVotes
          continue

        votesSum = labelVotesSum.sum()
        if votesSum == 0:
          continue
        labelVotesSum /= votesSum

        for label in range(NUM_ROOMS):
          obj += (l_dir_labels[lineIndex][direction][label] * (0.0 - labelVotesSum[label]) * labelWeight)
          continue
        continue
      continue

    ## Icon corner constraints (one icon corner belongs to at most one icon)
    pointIconsMap = {}
    for iconIndex, icon in enumerate(icons):
      for cornerIndex in range(4):
        pointIndex = icon[cornerIndex]
        if pointIndex not in pointIconsMap:
          pointIconsMap[pointIndex] = []
          pass
        pointIconsMap[pointIndex].append(iconIndex)
        continue
      continue

    for pointIndex, iconIndices in pointIconsMap.items():
      iconSum = LpAffineExpression()
      for iconIndex in iconIndices:
        iconSum += i_r[iconIndex]
        continue
      model += (iconSum <= 1)
      continue

    ## Wall confidence objective
    wallLineConfidenceMap = roomHeatmaps[:, :, WALL_LABEL_OFFSET]
    #cv2.imwrite(output_prefix + 'confidence.png', (wallLineConfidenceMap * 255).astype(np.uint8))
    wallConfidences = []
    for lineIndex, line in enumerate(wallLines):
      point_1 = np.array(wallPoints[line[0]][:2])
      point_2 = np.array(wallPoints[line[1]][:2])
      lineDim = calcLineDim(wallPoints, line)
      #if lineDim > 1: print('-*'*10, 1587); exit(1)

      if lineDim > 1:
        rs, cs = poly_sline(point_1, point_2, wallLineWidth, sizes[1], sizes[0])
        img = np.zeros_like(wallLineConfidenceMap)
        img[rs, cs] = 1
        wallLineConfidence = np.sum(wallLineConfidenceMap[img == 1]) / np.sum(img) - 0.5
        #print(wallLineConfidence, lineDim); exit(1)
      else:
        fixedValue = int(round((point_1[1 - lineDim] + point_2[1 - lineDim]) // 2))
        point_1[lineDim], point_2[lineDim] = min(point_1[lineDim], point_2[lineDim]), max(point_1[lineDim], point_2[lineDim])

        point_1[1 - lineDim] = fixedValue - wallLineWidth
        point_2[1 - lineDim] = fixedValue + wallLineWidth
        point_1 = np.maximum(point_1, 0).astype(np.int32)
        point_2 = np.minimum(point_2, sizes - 1).astype(np.int32)

        wallLineConfidence = np.sum(wallLineConfidenceMap[point_1[1]:point_2[1] + 1, point_1[0]:point_2[0] + 1]) / ((point_2[1] + 1 - point_1[1]) * (point_2[0] + 1 - point_1[0])) - 0.5
        #print(wallLineConfidence, lineDim)

      obj += (-wallLineConfidence * w_l[lineIndex] * wallWeight)

      wallConfidences.append(wallLineConfidence)
      continue

    if not gt:
      for wallIndex, wallLine in enumerate(wallLines):
        #print('wall confidence', wallIndex, [np.array(wallPoints[pointIndex][:2]).astype(np.int32).tolist() for pointIndex in wallLine], wallConfidences[wallIndex])
        continue
      pass


    ## Door confidence objective
    doorLineConfidenceMap = iconHeatmaps[:, :, DOOR_LABEL_OFFSET]
    #cv2.imwrite(output_prefix + 'confidence.png', (doorLineConfidenceMap * 255).astype(np.uint8))
    #cv2.imwrite(output_prefix + 'segmentation.png', drawSegmentationImage(doorCornerHeatmaps))

    for lineIndex, line in enumerate(doorLines):
      point_1 = np.array(doorPoints[line[0]][:2])
      point_2 = np.array(doorPoints[line[1]][:2])
      lineDim = calcLineDim(doorPoints, line, True)
      if lineDim > 1:
        if not gt:
          ps = rec_points_sline(point_1, point_2, doorLineWidth, height, width)
          rs, cs = polygon(ps[:, 1], ps[:, 0])
          img = np.zeros_like(doorLineConfidenceMap)
          img[rs, cs] = 1
          doorLineConfidence = np.sum(doorLineConfidenceMap[img == 1]) / np.sum(img)
          doorPointConfidence = sum(doorCornerHeatmaps[ps[i][1], ps[i][0], 3 - i] for i in range(4)) / 4
          doorConfidence = (doorLineConfidence + doorPointConfidence) * 0.5 - 0.5
          obj += (-doorConfidence * d_l[lineIndex] * doorWeight)
        else:
          obj += (-0.5 * d_l[lineIndex] * doorWeight)
        continue
      fixedValue = int(round((point_1[1 - lineDim] + point_2[1 - lineDim]) // 2))

      #assert(point_1[lineDim] < point_2[lineDim], 'door line reversed')
      point_1[lineDim], point_2[lineDim] = min(point_1[lineDim], point_2[lineDim]), max(point_1[lineDim], point_2[lineDim])

      point_1[1 - lineDim] = fixedValue - doorLineWidth
      point_2[1 - lineDim] = fixedValue + doorLineWidth

      point_1 = np.maximum(point_1, 0).astype(np.int32)
      point_2 = np.minimum(point_2, sizes - 1).astype(np.int32)

      if not gt:
        doorLineConfidence = np.sum(doorLineConfidenceMap[point_1[1]:point_2[1] + 1, point_1[0]:point_2[0] + 1]) / ((point_2[1] + 1 - point_1[1]) * (point_2[0] + 1 - point_1[0]))

        if lineDim == 0:
          doorPointConfidence = (doorCornerHeatmaps[point_1[1], point_1[0], 3] + doorCornerHeatmaps[point_2[1], point_2[0], 1]) / 2
        else:
          doorPointConfidence = (doorCornerHeatmaps[point_1[1], point_1[0], 0] + doorCornerHeatmaps[point_2[1], point_2[0], 2]) / 2
          pass
        doorConfidence = (doorLineConfidence + doorPointConfidence) * 0.5 - 0.5
        #print('door confidence', doorConfidence)
        obj += (-doorConfidence * d_l[lineIndex] * doorWeight)
      else:
        obj += (-0.5 * d_l[lineIndex] * doorWeight)
        pass
      continue

    ## Icon confidence objective  
    for iconIndex, icon in enumerate(icons):
      point_1 = iconPoints[icon[0]]
      point_2 = iconPoints[icon[1]]
      point_3 = iconPoints[icon[2]]
      point_4 = iconPoints[icon[3]]

      x_1 = int((point_1[0] + point_3[0]) // 2)
      x_2 = int((point_2[0] + point_4[0]) // 2)
      y_1 = int((point_1[1] + point_2[1]) // 2)
      y_2 = int((point_3[1] + point_4[1]) // 2)

      iconArea = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)

      if iconArea <= 1e-4:
        print(icon)
        print([iconPoints[pointIndex] for pointIndex in icon])
        print('zero size icon')
        exit(1)
        pass

      iconTypeConfidence = iconHeatmaps[y_1:y_2 + 1, x_1:x_2 + 1, :NUM_ICONS + 1].sum(axis=(0, 1)) / iconArea
      iconTypeConfidence = iconTypeConfidence[1:] - iconTypeConfidence[0]

      if not gt:
        iconPointConfidence = (iconCornerHeatmaps[int(round(point_1[1])), int(round(point_1[0])), 2] + iconCornerHeatmaps[int(round(point_2[1])), int(round(point_2[0])), 3] + iconCornerHeatmaps[int(round(point_3[1])), int(round(point_3[0])), 1] + iconCornerHeatmaps[int(round(point_4[1])), int(round(point_4[0])), 0]) // 4 - 0.5
        iconConfidence = (iconTypeConfidence + iconPointConfidence) * 0.5
      else:
        iconConfidence = iconTypeConfidence
        pass

      #print('icon confidence', iconConfidence)
      for typeIndex in range(NUM_ICONS):
        obj += (-i_types[iconIndex][typeIndex] * (iconConfidence[typeIndex]) * iconTypeWeight)
        continue
      continue

    ## Icon type one hot constraints
    for iconIndex in range(len(icons)):
      typeSum = LpAffineExpression()
      for typeIndex in range(NUM_ICONS - 1):
        typeSum += i_types[iconIndex][typeIndex]
        continue
      model += (typeSum == i_r[iconIndex])
      continue


    ## Line sum constraints (each orientation has at most one wall line)
    for pointIndex, orientationLinesMap in enumerate(wallPointOrientationLinesMap):
      for orientation, lines in orientationLinesMap.items():
        #if len(lines) > 1:
        #print(lines)
        lineSum = LpAffineExpression()
        for lineIndex in lines:
          lineSum += w_l[lineIndex]
          continue

        model += (lineSum == w_p[pointIndex], "line_sum_" + str(pointIndex) + "_" + str(orientation))
        continue
      continue

    ## Conflict constraints
    for index, conflictLinePair in enumerate(conflictWallLinePairs):
      model += (w_l[conflictLinePair[0]] + w_l[conflictLinePair[1]] <= 1, 'conflict_wall_line_pair_' + str(index))
      continue

    for index, conflictLinePair in enumerate(conflictDoorLinePairs):
      model += (d_l[conflictLinePair[0]] + d_l[conflictLinePair[1]] <= 1, 'conflict_door_line_pair_' + str(index))
      continue

    for index, conflictIconPair in enumerate(conflictIconPairs):
      model += (i_r[conflictIconPair[0]] + i_r[conflictIconPair[1]] <= 1, 'conflict_icon_pair_' + str(index))
      continue

    for index, conflictLinePair in enumerate(conflictIconWallPairs):
      model += (i_r[conflictLinePair[0]] + w_l[conflictLinePair[1]] <= 1, 'conflict_icon_wall_pair_' + str(index))
      continue


    ## Door wall constraints (a door must sit on one and only one wall)
    for doorIndex, lines in enumerate(doorWallLineMap):
      if len(lines) == 0:
        model += (d_l[doorIndex] == 0, 'door_not_on_walls_' + str(doorIndex))
        continue
      lineSum = LpAffineExpression()
      for lineIndex in lines:
        lineSum += w_l[lineIndex]
        continue
      model += (d_l[doorIndex] <= lineSum, 'd_wall_line_sum_' + str(doorIndex))
      continue

    doorWallPointMap = findLinePointMap(doorPoints, doorLines, wallPoints, gap=GAPS['door_point_conflict'])
    for doorIndex, points in enumerate(doorWallPointMap):
      if len(points) == 0:
        continue
      pointSum = LpAffineExpression()
      for pointIndex in points:
        model += (d_l[doorIndex] + w_p[pointIndex] <= 1, 'door_on_two_walls_' + str(doorIndex) + '_' + str(pointIndex))
        continue
      continue

    if False:
      #model += (w_l[6] == 1)
      pass

    model += obj
    model.solve()

    #model.writeLP(debug_prefix + '/model.lp')
    print('Optimization information', LpStatus[model.status], value(model.objective))

    if LpStatus[model.status] == 'Optimal':
      filteredWallLines = []
      filteredWallLabels = []
      filteredWallTypes = []
      wallPointLabels = [[-1, -1, -1, -1, -1, -1, -1, -1] for pointIndex in range(len(wallPoints))]

      for lineIndex, lineVar in enumerate(w_l):
        #print(lineVar.varValue)
        if lineVar.varValue < 0.5: # wall_thres
          continue
        filteredWallLines.append(wallLines[lineIndex])

        filteredWallTypes.append(0)

        labels = [11, 11]
        for direction in range(2):
          for label in range(NUM_ROOMS):
            if l_dir_labels[lineIndex][direction][label].varValue > 0.5:
              labels[direction] = label
              break
            continue
          continue

        filteredWallLabels.append(labels)
        print('wall', lineIndex, labels, [np.array(wallPoints[pointIndex][:2]).astype(np.int32).tolist() for pointIndex in wallLines[lineIndex]], wallLineNeighbors[lineIndex][0].keys(), wallLineNeighbors[lineIndex][1].keys())
        line = wallLines[lineIndex]
        lineDim = calcLineDim(wallPoints, line)
        i = 2
        m = {0: [1, 1], 1: [1, 0], 2: [0, 0], 3: [0, 1]}
        if lineDim == 0:
          wallPointLabels[line[0]][0] = labels[0]
          wallPointLabels[line[0]][1] = labels[0]
          wallPointLabels[line[0]][2] = labels[1]
          wallPointLabels[line[0]][3] = labels[1]

          wallPointLabels[line[1]][7] = labels[0]
          wallPointLabels[line[1]][6] = labels[0]
          wallPointLabels[line[1]][5] = labels[1]
          wallPointLabels[line[1]][4] = labels[1]
        elif lineDim == 1:
          wallPointLabels[line[0]][2] = labels[0]
          wallPointLabels[line[0]][3] = labels[0]
          wallPointLabels[line[0]][4] = labels[1]
          wallPointLabels[line[0]][5] = labels[1]

          wallPointLabels[line[1]][1] = labels[0]
          wallPointLabels[line[1]][0] = labels[0]
          wallPointLabels[line[1]][7] = labels[1]
          wallPointLabels[line[1]][6] = labels[1]
        elif lineDim == 2:
          #print(wallPoints[line[0]], wallPoints[line[1]], ' dim=2')
          wallPointLabels[line[0]][1] = labels[m[i][0]]
          wallPointLabels[line[0]][2] = labels[m[i][0]]
          wallPointLabels[line[0]][3] = labels[1 - m[i][0]]
          wallPointLabels[line[0]][4] = labels[1 - m[i][0]]

          wallPointLabels[line[1]][0] = labels[m[i][0]]
          wallPointLabels[line[1]][7] = labels[m[i][0]]
          wallPointLabels[line[1]][6] = labels[1 - m[i][0]]
          wallPointLabels[line[1]][5] = labels[1 - m[i][0]]
        else:
          #print(wallPoints[line[0]], wallPoints[line[1]], ' dim=3')
          wallPointLabels[line[0]][7] = labels[m[i][1]]
          wallPointLabels[line[0]][0] = labels[m[i][1]]
          wallPointLabels[line[0]][1] = labels[1 - m[i][1]]
          wallPointLabels[line[0]][2] = labels[1 - m[i][1]]

          wallPointLabels[line[1]][6] = labels[m[i][1]]
          wallPointLabels[line[1]][5] = labels[m[i][1]]
          wallPointLabels[line[1]][4] = labels[1 - m[i][1]]
          wallPointLabels[line[1]][3] = labels[1 - m[i][1]]
          pass
        continue

      if not gt:
        adjustPoints(wallPoints, filteredWallLines)
        mergePoints(wallPoints, filteredWallLines)
        adjustPoints(wallPoints, filteredWallLines)
        filteredWallLabels = [filteredWallLabels[lineIndex] for lineIndex in range(len(filteredWallLines)) if filteredWallLines[lineIndex][0] != filteredWallLines[lineIndex][1]]
        filteredWallLines = [line for line in filteredWallLines if line[0] != line[1]]
        pass
      print('----------number of filteredWallLines: ', len(filteredWallLines))
      drawLines(output_prefix + 'result_line.png', width, height, wallPoints, filteredWallLines, filteredWallLabels, lineColor=255)
      #resultImage = drawLines('', width, height, wallPoints, filteredWallLines, filteredWallLabels, None, lineWidth=5, lineColor=255)

      filteredDoorLines = []
      filteredDoorTypes = []
      for lineIndex, lineVar in enumerate(d_l):
        if lineVar.varValue < 0: # door_t
          continue
        print(('door', lineIndex, [doorPoints[pointIndex][:2] for pointIndex in doorLines[lineIndex]]))
        filteredDoorLines.append(doorLines[lineIndex])

        filteredDoorTypes.append(0)
        continue
      print('-------------number of filteredDoorLines:', len(filteredDoorLines))
      filteredDoorWallMap = findLineMapSingle(doorPoints, filteredDoorLines, wallPoints, filteredWallLines, gap=GAPS['wall_door_neighbor'])
      adjustDoorPoints(doorPoints, filteredDoorLines, wallPoints, filteredWallLines, filteredDoorWallMap)
      drawLines(output_prefix + 'result_door.png', width, height, doorPoints, filteredDoorLines, lineColor=255)

      filteredIcons = []
      filteredIconTypes = []
      for iconIndex, iconVar in enumerate(i_r):
        if iconVar.varValue < 0.5:
          continue

        filteredIcons.append(icons[iconIndex])
        iconType = -1
        for typeIndex in range(NUM_ICONS):
          if i_types[iconIndex][typeIndex].varValue > 0.5:
            iconType = typeIndex
            break
          continue

        print(('icon', iconIndex, iconType, [iconPoints[pointIndex][:2] for pointIndex in icons[iconIndex]]))

        filteredIconTypes.append(iconType)
        continue

      #adjustPoints(iconPoints, filteredIconLines)
      #drawLines(output_prefix + 'lines_results_icon.png', width, height, iconPoints, filteredIconLines)
      drawRectangles(output_prefix + 'result_icon.png', width, height, iconPoints, filteredIcons, filteredIconTypes)

      #resultImage = drawLines('', width, height, doorPoints, filteredDoorLines, [], resultImage, lineWidth=3, lineColor=0)
      #resultImage = drawRectangles('', width, height, iconPoints, filteredIcons, filteredIconTypes, 2, resultImage)
      #cv2.imwrite(output_prefix + 'result.png', resultImage)

      filteredWallPoints = []
      filteredWallPointLabels = []
      orientationMap = {}
      for pointType, orientationOrientations in enumerate(POINT_ORIENTATIONS):
        for orientation, orientations in enumerate(orientationOrientations):
          orientationMap[orientations] = orientation

      for pointIndex, point in enumerate(wallPoints):
        orientations = []
        orientationLines = {}
        for orientation, lines in wallPointOrientationLinesMap[pointIndex].items():
          orientationLine = -1
          for lineIndex in lines:
            if w_l[lineIndex].varValue > 0.5:
              orientations.append(orientation)
              orientationLines[orientation] = lineIndex
              break
            continue
          continue

        if len(orientations) == 0:
          continue

        #print((pointIndex, orientationLines))

        if len(orientations) < len(wallPointOrientationLinesMap[pointIndex]):
          print('invalid point', pointIndex, orientations, wallPointOrientationLinesMap[pointIndex])
          print(wallPoints[pointIndex])
          wallPoints[pointIndex][2] = len(orientations) - 1
          orientations = tuple(orientations)
          if orientations not in orientationMap:
            continue
          wallPoints[pointIndex][3] = orientationMap[orientations]
          print(wallPoints[pointIndex])
          exit(1)
          pass

        filteredWallPoints.append(wallPoints[pointIndex])
        filteredWallPointLabels.append(wallPointLabels[pointIndex])
        continue


      with open(output_prefix + 'floorplan.txt', 'w') as result_file:
        result_file.write(str(width) + '\t' + str(height) + '\n')
        result_file.write(str(len(filteredWallLines)) + '\n')
        for wallIndex, wall in enumerate(filteredWallLines):
          point_1 = wallPoints[wall[0]]
          point_2 = wallPoints[wall[1]]

          result_file.write(str(point_1[0]) + '\t' + str(point_1[1]) + '\t')
          result_file.write(str(point_2[0]) + '\t' + str(point_2[1]) + '\t')
          result_file.write(str(filteredWallLabels[wallIndex][0]) + '\t' + str(filteredWallLabels[wallIndex][1]) + '\n')

        for doorIndex, door in enumerate(filteredDoorLines):
          point_1 = doorPoints[door[0]]
          point_2 = doorPoints[door[1]]

          result_file.write(str(point_1[0]) + '\t' + str(point_1[1]) + '\t')
          result_file.write(str(point_2[0]) + '\t' + str(point_2[1]) + '\t')
          result_file.write('door\t')
          result_file.write(str(filteredDoorTypes[doorIndex] + 1) + '\t1\n')

        for iconIndex, icon in enumerate(filteredIcons):
          point_1 = iconPoints[icon[0]]
          point_2 = iconPoints[icon[1]]
          point_3 = iconPoints[icon[2]]
          point_4 = iconPoints[icon[3]]

          x_1 = int((point_1[0] + point_3[0]) // 2)
          x_2 = int((point_2[0] + point_4[0]) // 2)
          y_1 = int((point_1[1] + point_2[1]) // 2)
          y_2 = int((point_3[1] + point_4[1]) // 2)

          result_file.write(str(x_1) + '\t' + str(y_1) + '\t')
          result_file.write(str(x_2) + '\t' + str(y_2) + '\t')
          result_file.write(iconNumberNameMap[filteredIconTypes[iconIndex]] + '\t')
          #result_file.write(str(iconNumberStyleMap[filteredIconTypes[iconIndex]]) + '\t')
          result_file.write('1\t')
          result_file.write('1\n')

        result_file.close()

      writePoints(filteredWallPoints, filteredWallPointLabels, output_prefix=output_prefix)
        
      if len(filteredDoorLines) > 0:
        writeDoors(doorPoints, filteredDoorLines, filteredDoorTypes, output_prefix=output_prefix)
        pass
      else:
        try:
          os.remove(output_prefix + 'doors_out.txt')
        except OSError:
          pass

      if len(filteredIcons) > 0:
        writeIcons(iconPoints, filteredIcons, filteredIconTypes, output_prefix=output_prefix)
        pass
      else:
        try:
          os.remove(output_prefix + 'icons_out.txt')
        except OSError:
          pass
        pass

    else:
      print('infeasible')
      #model.ComputeIIS()
      #model.write("test/model.ilp")
      return {}
      pass

  result_dict = {'wall': [wallPoints, filteredWallLines, filteredWallLabels], 'door': [doorPoints, filteredDoorLines, []], 'icon': [iconPoints, filteredIcons, filteredIconTypes]}
  return result_dict
