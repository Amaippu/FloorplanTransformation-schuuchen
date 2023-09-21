import numpy as np
import cv2
from skimage.draw import polygon

import torchvision.transforms.functional as F
from PIL import Image
np.bool = np.bool_

# added 45 wall corner patterns and 4 opening corner patterns
NUM_WALL_CORNERS = 58#13
NUM_CORNERS = 70#21
#new CORNER_RANGES = {'wall': (0, 57), 'opening': (58, 65), 'icon': (66, 69)}
#CORNER_RANGES = {'wall': (0, 13), 'opening': (13, 17), 'icon': (17, 21)}

NUM_ICONS = 7
NUM_ROOMS = 10
DOOR_ORIS = [[(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)]]
POINT_ORIENTATIONS = [[(2, ), (3, ), (0, ), (1, )],
                      #[(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)],

                      [(0, 3), (0, 1), (1, 2), (2, 3),
                       (2, 4), (3, 5), (0, 6), (1, 7), 
                       (2, 5), (3, 6), (0, 7), (1, 4), 
                       (5, 6), (6, 7), (4, 7), (4, 5), 
                       (0, 5), (2, 6), (2, 7), (0, 4),
                       (3, 4), (3, 7), (1, 5), (1, 6)],

                      [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2),
                       (0, 2, 4), (0, 2, 7), (0, 2, 5), (0, 2, 6), 
                       (2, 4, 6), (2, 5, 7), (0, 5, 7), (0, 4, 6), 
                       (1, 3, 4), (1, 3, 5), (1, 3, 7), (1, 3, 6), 
                       (1, 4, 6), (1, 5, 7), (3, 5, 7), (3, 4, 6), 
                       (4, 5, 7), (5, 6, 7), (4, 6, 7), (4, 5, 6)],

                      [(0, 1, 2, 3), (1, 3, 4, 6), (1, 3, 5, 7), (0, 2, 4, 6), (0, 2, 5, 7), (4, 5, 6, 7)]] 

class ColorPalette:
    def __init__(self, numColors):
        #np.random.seed(2)
        #self.colorMap = np.random.randint(255, size = (numColors, 3))
        #self.colorMap[0] = 0

        
        self.colorMap = np.array([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [255, 230, 180],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [100, 0, 0],
                                  [0, 100, 0],                                   
                                  [255, 255, 0],                                  
                                  [50, 150, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [128, 128, 80],
                                  [0, 50, 128],                                  
                                  [0, 100, 100],
                                  [0, 255, 128],                                  
                                  [0, 128, 255],
                                  [255, 0, 128],                                  
                                  [128, 0, 255],
                                  [255, 128, 0],                                  
                                  [128, 255, 0],                                                                    
        ])

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.random.randint(255, size = (numColors, 3))
            pass
        
        return

    def getColorMap(self):
        return self.colorMap
    
    def getColor(self, index):
        if index >= colorMap.shape[0]:
            return np.random.randint(255, size = (3))
        else:
            return self.colorMap[index]
            pass
        return

def isManhattan(line, gap=3):
    return min(abs(line[0][0] - line[1][0]), abs(line[0][1] - line[1][1])) < gap

# get four rectange points of a slant line
def rec_points_sline(point_1, point_2, lineWidth, height, width):
    v01, v10 = [point_2[0] - point_1[0], point_2[1] - point_1[1]], [point_1[0] - point_2[0], point_1[1] - point_2[1]]
    v010, v011 = [v01[1], -v01[0]], [-v01[1], v01[0]]
    v102, v103 = [v10[1], -v10[0]], [-v10[1], v10[0]]

    # four points wall rectangle
    p0, p1, p2, p3 = move_p_invec(point_1, v010, lineWidth), move_p_invec(point_1, v011, lineWidth), move_p_invec(point_2, v102, lineWidth), move_p_invec(point_2, v103, lineWidth)
    ps = [np.minimum(np.maximum(p, 0), (height - 1, width - 1)).astype(np.int32) for p in [p0, p1, p2, p3]]
    return np.asarray(ps)

# get the polygon of a slant line
def poly_sline(point_1, point_2, lineWidth, height, width):
  ps = rec_points_sline(point_1, point_2, lineWidth, height, width)
  return polygon(ps[:, 1], ps[:, 0])

# Euclidean distance between two points
def p2p(p1, p2):
    #print('-'*20, p1, p2)
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

# Manhattan distance between line and line
def l2l(l1, l2):
    return min(pointDistance(l1[i], l2[j]) for i in range(2) for j in range(2))

# move point inline towards the second point of line for gap distance
def move_p_inline(p, line, gap=3):
    p0, p1 = line[0], line[1]
    vec = [p1[0] - p0[0], p1[1] - p0[1]]
    return move_p_invec(p, vec, gap)

# move point in direction of vector for gap distance
def move_p_invec(p, v, gap=3):
  l = p2p(v, [0, 0])
  return [p[0] + gap * v[0] / l, p[1] + gap * v[1] / l]

# check if point is in a line
def pInLine(p, l, gap=3):
    return abs(p2p(p, l[0]) + p2p(p, l[1]) - p2p(*l)) < gap*2

# Euclidean distance of a point to a line
def p2line(p, line):
  projected = project_p_toline(p, line)
  return p2p(p, projected)

# Euclidean distance between two lines
def line2line(l1, l2):
  return (sum(p2line(p, l2) for p in l1) * 0.5 + sum(p2line(p, l1) for p in l2) * 0.5) * 0.5

# projection of a point onto a line
def project_p_toline(p, line):

  x = np.array(p, dtype=float)

  u = np.array(line[0], dtype=float)
  v = np.array(line[1], dtype=float)

  n = v - u
  n /= np.linalg.norm(n, 2)

  return u + n*np.dot(x - u, n)

# distance between point and line
def p2l(p, l):
    return min(pointDistance(p, l[i]) for i in range(2))

# intersection point between two lines
def interLine(l1, l2):
  def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

  def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False
  return intersection(line(*l1), line(*l2))

# convert PIL image to ndarray
def pil2np(img):
  if isinstance(img, Image.Image):
    img = np.asarray(img)
  return img

# convert ndarray to PIL image
def np2pil(img):
  if isinstance(img, np.ndarray):
    if img.dtype != np.uint8:
      img = img.astype(np.uint8)
    img = F.to_pil_image(img)
  return img

def calcLineDim2(points, line):
    point_1 = points[line[0]]
    point_2 = points[line[1]]
    if abs(point_2[0] - point_1[0]) > abs(point_2[1] - point_1[1]):
        #print('-'*20, point_1, point_2); exit(1)
        lineDim = 0
    else:
        #print('-'*20, point_1, point_2); exit(1)
        lineDim = 1
        pass
    return lineDim

'''
line dimension: (first point orientation, second point orientation)
0: (0, 2)
1: (3, 1)
2: (6, 4)
3: (5, 7)
'''
def calcLineDim(points, line, door=False):
    point_1 = points[line[0]]
    point_2 = points[line[1]]
    return calcLineDim_(point_1, point_2, door)

def calcLineDim_(point_1, point_2, door=False):
    if abs(point_2[0] - point_1[0]) > abs(point_2[1] - point_1[1]):
        lineDim = 0
    else:
        lineDim = 1
    if door:
      oris_1 = np.asarray(DOOR_ORIS[point_1[2]][point_1[3]])
      oris_2 = np.asarray(DOOR_ORIS[point_2[2]][point_2[3]])
    else:
      try:
        oris_1 = np.asarray(POINT_ORIENTATIONS[point_1[2]][point_1[3]])
        oris_2 = np.asarray(POINT_ORIENTATIONS[point_2[2]][point_2[3]])
      except:
        print(point_1, point_2, door); raise

    if any(oris_1 > 3) and any(oris_2 > 3):
        if 4 in oris_1 and 6 in oris_2 or 6 in oris_1 and 4 in oris_2: lineDim = 2
        elif 5 in oris_1 and 7 in oris_2 or 7 in oris_1 and 5 in oris_2: lineDim = 3

    return lineDim

def calcLineDirection(line, gap=3):
    if isManhattan(line):
    	return int(abs(line[0][0] - line[1][0]) < abs(line[0][1] - line[1][1]))
    # slant patterns
    return ((line[0][0] - line[1][0]) * (line[0][1] - line[1][1]) < 0) + 2

def calcLineDirection2(line, gap=3):
    return int(abs(line[0][0] - line[1][0]) < abs(line[0][1] - line[1][1]))

## Draw segmentation image. The input could be either HxW or HxWxC
def drawSegmentationImage(segmentations, numColors=42, blackIndex=-1, blackThreshold=-1):
    if segmentations.ndim == 2:
        numColors = max(numColors, segmentations.max() + 2)
    else:
        if blackThreshold > 0:
            segmentations = np.concatenate([segmentations, np.ones((segmentations.shape[0], segmentations.shape[1], 1)) * blackThreshold], axis=2)
            blackIndex = segmentations.shape[2] - 1
            pass

        numColors = max(numColors, segmentations.shape[2] + 2)
        pass
    randomColor = ColorPalette(numColors).getColorMap()
    if blackIndex >= 0:
        randomColor[blackIndex] = 0
        pass
    width = segmentations.shape[1]
    height = segmentations.shape[0]
    if segmentations.ndim == 3:
        #segmentation = (np.argmax(segmentations, 2) + 1) * (np.max(segmentations, 2) > 0.5)
        segmentation = np.argmax(segmentations, 2)
    else:
        segmentation = segmentations
        pass

    segmentation = segmentation.astype(np.int32)
    return randomColor[segmentation.reshape(-1)].reshape((height, width, 3))


def drawWallMask(walls, width, height, thickness=3, indexed=False):
    if indexed:
        wallMask = np.full((height, width), -1, dtype=np.int32)
        for wallIndex, wall in enumerate(walls):
            cv2.line(wallMask, (int(wall[0][0]), int(wall[0][1])), (int(wall[1][0]), int(wall[1][1])), color=wallIndex, thickness=thickness)
            continue
    else:
        wallMask = np.zeros((height, width), dtype=np.int32)
        for wall in walls:
            cv2.line(wallMask, (int(wall[0][0]), int(wall[0][1])), (int(wall[1][0]), int(wall[1][1])), color=1, thickness=thickness)
            continue
        wallMask = wallMask.astype(np.bool)
        pass
    return wallMask


def extractCornersFromHeatmaps(heatmaps, heatmapThreshold=0.5, numPixelsThreshold=5, returnRanges=True):
    """Extract corners from heatmaps"""
    from skimage import measure
    heatmaps = (heatmaps > heatmapThreshold).astype(np.float32)
    orientationPoints = []
    #kernel = np.ones((3, 3), np.float32)
    for heatmapIndex in range(0, heatmaps.shape[-1]):
        heatmap = heatmaps[:, :, heatmapIndex]
        #heatmap = cv2.dilate(cv2.erode(heatmap, kernel), kernel)
        components = measure.label(heatmap, background=0)
        points = []
        for componentIndex in range(components.min() + 1, components.max() + 1):
            ys, xs = (components == componentIndex).nonzero()
            if ys.shape[0] <= numPixelsThreshold:
                continue
            #print(heatmapIndex, xs.shape, ys.shape, componentIndex)
            if returnRanges:
                points.append(((xs.mean(), ys.mean()), (xs.min(), ys.min()), (xs.max(), ys.max())))
            else:
                points.append((xs.mean(), ys.mean()))
                pass
            continue
        orientationPoints.append(points)
        continue
    return orientationPoints

def extractCornersFromSegmentation(segmentation, cornerTypeRange=[0, NUM_WALL_CORNERS]):
    """Extract corners from segmentation"""
    from skimage import measure
    orientationPoints = []
    for heatmapIndex in range(cornerTypeRange[0], cornerTypeRange[1]):
        heatmap = segmentation == heatmapIndex
        #heatmap = cv2.dilate(cv2.erode(heatmap, kernel), kernel)
        components = measure.label(heatmap, background=0)
        points = []
        for componentIndex in range(components.min()+1, components.max() + 1):
            ys, xs = (components == componentIndex).nonzero()
            points.append((xs.mean(), ys.mean()))
            continue
        orientationPoints.append(points)
        continue
    return orientationPoints

def getOrientationRanges(width, height):
    #orientationRanges = [[width, 0, 0, 0], [width, height, width, 0], [width, height, 0, height], [0, height, 0, 0]]
    orientationRanges = [[0, height, 0, 0], [width, height, 0, height], [width, height, width, 0], [width, 0, 0, 0]]
    return orientationRanges

def getIconNames():
    iconNames = []
    iconLabelMap = getIconLabelMap()
    for iconName, _ in iconLabelMap.items():
        iconNames.append(iconName)
        continue
    return iconNames

def getIconLabelMap():
    labelMap = {}
    labelMap['bathtub'] = 1
    labelMap['cooking_counter'] = 2
    labelMap['toilet'] = 3
    labelMap['entrance'] = 4
    labelMap['washing_basin'] = 5
    labelMap['special'] = 6
    labelMap['stairs'] = 7
    labelMap['door'] = 8
    return labelMap


def drawPoints(filename, width, height, points, backgroundImage=None, pointSize=5, pointColor=None):
  colorMap = ColorPalette(NUM_CORNERS).getColorMap()
  if np.all(np.equal(backgroundImage, None)):
    image = np.zeros((height, width, 3), np.uint8)
  else:
    if backgroundImage.ndim == 2:
      image = np.tile(np.expand_dims(backgroundImage, -1), [1, 1, 3])
    else:
      image = backgroundImage
      pass
  pass
  no_point_color = pointColor is None
  for point in points:
    if no_point_color:
        pointColor = colorMap[point[2] * 4 + point[3]]
        pass
    #print('used', pointColor)
    #print('color', point[2] , point[3])
    image[max(int(round(point[1])) - pointSize, 0):min(int(round(point[1])) + pointSize, height), max(int(round(point[0])) - pointSize, 0):min(int(round(point[0])) + pointSize, width)] = pointColor
    continue

  if filename != '':
    cv2.imwrite(filename, image)
    return
  else:
    return image

def drawPointsSeparately(path, width, height, points, backgroundImage=None, pointSize=5):
  if np.all(np.equal(backgroundImage, None)):
    image = np.zeros((height, width, NUM_WALL_CORNERS), np.uint8)
  else:
    image = np.tile(np.expand_dims(backgroundImage, -1), [1, 1, NUM_WALL_CORNERS])
    pass

  for point in points:
    image[max(int(round(point[1])) - pointSize, 0):min(int(round(point[1])) + pointSize, height), max(int(round(point[0])) - pointSize, 0):min(int(round(point[0])) + pointSize, width), int(point[2] * 4 + point[3])] = 255
    continue
  for channel in range(NUM_WALL_CORNERS):
    cv2.imwrite(path + '_' + str(channel) + '.png', image[:, :, channel])
    continue
  return

def drawLineMask(width, height, points, lines, lineWidth = 5, backgroundImage = None):
  lineMask = np.zeros((height, width))

  for lineIndex, line in enumerate(lines):
    point_1 = points[line[0]]
    point_2 = points[line[1]]
    direction = calcLineDirectionPoints(points, line)
    if direction > 1:
      rs, cs = poly_sline(point_1, point_2, lineWidth, height, width)
      lineMask[rs, cs] = 1
    else:
      fixedValue = int(round((point_1[1 - direction] + point_2[1 - direction]) / 2))
      minValue = int(min(point_1[direction], point_2[direction]))
      maxValue = int(max(point_1[direction], point_2[direction]))
      if direction == 0:
        lineMask[max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, height), minValue:maxValue + 1] = 1
      else:
        lineMask[minValue:maxValue + 1, max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, width)] = 1
        pass
    continue
  return lineMask



def drawLines(filename, width, height, points, lines, lineLabels = [], backgroundImage = None, lineWidth = 5, lineColor = None):
  colorMap = ColorPalette(len(lines)).getColorMap()
  if backgroundImage is None:
    image = np.ones((height, width, 3), np.uint8) * 0
  else:
    if backgroundImage.ndim == 2:
      image = np.stack([backgroundImage, backgroundImage, backgroundImage], axis=2)
    else:
      image = backgroundImage
      pass
    pass

  for lineIndex, line in enumerate(lines):
    point_1 = points[line[0]]
    point_2 = points[line[1]]
    direction = calcLineDirectionPoints(points, line)

    fixedValue = int(round((point_1[1 - direction] + point_2[1 - direction]) / 2))
    minValue = int(round(min(point_1[direction], point_2[direction])))
    maxValue = int(round(max(point_1[direction], point_2[direction])))
    if len(lineLabels) == 0:
      if np.any(lineColor == None):
        lineColor = np.random.rand(3) * 255
        pass

      if direction > 1:
        rs, cs = poly_sline(point_1, point_2, lineWidth, height, width)
        image[rs, cs, :] = lineColor
      elif direction == 0:
        image[max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, height), minValue:maxValue + 1, :] = lineColor
      elif direction == 1:
        image[minValue:maxValue + 1, max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, width), :] = lineColor
    else:
      labels = lineLabels[lineIndex]
      isExterior = False

      if direction > 1:
        #print(direction, 'with label'); exit(1)
        ps = rec_points_sline(point_1, point_2, lineWidth, height, width)
        ps1 = np.asarray([ps[0], np.array(point_1[:2]), np.array(point_2[:2]), ps[3]], dtype=int)
        ps2 = np.asarray([np.array(point_1[:2]), ps[1], ps[2], np.array(point_2[:2])], dtype=int)
        #print(ps1, ps2); exit(1)
        img = np.zeros_like(image)
        rs1, cs1 = polygon(ps1[:, 1], ps1[:, 0])
        rs2, cs2 = polygon(ps2[:, 1], ps2[:, 0])

        for c in range(3):
          image[rs1, cs1, c] = colorMap[labels[0]][c]
          image[rs2, cs2, c] = colorMap[labels[1]][c]

      elif direction == 0:
        for c in range(3):
          image[max(fixedValue - lineWidth, 0):min(fixedValue, height), minValue:maxValue, c] = colorMap[labels[0]][c]
          image[max(fixedValue, 0):min(fixedValue + lineWidth + 1, height), minValue:maxValue, c] = colorMap[labels[1]][c]
          continue
      else:
        for c in range(3):
          image[minValue:maxValue, max(fixedValue - lineWidth, 0):min(fixedValue, width), c] = colorMap[labels[1]][c]
          image[minValue:maxValue, max(fixedValue, 0):min(fixedValue + lineWidth + 1, width), c] = colorMap[labels[0]][c]
          continue
        pass
      pass
    continue

  if filename == '':
    return image
  else:
    cv2.imwrite(filename, image)


def drawRectangles(filename, width, height, points, rectangles, labels, lineWidth = 2, backgroundImage = None, rectangleColor = None):
  colorMap = ColorPalette(NUM_ICONS).getColorMap()
  if backgroundImage is None:
    image = np.ones((height, width, 3), np.uint8) * 0
  else:
    image = backgroundImage
    pass

  for rectangleIndex, rectangle in enumerate(rectangles):
    point_1 = points[rectangle[0]]
    point_2 = points[rectangle[1]]
    point_3 = points[rectangle[2]]
    point_4 = points[rectangle[3]]


    if len(labels) == 0:
      if rectangleColor is None:
        color = np.random.rand(3) * 255
      else:
        color = rectangleColor
    else:
      color = colorMap[labels[rectangleIndex]]
      pass

    x_1 = int(round((point_1[0] + point_3[0]) / 2))
    x_2 = int(round((point_2[0] + point_4[0]) / 2))
    y_1 = int(round((point_1[1] + point_2[1]) / 2))
    y_2 = int(round((point_3[1] + point_4[1]) / 2))

    cv2.rectangle(image, (x_1, y_1), (x_2, y_2), color=tuple(color.tolist()), thickness = 2)
    continue

  if filename == '':
    return image
  else:
    cv2.imwrite(filename, image)
    pass

def pointDistance(point_1, point_2):
    #return np.sqrt(pow(point_1[0] - point_2[0], 2) + pow(point_1[1] - point_2[1], 2))
    return max(abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))

def calcLineDirectionPoints(points, line):
  point_1 = points[line[0]]
  point_2 = points[line[1]]
  if isinstance(point_1[0], tuple):
      point_1 = point_1[0]
      pass
  if isinstance(point_2[0], tuple):
      point_2 = point_2[0]
      pass
  return calcLineDirection((point_1, point_2))
