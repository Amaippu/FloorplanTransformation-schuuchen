from torch.utils.data import Dataset
import pathlib
import numpy as np
import time

#from plane_dataset_scannet import PlaneDatasetScanNet
#from augmentation import *
from utils import *
from skimage import measure
import cv2
import copy
import os

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

def pointDistance(point_1, point_2):
    #return np.sqrt(pow(point_1[0] - point_2[0], 2) + pow(point_1[1] - point_2[1], 2))
    return max(abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))

def divideWalls(walls):
    horizontalWalls = []
    verticalWalls = []
    for wall in walls:
        if calcLineDirection(wall) == 0:
            horizontalWalls.append(wall)
        else:
            verticalWalls.append(wall)
            pass
        continue
    return horizontalWalls, verticalWalls

def mergeLines(line_1, line_2):
    d = calcLineDirection(line_1)
    min_x, max_x = min(p[0] for l in [line_1, line_2] for p in l), max(p[0] for l in [line_1, line_2] for p in l)
    min_y, max_y = min(p[1] for l in [line_1, line_2] for p in l), max(p[1] for l in [line_1, line_2] for p in l)

    if d == 0:
        #print(d, (min_x, (min_y + max_y) // 2), (max_x, (min_y + max_y) // 2))
        return [(min_x, (min_y + max_y) // 2), (max_x, (min_y + max_y) // 2)]
    elif d == 1:
        #print(d, ((min_x + max_x) // 2, min_y), ((min_x + max_x) // 2, max_y))
        return [((min_x + max_x) // 2, min_y), ((min_x + max_x) // 2, max_y)]
    elif d == 2:
        #print(d, (min_x, min_y), (max_x, max_y))
        return [(min_x, min_y), (max_x, max_y)]
    else:
        #print(d, (min_x, max_y), (max_x, min_y))
        return [(min_x, max_y), (max_x, min_y)]

def findConnections(line_1, line_2, gap, fpath):
    
    if True:
        def sele_ps_offline(c, l):
            return l[0] if pointDistance(c, l[0]) > pointDistance(c, l[1]) else l[1]

        def sele_ps_inline(c, l):
            return [p for p in l if pointDistance(c, p) > gap * 2]

        def p2d(c, ps, f=1):
            d = []
            for p in ps:
                direction = calcLineDirection((p, c))
                if direction == 0:
                    d.append(2 if p[0] > c[0] else 0)
                elif direction == 1:
                    d.append(1 if p[1] > c[1] else 3)
                elif direction == 2:
                    d.append(6 if p[0] > c[0] and p[1] > c[1] else 4)
                else:
                    d.append(7 if p[0] > c[0] and p[1] < c[1] else 5)
            return sorted(d)

        #l1, l2 = line(*line_1), line(*line_2)
        #p = intersection(l1, l2)
        p = interLine(line_1, line_2)
        
        w1, w2 = pInLine(p, line_1, gap), pInLine(p, line_2, gap)

        if not w1 and not w2 and l2l(line_1, line_2) < gap:
            ps = [sele_ps_offline(p, l) for l in [line_1, line_2]]
            return p2d(p, ps), p
        elif w1 and not w2 and p2l(p, line_2) < gap:
            ps = sele_ps_inline(p, line_1) + [sele_ps_offline(p, line_2)]
            return p2d(p, ps), p
        elif not w1 and w2 and p2l(p, line_1) < gap:
            ps = sele_ps_inline(p, line_2) + [sele_ps_offline(p, line_1)]
            return p2d(p, ps, 0), p
        elif w1 and w2:
            ps1 = sele_ps_inline(p, line_1)
            ps2 = sele_ps_inline(p, line_2)
            if ps1 and ps2:
                return p2d(p, ps1 + ps2), p

        return [-1, -1], (0, 0)

def lines2Corners(fpath, lines, gap):
    success = True
    corners = []
    lineConnections = []
    for _ in range(len(lines)):
        lineConnections.append({})
        continue

    connectionCornerMap = {}

    # Manhattan patterns (with comment)
    connectionCornerMap[(0, 3)] = 4 # (0, 3)
    connectionCornerMap[(0, 1)] = 5 # (0, 1)
    connectionCornerMap[(1, 2)] = 6 # (1, 2)
    connectionCornerMap[(2, 3)] = 7 # (2, 3)

    # slant patterns
    connectionCornerMap[(2, 4)] = 8
    connectionCornerMap[(3, 5)] = 9
    connectionCornerMap[(0, 6)] = 10
    connectionCornerMap[(1, 7)] = 11

    connectionCornerMap[(2, 5)] = 12
    connectionCornerMap[(3, 6)] = 13
    connectionCornerMap[(0, 7)] = 14
    connectionCornerMap[(1, 4)] = 15

    connectionCornerMap[(5, 6)] = 16
    connectionCornerMap[(6, 7)] = 17
    connectionCornerMap[(4, 7)] = 18
    connectionCornerMap[(4, 5)] = 19

    connectionCornerMap[(0, 5)] = 20
    connectionCornerMap[(2, 6)] = 21
    connectionCornerMap[(2, 7)] = 22
    connectionCornerMap[(0, 4)] = 23
 
    connectionCornerMap[(3, 4)] = 24
    connectionCornerMap[(3, 7)] = 25
    connectionCornerMap[(1, 5)] = 26
    connectionCornerMap[(1, 6)] = 27

    connectionCornerMap[(1, 2, 3)] = 28 # (1, 2, 3)
    connectionCornerMap[(0, 2, 3)] = 29 # (0, 2, 3)
    connectionCornerMap[(0, 1, 3)] = 30 # (0, 1, 3)
    connectionCornerMap[(0, 1, 2)] = 31 # (0, 1, 2)

    connectionCornerMap[(0, 2, 4)] = 32
    connectionCornerMap[(0, 2, 7)] = 33
    connectionCornerMap[(0, 2, 5)] = 34
    connectionCornerMap[(0, 2, 6)] = 35

    connectionCornerMap[(2, 4, 6)] = 36
    connectionCornerMap[(2, 5, 7)] = 37
    connectionCornerMap[(0, 5, 7)] = 38
    connectionCornerMap[(0, 4, 6)] = 39

    connectionCornerMap[(1, 3, 4)] = 40
    connectionCornerMap[(1, 3, 5)] = 41
    connectionCornerMap[(1, 3, 7)] = 42
    connectionCornerMap[(1, 3, 6)] = 43

    connectionCornerMap[(1, 4, 6)] = 44
    connectionCornerMap[(1, 5, 7)] = 45
    connectionCornerMap[(3, 5, 7)] = 46
    connectionCornerMap[(3, 4, 6)] = 47

    connectionCornerMap[(4, 5, 7)] = 48
    connectionCornerMap[(5, 6, 7)] = 49
    connectionCornerMap[(4, 6, 7)] = 50
    connectionCornerMap[(4, 5, 6)] = 51

    connectionCornerMap[(0, 1, 2, 3)] = 52 # (0, 1, 2, 3)
    connectionCornerMap[(1, 3, 4, 6)] = 53
    connectionCornerMap[(1, 3, 5, 7)] = 54
    connectionCornerMap[(0, 2, 4, 6)] = 55
    connectionCornerMap[(0, 2, 5, 7)] = 56
    connectionCornerMap[(4, 5, 6, 7)] = 57


    corners = []
    visited = []
    for lineIndex_1, line_1 in enumerate(lines):
        for lineIndex_2, line_2 in enumerate(lines):
            if lineIndex_2 == lineIndex_1 or calcLineDirection(line_1) == calcLineDirection(line_2) or (lineIndex_2, lineIndex_1) in visited:
                continue
            visited += [(lineIndex_1, lineIndex_2)]
            connections, connectionPoint = findConnections(line_1, line_2, gap, fpath)
            if connections[0] == -1 and connections[1] == -1:
                continue

            indices = [lineIndex_1, lineIndex_2]
            for c in range(2):
                if connections[c] in [0, 1] and connections[c] in lineConnections[indices[c]] and isManhattan(line_1) and isManhattan(line_2):
                    success = False
                    continue

                lineConnections[indices[c]][connections[c]] = True
                continue
            try:
                corners.append((connectionPoint, connectionCornerMap[tuple(connections)]))
            except:
                print(connections, line_1, line_2, fpath)
                raise
            continue
        continue
    #print(corners)
    return corners, success

def getRoomLabelMap():
    labelMap = {}
    labelMap['living_room'] = 1
    labelMap['kitchen'] = 2
    labelMap['bedroom'] = 3
    labelMap['bathroom'] = 4
    labelMap['restroom'] = 5
    labelMap['washing_room'] = 9
    labelMap['office'] = 3
    labelMap['closet'] = 6
    labelMap['balcony'] = 7
    labelMap['corridor'] = 8
    labelMap['dining_room'] = 9
    labelMap['laundry_room'] = 10
    labelMap['PS'] = 10    
    return labelMap

def getIconLabelMap():
    labelMap = {}
    labelMap['bathtub'] = 1
    labelMap['cooking_counter'] = 2
    labelMap['toilet'] = 3
    labelMap['entrance'] = 4
    labelMap['washing_basin'] = 5
    labelMap['special'] = 6
    labelMap['stairs'] = 7
    return labelMap

def loadLabelMap():
    roomMap = getRoomLabelMap()
    iconMap = getIconLabelMap()

    labelMap = {}
    for icon, label in iconMap.items():
        labelMap[icon] = ('icons', label)
        continue
    for room, label in roomMap.items():    
        labelMap[room] = ('rooms', label)
        continue
    labelMap['door'] = 8
    return labelMap

def augmentSample(options, image, background_colors=[], split='train'):
    max_size = np.random.randint(low=int(options.width * 3 / 4), high=options.width + 1)
    if split != 'train':
        max_size = options.width
        pass
    image_sizes = np.array(image.shape[:2]).astype(np.float32)
    transformation = np.zeros((3, 3))
    transformation[0][0] = transformation[1][1] = float(max_size) / image_sizes.max()
    transformation[2][2] = 1
    image_sizes = (image_sizes / image_sizes.max() * max_size).astype(np.int32)
    
    if image_sizes[1] == options.width or split != 'train':
        offset_x = 0
    else:
        offset_x = np.random.randint(options.width - image_sizes[1])
        pass
    if image_sizes[0] == options.height or split != 'train':
        offset_y = 0
    else:
        offset_y = np.random.randint(options.height - image_sizes[0])
        pass

    transformation[0][2] = offset_x
    transformation[1][2] = offset_y

    if len(background_colors) == 0:
        full_image = np.full((options.height, options.width, 3), fill_value=255)
    else:
        full_image = background_colors[np.random.choice(np.arange(len(background_colors), dtype=np.int32), options.width * options.height)].reshape((options.height, options.width, 3))
        pass
        
    #full_image = np.full((options.height, options.width, 3), fill_value=-1, dtype=np.float32)
    full_image[offset_y:offset_y + image_sizes[0], offset_x:offset_x + image_sizes[1]] = cv2.resize(image, (image_sizes[1], image_sizes[0]))
    image = full_image

    if np.random.randint(2) == 0 and split == 'train':
        image = np.ascontiguousarray(image[:, ::-1])
        transformation[0][0] *= -1
        transformation[0][2] = options.width - transformation[0][2]
        pass
    return image, transformation

def convertToPoint(x, y):
    return (int(round(float(x))), int(round(float(y))))

def transformPoint(transformation, point):
    point = np.array(point)
    point = np.concatenate([point, np.ones(1)], axis=0)
    point = np.matmul(transformation, point)
    return tuple(np.round(point[:2] / point[2]).astype(np.int32).tolist())

## Plane dataset class
class FloorplanDataset(Dataset):
    def __init__(self, options, split, random=True, augment=False, test_batch=False):
        self.options = options
        self.split = split
        self.random = random
        self.augment = augment
        self.imagePaths = []
        # self.dataFolder = '../data/'
        self.dataFolder = f"{options.__dict__.get('dataFolder', pathlib.Path('../data'))}/"#'../data/'
        self.test_batch = test_batch

        bad_anno = ['./anno_label/dqf788hn0qar6zm_stitch.txt',
                    './anno_label/yxty88e5y7o50ra_stitch.txt',
                    './anno_label/7k0a88hn0qar1qq_stitch.txt',
                    './anno_label/4a4h88m7e6ehsvl_stitch.txt',
                    './anno_label/ljpg88k3305opsi_stitch.txt',
                    './anno_label/6ier88q3ct0zqhi_stitch.txt',
                    './anno_label/mk7y88q3ct11idi_stitch.txt',
                    './anno_label/46ur88o6shg1w7a_stitch.txt',
                    './anno_label/ua6x88q3ct10buy_stitch.txt',
                    './anno_label/to3488mncf9aazu_stitch.txt',
                    './anno_label/1fp788jhfvcqm0e_stitch.txt',
                    ]

        with open(self.dataFolder + split + '.txt') as f:
            for line in f:
                #if line.split('\t')[1][:-1] in bad_anno: continue
                self.imagePaths.append([value.strip() for value in line.split('\t')])
                continue
            
        if options.numTrainingImages > 0 and split == 'train':
            self.numImages = options.numTrainingImages
        else:
            self.numImages = len(self.imagePaths)            
            pass
        self.labelMap = loadLabelMap()        
        return
    
    def __len__(self):
        return self.numImages

    def __getitem__(self, index):
        if self.random:
            t = int(time.time() * 1000000)
            np.random.seed(((t & 0xff000000) >> 24) +
                           ((t & 0x00ff0000) >> 8) +
                           ((t & 0x0000ff00) << 8) +
                           ((t & 0x000000ff) << 24))
            index = np.random.randint(len(self.imagePaths))
        else:
            index = index % len(self.imagePaths)
            pass

        debug = -1
        if debug >= 0:
            index = debug
            print(index, self.imagePaths[index][1])
            pass
        image = cv2.imread(self.dataFolder + self.imagePaths[index][0])

        if self.test_batch:
          image, _ = augmentSample(self.options, image, background_colors=[], split=self.split)
          image = (image.astype(np.float32) / 255 - 0.5).transpose((2, 0, 1))
          return [image, self.imagePaths[index][0]]
        image_ori = image
        image_width, image_height = image.shape[1], image.shape[0]

        #def transformPoint(x, y, resize=False):
        #if resize:
        #return (int(round(float(x) * self.options.width / image_width)), int(round(float(y) * self.options.height / image_height)))
        #else:
        #return (int(round(float(x))), int(round(float(y))))            
        
        walls = []
        wall_types = []
        doors = []
        semantics = {}
        fpath = self.imagePaths[index][1]
        with open(self.dataFolder + self.imagePaths[index][1]) as info_file:
            line_index = 0
            for line in info_file:
                line = line.split('\t')
                label = line[4].strip()
                if label == 'point': continue
                if label == 'wall':
                    walls.append((convertToPoint(line[0], line[1]), convertToPoint(line[2], line[3])))
                    wall_types.append(int(line[5].strip()) - 1)
                elif label in ['door', 'window']:
                    doors.append((convertToPoint(line[0], line[1]), convertToPoint(line[2], line[3])))
                else:
                    if label not in semantics:
                        semantics[label] = []
                        pass
                    semantics[label].append((convertToPoint(line[0], line[1]), convertToPoint(line[2], line[3])))
                    pass
                continue
            pass
        #print([calcLineDirection(d) for d in doors])
        #print(doors)
        gap = 5
        #print(semantics)
        invalid_indices = {}
        for wall_index_1, (wall_1, wall_type_1) in enumerate(zip(walls, wall_types)):
            for wall_index_2, (wall_2, wall_type_2) in enumerate(zip(walls, wall_types)):
                if wall_type_1 == 0 and wall_type_2 == 1 and calcLineDirection(wall_1) == calcLineDirection(wall_2):
                    if min([pointDistance(wall_1[c_1], wall_2[c_2]) for c_1, c_2 in [(0, 0), (0, 1), (1, 0), (1, 1)]]) <= gap * 2:
                        walls[wall_index_1] = mergeLines(wall_1, wall_2)
                        invalid_indices[wall_index_2] = True
                        pass
                    pass
                continue
            continue
        walls = [wall for wall_index, wall in enumerate(walls) if wall_index not in invalid_indices]

        background_mask = measure.label(1 - drawWallMask(walls, image_width, image_height), background=0)
        wall_index = background_mask.min()
        background_colors = []
        if np.random.randint(2) == 0:
            for pixel in [(0, 0), (0, background_mask.shape[0] - 1), (background_mask.shape[1] - 1, 0), (background_mask.shape[1] - 1, background_mask.shape[0] - 1)]:
                index = background_mask[pixel[1]][pixel[0]]
                if index != wall_index:
                    background_colors = image[background_mask == index]
                    break
                continue
            pass
        
        #walls = connectWalls(walls, roomSegmentation, gap=gap)
        corners, success = lines2Corners(fpath, walls, gap=gap)
        if not success:
            #print('warning', index, self.imagePaths[index][1])
            pass


        if self.split == 'train':
            image, transformation = augmentSample(self.options, image, background_colors)
        else:
            image, transformation = augmentSample(self.options, image, background_colors, split=self.split)
            pass
        
        corners = [(transformPoint(transformation, corner[0]), corner[1]) for corner in corners]
        walls = [[transformPoint(transformation, wall[c]) for c in range(2)] for wall in walls]
        doors = [[transformPoint(transformation, door[c]) for c in range(2)] for door in doors]        
        for semantic, items in semantics.items():
            semantics[semantic] = [[transformPoint(transformation, item[c]) for c in range(2)] for item in items]
            continue

        width = self.options.width
        height = self.options.height
        #print('='*20, width, height)
        #print('='*20, NUM_ROOMS); exit(1)
        roomSegmentation = np.zeros((height, width), dtype=np.uint8)
        roomSegmentation2 = np.zeros((height, width, NUM_ROOMS+2), dtype=np.uint8)
        for line in walls:
            #cv2.line(roomSegmentation, line[0], line[1], color=NUM_ROOMS + 1 + calcLineDirection(line), thickness=gap)
            cv2.line(roomSegmentation, line[0], line[1], color=NUM_ROOMS + 1, thickness=gap)
            continue
        roomSegmentation2[:, :, NUM_ROOMS + 1][roomSegmentation == NUM_ROOMS + 1] = 1
        rooms = measure.label(roomSegmentation == 0, background=0)
        #print(rooms);exit(1)
        
        corner_gt = []
        for corner in corners:
            corner_gt.append((corner[0][0], corner[0][1], corner[1] + 1))
            continue

        #openingCornerMap = [[3, 1], [0, 2]]
        openingCornerMap = [[2, 0], [1, 3], [6, 4], [7, 5]]
        for opening in doors:
            direction = calcLineDirection(opening)
            for cornerIndex, corner in enumerate(opening):
                corner_gt.append((int(round(corner[0])), int(round(corner[1])), 1 + NUM_WALL_CORNERS + openingCornerMap[direction][cornerIndex])) # 14
                continue
            continue

        wallIndex = rooms.min()
        for pixel in [(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)]:
            backgroundIndex = rooms[pixel[1]][pixel[0]]
            if backgroundIndex != wallIndex:
                break
            continue
        iconSegmentation = np.zeros((height, width), dtype=np.uint8)
        iconSegmentation2 = np.zeros((height, width, NUM_ICONS + 2), dtype=np.uint8)
        for line in doors:
            cv2.line(iconSegmentation, line[0], line[1], color = self.labelMap['door'], thickness=gap - 1)
            continue
        iconSegmentation2[:, :, self.labelMap['door']][iconSegmentation == self.labelMap['door']] = 1
        roomLabelMap = {}
        for semantic, items in semantics.items():
            group, label = self.labelMap[semantic]
            #print('-' * 20)
            #print(self.labelMap); exit(1)
            for corners in items:
                if group == 'icons':
                    if label == 0:
                        continue
                    #print(label)
                    cv2.rectangle(iconSegmentation, (int(round(corners[0][0])), int(round(corners[0][1]))), (int(round(corners[1][0])), int(round(corners[1][1]))), color=label, thickness=-1)
                    #print(corners[0], corners[1])
                    x0, x1, y0, y1 = int(round(corners[0][0])), int(round(corners[1][0])), int(round(corners[0][1])), int(round(corners[1][1]))
                    x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
                    iconSegmentation2[:, :, label][y0: y1+1, x0: x1+1] = 1
                    corner_gt.append((corners[0][0], corners[0][1], 9 + NUM_WALL_CORNERS + 2)) # 18
                    corner_gt.append((corners[0][0], corners[1][1], 9 + NUM_WALL_CORNERS + 1))
                    corner_gt.append((corners[1][0], corners[0][1], 9 + NUM_WALL_CORNERS + 3))
                    corner_gt.append((corners[1][0], corners[1][1], 9 + NUM_WALL_CORNERS + 0))
                else:
                    roomIndex = rooms[(corners[0][1] + corners[1][1]) // 2][(corners[0][0] + corners[1][0]) // 2]
                    if roomIndex == wallIndex or roomIndex == backgroundIndex:
                        #if roomIndex == backgroundIndex:
                        #print('label on background', corners, semantic, index, self.imagePaths[index][1])
                        #pass
                        continue
                    if roomIndex in roomLabelMap:
                        #print('room has more than one labels', corners, label, roomLabelMap[roomIndex])
                        #exit(1)
                        continue
                        pass
                    roomLabelMap[roomIndex] = label
                    roomSegmentation[rooms == roomIndex] = label
                    roomSegmentation2[:, :, label][rooms == roomIndex] = 1
                    pass
                continue
            continue
        
        #print(roomLabelMap)
        if debug >= 0:
            cv2.imwrite('test/floorplan/rooms.png', drawSegmentationImage(rooms, blackIndex=backgroundIndex))
            exit(1)
            pass
        
        for roomIndex in range(rooms.min(), rooms.max() + 1):
            if roomIndex == wallIndex or roomIndex == backgroundIndex:
                continue
            if roomIndex not in roomLabelMap:
                roomSegmentation[rooms == roomIndex] = 10
                roomSegmentation2[:, :, 10][rooms == roomIndex] = 1
                #print('room has no label', roomIndex, rooms.max(), np.stack((rooms == roomIndex).nonzero(), axis=-1).mean(0)[::-1])
                #exit(1)
                pass
            continue

        cornerSegmentation = np.zeros((height, width, NUM_CORNERS), dtype=np.uint8)
        for corner in corner_gt:
            cornerSegmentation[min(max(corner[1], 0), height - 1), min(max(corner[0], 0), width - 1), corner[2] - 1] = 1
            continue
        roomSegmentation = roomSegmentation2
        iconSegmentation = iconSegmentation2
        #if True:
        if False:
            img_path = os.path.splitext(self.imagePaths[index][0])[0].split('/')[-1]
            cv2.imwrite('test/%s_image.png' % img_path, image)
            cv2.imwrite('test/%s_icon_segmentation.png' % img_path, drawSegmentationImage(iconSegmentation))
            cv2.imwrite('test/%s_room_segmentation.png' % img_path, drawSegmentationImage(roomSegmentation))
            cv2.imwrite('test/%s_corner_segmentation.png' % img_path, drawSegmentationImage(cornerSegmentation, blackIndex=0))
            print([(seg.min(), seg.max(), seg.shape) for seg in [cornerSegmentation, iconSegmentation, roomSegmentation]])
            exit(1)            
            pass

        # data augmentation
        #if False:
        if self.split == 'train' and self.augment:
          img_dtype = None
          if image.dtype != np.uint8:
            img_dtype = image.dtype

          # random rotations
          if np.random.randint(2) == 0:
            #ang = np.random.randint(360)
            ang = np.random.choice([90, -90])
            image = np.dstack([F.rotate(np2pil(image[:, :, i]), ang) for i in range(3)])
            cornerSegmentation = np.dstack([F.rotate(np2pil(cornerSegmentation[:, :, i]), ang) for i in range(NUM_CORNERS)])
            iconSegmentation = np.dstack([F.rotate(np2pil(iconSegmentation[:, :, i]), ang) for i in range(NUM_ICONS + 2)])
            roomSegmentation = np.dstack([F.rotate(np2pil(roomSegmentation[:, :, i]), ang) for i in range(NUM_ROOMS + 2)])

          # random h-flips
          if np.random.randint(2) == 0:
              image = np.dstack([F.hflip(np2pil(image[:, :, i])) for i in range(3)])
              cornerSegmentation = np.dstack([F.hflip(np2pil(cornerSegmentation[:, :, i])) for i in range(NUM_CORNERS)])
              iconSegmentation = np.dstack([F.hflip(np2pil(iconSegmentation[:, :, i])) for i in range(NUM_ICONS + 2)])
              roomSegmentation = np.dstack([F.hflip(np2pil(roomSegmentation[:, :, i])) for i in range(NUM_ROOMS + 2)])

          # random v-flips
          if np.random.randint(2) == 0:
              image = np.dstack([F.vflip(np2pil(image[:, :, i])) for i in range(3)])
              cornerSegmentation = np.dstack([F.vflip(np2pil(cornerSegmentation[:, :, i])) for i in range(NUM_CORNERS)])
              iconSegmentation = np.dstack([F.vflip(np2pil(iconSegmentation[:, :, i])) for i in range(NUM_ICONS + 2)])
              roomSegmentation = np.dstack([F.vflip(np2pil(roomSegmentation[:, :, i])) for i in range(NUM_ROOMS + 2)])

          # random crops
          if np.random.randint(2) == 0:
            i, j, h, w = transforms.RandomCrop.get_params(np2pil(cornerSegmentation), output_size=(height//2, width//2))
            image = np.dstack([F.resized_crop(np2pil(image[:, :, ii]), i, j, h, w, (height, width)) for ii in range(3)])
            cornerSegmentation = np.dstack([F.resized_crop(np2pil(cornerSegmentation[:, :, ii]), i, j, h, w, (height, width)) for ii in range(NUM_CORNERS)])
            iconSegmentation = np.dstack([F.resized_crop(np2pil(iconSegmentation[:, :, ii]), i, j, h, w, (height, width)) for ii in range(NUM_ICONS + 2)])
            roomSegmentation = np.dstack([F.resized_crop(np2pil(roomSegmentation[:, :, ii]), i, j, h, w, (height, width)) for ii in range(NUM_ROOMS + 2)])

          if img_dtype:
            image = image.astype(img_dtype)
          #print(cornerSegmentation.shape, iconSegmentation.shape, roomSegmentation.shape, image.shape)

        image = (image.astype(np.float32) / 255 - 0.5).transpose((2, 0, 1))
        kernel = np.zeros((3, 3), dtype=np.uint8)
        kernel[1] = 1
        kernel[:, 1] = 1
        cornerSegmentation = cv2.dilate(cornerSegmentation, kernel, iterations=5)

        #print(cornerSegmentation.shape, iconSegmentation.shape, roomSegmentation.shape, image.shape)
        #sample = [image, cornerSegmentation.astype(np.float32), iconSegmentation.astype(np.int64), roomSegmentation.astype(np.int64)]
        sample = [image, cornerSegmentation.astype(np.float32), iconSegmentation.astype(np.float32), roomSegmentation.astype(np.float32)]
        return sample
