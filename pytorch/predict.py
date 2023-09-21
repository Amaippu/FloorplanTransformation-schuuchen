import pathlib
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as NF

from tqdm import tqdm
import numpy as np
import os
import cv2

from utils import *
from options import parse_args

from models.model import Model

from datasets.floorplan_dataset import FloorplanDataset
from IP import reconstructFloorplan
from train import visualizeBatch


#USAGE
# python predict.py --prediction_dir ./prediction_inputs/ --testdir ./prediction_inputs/predictions

def predictForInputImages(options, model):
    model.eval()

    directory_prediction: pathlib.Path = options.prediction_dir
    prediction_file: pathlib.Path = directory_prediction.joinpath('predict.txt')
    if (not directory_prediction.exists()):
        raise ValueError('The prediction directory does not exist')

    images: list[pathlib.Path] = list(directory_prediction.glob('*.png'))
    images.extend(directory_prediction.glob('*.jpg'))

    if (not len(images)):
        print(
            f'Nothing to do as there are no images inside the directory {directory_prediction}')
        exit(1)

    images.sort()

    f = open(prediction_file, 'w')
    for im_path in images:
        blank_annotation_file: pathlib.Path = im_path.parent.joinpath(
            f'{im_path.stem}.txt')
        annotation_file = open(blank_annotation_file, 'w')
        annotation_file.close()
        f.write(
            f'{im_path.stem}{im_path.suffix}\t{blank_annotation_file.stem}{blank_annotation_file.suffix}\n')
    f.close()

    options.dataFolder = directory_prediction
    options.batchSize = 1

    dataset = FloorplanDataset(options, split='predict', random=False)

    dataloader = DataLoader(
        dataset, batch_size=options.batchSize, shuffle=False, num_workers=1)

    data_iterator = tqdm(dataloader, total=len(
        dataset) // options.batchSize + 1)
    for sampleIndex, sample in enumerate(data_iterator):
        wallInformationFile: pathlib.Path = pathlib.Path(options.test_dir).joinpath(f'{sampleIndex}_0_floorplan.txt')
        wallInformationDrawing: pathlib.Path = pathlib.Path(options.test_dir).joinpath(f'{sampleIndex}_0_image_floorplan_drawing.jpg')
        floorplanOriginalImage: pathlib.Path = pathlib.Path(options.test_dir).joinpath(f'{sampleIndex}_0_image.png')
        wallInformationCombinedDrawing: pathlib.Path = pathlib.Path(options.test_dir).joinpath(f'{sampleIndex}_0_image_wall_result_combined.jpg')


        images, corner_gt, icon_gt, room_gt = sample[0].cuda(
        ), sample[1].cuda(), sample[2].cuda(), sample[3].cuda()

        corner_pred, icon_pred, room_pred = model(images)
        if(not wallInformationFile.exists()):
            visualizeBatch(
                options,
                images.detach().cpu().numpy(),
                [
                    (
                        'gt',
                        {
                            'corner': corner_gt.detach().cpu().numpy(),
                            'icon': icon_gt.detach().cpu().numpy(),
                            'room': room_gt.detach().cpu().numpy()
                        }
                    ),
                    (
                        'pred',
                        {
                            'corner': corner_pred.max(-1)[1].detach().cpu().numpy(),
                            'icon': icon_pred.max(-1)[1].detach().cpu().numpy(),
                            'room': room_pred.max(-1)[1].detach().cpu().numpy()
                        }
                    )
                ], 
                0, 
                prefix=f'{sampleIndex}_'
                )
            
            for batchIndex in range(len(images)):
                corner_heatmaps = torch.sigmoid(corner_pred[batchIndex]).detach().cpu().numpy()
                icon_heatmaps = torch.sigmoid(icon_pred[batchIndex]).detach().cpu().numpy()
                room_heatmaps = torch.sigmoid(room_pred[batchIndex]).detach().cpu().numpy()

                reconstructFloorplan(
                    corner_heatmaps[:, :, :NUM_WALL_CORNERS], 
                    corner_heatmaps[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 8], 
                    corner_heatmaps[:, :, -4:], 
                    icon_heatmaps, 
                    room_heatmaps, 
                    output_prefix=options.test_dir + f'/{sampleIndex}_{batchIndex}_', 
                    densityImage=None, 
                    gt_dict=None, 
                    gt=False, 
                    gap=-1, 
                    distanceThreshold=options.distanceThreshold2D, 
                    lengthThreshold=options.lengthThreshold2D, 
                    debug_prefix=f'test-{sampleIndex}_{batchIndex}_', 
                    heatmapValueThresholdWall=options.heatmapThreshold, 
                    heatmapValueThresholdDoor=options.heatmapThreshold, 
                    heatmapValueThresholdIcon=options.heatmapThreshold, 
                    enableAugmentation=True)
                
        if(wallInformationFile.exists()):
            wall_start_index = 2
            f = open(f'{wallInformationFile}', 'r')
            lines = f.read().split('\n')
            w, h = [int(val.strip()) for val in lines[0].split('\t')]
            total_walls = [int(val.strip()) for val in lines[1].split('\t')][0]

            floorplan_image = cv2.imread(f'{floorplanOriginalImage}')
            drawing_image = np.zeros((h, w, 3), np.uint8)
            drawing_image[:,:,:,] = 255

            for i in range(total_walls):
                index = i + wall_start_index
                x1, y1, x2, y2, _, _ = [int(float(val.strip())) for val in lines[index].split('\t')]
                cv2.circle(drawing_image, (x1, y1), 1, (0, 0, 255), -1 )
                cv2.circle(drawing_image, (x2, y2), 1, (0, 0, 255), -1 )
                cv2.line(drawing_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

            f.close()
            cv2.imwrite(f'{wallInformationCombinedDrawing}', np.concatenate((floorplan_image, drawing_image), axis=1))
            cv2.imwrite(f'{wallInformationDrawing}', drawing_image)


def main(options):
    base = 'best'
    test_dir = pathlib.Path(options.test_dir)
    test_dir.mkdir(exist_ok=True, parents=True)

    model = Model(options)
    model.cuda()
    model.train()

    checkpoint = torch.load(options.checkpoint_dir +
                            '/checkpoint_%s.pth' % (base))
    model.load_state_dict(checkpoint)

    predictForInputImages(options, model)
    exit(1)


if __name__ == '__main__':
    args = parse_args()

    args.keyname = 'floorplan'

    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass

    args.checkpoint_dir = 'checkpoint/' + args.keyname
    args.test_dir = f'{pathlib.Path(args.test_dir).joinpath(args.keyname)}'

    print('keyname=%s task=%s started' % (args.keyname, args.task))
    print(args.test_dir)
    main(args)
