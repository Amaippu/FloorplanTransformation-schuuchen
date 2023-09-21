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
import csv

def main(options):
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass
    
    model = Model(options)
    model.cuda()
    model.train()

    base = 'best'

    if options.restore == 1:#Either for training /testing / prediction you still need to load the weights
        print('restore from ' + options.checkpoint_dir + '/checkpoint_%s.pth' % (base))
        checkpoint = torch.load(options.checkpoint_dir + '/checkpoint_%s.pth' % (base))
        model.load_state_dict(checkpoint)
    
    if options.task == 'test':
        print('-'*20, 'test')
        dataset_test = FloorplanDataset(options, split='test_3', random=False)
        print('the number of test images', len(dataset_test))    
        testOneEpoch(options, model, dataset_test)
        exit(1)
    
    if options.task == 'predict':
        print('-'*20, 'predictions')
        predictForInputImages(options, model)
        exit(1)
        

    if options.task == 'test_batch':
        print('-'*20, 'test_batch')
        dataset_test = FloorplanDataset(options, split='test_batch', random=False, test_batch=True)
        print('the number of test_batch images', len(dataset_test))    
        testBatch_unet(options, model, dataset_test)
        exit(1)

    # dataset = FloorplanDataset(options, split='sb_train++', random=True, augment=options.augment)
    dataset = FloorplanDataset(options, split='train', random=True, augment=options.augment)
    print('the number of training images', len(dataset), ', batch size: ', options.batchSize, ' augment: ', options.augment)    
    
    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=True, num_workers=10, drop_last=True)
   
    optimizer = torch.optim.Adam(model.parameters(), lr = options.LR)
    if options.restore == 1 and os.path.exists(options.checkpoint_dir + '/optim_%s.pth' % (base)):
        print('optimizer using ' + options.checkpoint_dir + '/optim_%s.pth' % (base))
        optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/optim_%s.pth' % (base)))
        pass
    
    with open('loss_file.csv', 'w') as loss_file:
      writer = csv.writer(loss_file, delimiter=',', quotechar='"')
      best_loss = float('inf')
      for epoch in range(options.numEpochs):
          epoch_losses = []
          data_iterator = tqdm(dataloader, total=len(dataset) // options.batchSize + 1)
          for sampleIndex, sample in enumerate(data_iterator):
              optimizer.zero_grad()
            
              images, corner_gt, icon_gt, room_gt = sample[0].cuda(), sample[1].cuda(), sample[2].cuda(), sample[3].cuda()

              corner_pred, icon_pred, room_pred = model(images)
              #print([(v.shape, v.min(), v.max()) for v in [corner_pred, icon_pred, room_pred, corner_gt, icon_gt, room_gt]])
              #print([(v.shape, v.type()) for v in [corner_pred, icon_pred, room_pred, corner_gt, icon_gt, room_gt]]);exit(1)
              #print(corner_pred.shape, corner_gt.shape)
              corner_loss = NF.binary_cross_entropy_with_logits(corner_pred, corner_gt)
              #icon_loss = NF.cross_entropy(icon_pred.view(-1, NUM_ICONS + 2), icon_gt.view(-1))
              icon_loss = NF.binary_cross_entropy_with_logits(icon_pred, icon_gt)
              #room_loss = NF.cross_entropy(room_pred.view(-1, NUM_ROOMS + 2), room_gt.view(-1))            
              room_loss = NF.binary_cross_entropy_with_logits(room_pred, room_gt)            
              losses = [corner_loss, icon_loss, room_loss]
              loss = sum(losses)

              loss_values = [l.data.item() for l in losses]
              writer.writerow(loss_values)
              loss_file.flush()

              epoch_losses.append(loss_values)
              status = str(epoch + 1) + ' loss: '
              for l in loss_values:
                  status += '%0.5f '%l
                  continue
              data_iterator.set_description(status)
              loss.backward()
              optimizer.step()

              if sampleIndex % 500 == 0:
                  visualizeBatch(options, images.detach().cpu().numpy(), [('gt', {'corner': corner_gt.detach().cpu().numpy(), 'icon': icon_gt.detach().cpu().numpy(), 'room': room_gt.detach().cpu().numpy()}), ('pred', {'corner': corner_pred.max(-1)[1].detach().cpu().numpy(), 'icon': icon_pred.max(-1)[1].detach().cpu().numpy(), 'room': room_pred.max(-1)[1].detach().cpu().numpy()})])
                  if options.visualizeMode == 'debug':
                      exit(1)
                      pass
              continue
          print('loss', np.array(epoch_losses).mean(0))
          if (epoch + 1) % 100 == 0:
              epoch_key = f'epoch-{epoch + 1}'
              torch.save(model.state_dict(), options.checkpoint_dir + '/checkpoint_%s.pth' % (epoch_key))
              torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim_%s.pth' % (epoch_key))
              pass

          if loss.item() < best_loss:
              best_loss = loss.item()
              torch.save(model.state_dict(), options.checkpoint_dir + '/checkpoint_best.pth')
              torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim_best.pth')
              print('best loss: ', best_loss)
          #testOneEpoch(options, model, dataset_test)        
          continue
      return

def predictForInputImages(options, model):
    model.eval()

    directory_prediction: pathlib.Path = options.prediction_dir
    prediction_file: pathlib.Path = directory_prediction.joinpath('predict.txt')
    if (not directory_prediction.exists()):
        raise ValueError('The prediction directory does not exist')
    
    images: list[pathlib.Path] = list(directory_prediction.glob('*.png'))
    images.extend(directory_prediction.glob('*.jpg'))

    if(not len(images)):
        print(f'Nothing to do as there are no images inside the directory {directory_prediction}')
        exit(1)
    
    images.sort()

    f = open(prediction_file, 'w')
    for im_path in images:
        blank_annotation_file: pathlib.Path = im_path.parent.joinpath(f'{im_path.stem}.txt')
        annotation_file = open(blank_annotation_file, 'w')
        annotation_file.close()
        f.write(f'{im_path.stem}{im_path.suffix}\t{blank_annotation_file.stem}{blank_annotation_file.suffix}\n')
    f.close()

    options.dataFolder = directory_prediction
    options.batchSize = 1

    dataset = FloorplanDataset(options, split='predict', random=False)
    
    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=False, num_workers=1)
    
    epoch_losses = []    
    data_iterator = tqdm(dataloader, total=len(dataset) // options.batchSize + 1)
    for sampleIndex, sample in enumerate(data_iterator):

        images, corner_gt, icon_gt, room_gt = sample[0].cuda(), sample[1].cuda(), sample[2].cuda(), sample[3].cuda()
        
        corner_pred, icon_pred, room_pred = model(images)
        '''
        corner_loss = NF.binary_cross_entropy(corner_pred, corner_gt)
        icon_loss = NF.cross_entropy(icon_pred.view(-1, NUM_ICONS + 2), icon_gt.view(-1))
        room_loss = NF.cross_entropy(room_pred.view(-1, NUM_ROOMS + 2), room_gt.view(-1))            
        '''
        corner_loss = NF.binary_cross_entropy_with_logits(corner_pred, corner_gt)
        icon_loss = NF.binary_cross_entropy_with_logits(icon_pred, icon_gt)
        room_loss = NF.binary_cross_entropy_with_logits(room_pred, room_gt)            

        losses = [corner_loss, icon_loss, room_loss]
        
        loss = sum(losses)

        loss_values = [l.data.item() for l in losses]
        epoch_losses.append(loss_values)
        status = 'val loss: '
        for l in loss_values:
            status += '%0.5f '%l
            continue
        data_iterator.set_description(status)

        if sampleIndex % 500 == 0:
            #print(images.size()); exit(1)
            visualizeBatch(options, images.detach().cpu().numpy(), [('gt', {'corner': corner_gt.detach().cpu().numpy(), 'icon': icon_gt.detach().cpu().numpy(), 'room': room_gt.detach().cpu().numpy()}), ('pred', {'corner': corner_pred.max(-1)[1].detach().cpu().numpy(), 'icon': icon_pred.max(-1)[1].detach().cpu().numpy(), 'room': room_pred.max(-1)[1].detach().cpu().numpy()})])            
            for batchIndex in range(len(images)):
                corner_heatmaps = torch.sigmoid(corner_pred[batchIndex]).detach().cpu().numpy()
                '''
                icon_heatmaps = NF.softmax(icon_pred[batchIndex], dim=-1).detach().cpu().numpy()
                room_heatmaps = NF.softmax(room_pred[batchIndex], dim=-1).detach().cpu().numpy()
                '''
                icon_heatmaps = torch.sigmoid(icon_pred[batchIndex]).detach().cpu().numpy()
                room_heatmaps = torch.sigmoid(room_pred[batchIndex]).detach().cpu().numpy()

                reconstructFloorplan(corner_heatmaps[:, :, :NUM_WALL_CORNERS], corner_heatmaps[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 8], corner_heatmaps[:, :, -4:], icon_heatmaps, room_heatmaps, output_prefix=options.test_dir + '/' + str(batchIndex) + '_', densityImage=None, gt_dict=None, gt=False, gap=-1, distanceThreshold=-1, lengthThreshold=-1, debug_prefix='test', heatmapValueThresholdWall=None, heatmapValueThresholdDoor=None, heatmapValueThresholdIcon=None, enableAugmentation=True)
                continue
            if options.visualizeMode == 'debug':
                exit(1)
                pass
        continue
    print('validation loss', np.array(epoch_losses).mean(0))

    model.train()
    return

def testOneEpoch(options, model, dataset):
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=False, num_workers=1)
    
    epoch_losses = []    
    data_iterator = tqdm(dataloader, total=len(dataset) // options.batchSize + 1)
    for sampleIndex, sample in enumerate(data_iterator):

        images, corner_gt, icon_gt, room_gt = sample[0].cuda(), sample[1].cuda(), sample[2].cuda(), sample[3].cuda()
        
        corner_pred, icon_pred, room_pred = model(images)
        '''
        corner_loss = NF.binary_cross_entropy(corner_pred, corner_gt)
        icon_loss = NF.cross_entropy(icon_pred.view(-1, NUM_ICONS + 2), icon_gt.view(-1))
        room_loss = NF.cross_entropy(room_pred.view(-1, NUM_ROOMS + 2), room_gt.view(-1))            
        '''
        corner_loss = NF.binary_cross_entropy_with_logits(corner_pred, corner_gt)
        icon_loss = NF.binary_cross_entropy_with_logits(icon_pred, icon_gt)
        room_loss = NF.binary_cross_entropy_with_logits(room_pred, room_gt)            

        losses = [corner_loss, icon_loss, room_loss]
        
        loss = sum(losses)

        loss_values = [l.data.item() for l in losses]
        epoch_losses.append(loss_values)
        status = 'val loss: '
        for l in loss_values:
            status += '%0.5f '%l
            continue
        data_iterator.set_description(status)

        if sampleIndex % 500 == 0:
            #print(images.size()); exit(1)
            visualizeBatch(options, images.detach().cpu().numpy(), [('gt', {'corner': corner_gt.detach().cpu().numpy(), 'icon': icon_gt.detach().cpu().numpy(), 'room': room_gt.detach().cpu().numpy()}), ('pred', {'corner': corner_pred.max(-1)[1].detach().cpu().numpy(), 'icon': icon_pred.max(-1)[1].detach().cpu().numpy(), 'room': room_pred.max(-1)[1].detach().cpu().numpy()})])            
            for batchIndex in range(len(images)):
                corner_heatmaps = torch.sigmoid(corner_pred[batchIndex]).detach().cpu().numpy()
                '''
                icon_heatmaps = NF.softmax(icon_pred[batchIndex], dim=-1).detach().cpu().numpy()
                room_heatmaps = NF.softmax(room_pred[batchIndex], dim=-1).detach().cpu().numpy()
                '''
                icon_heatmaps = torch.sigmoid(icon_pred[batchIndex]).detach().cpu().numpy()
                room_heatmaps = torch.sigmoid(room_pred[batchIndex]).detach().cpu().numpy()

                reconstructFloorplan(corner_heatmaps[:, :, :NUM_WALL_CORNERS], corner_heatmaps[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 8], corner_heatmaps[:, :, -4:], icon_heatmaps, room_heatmaps, output_prefix=options.test_dir + '/' + str(batchIndex) + '_', densityImage=None, gt_dict=None, gt=False, gap=-1, distanceThreshold=-1, lengthThreshold=-1, debug_prefix='test', heatmapValueThresholdWall=None, heatmapValueThresholdDoor=None, heatmapValueThresholdIcon=None, enableAugmentation=True)
                continue
            if options.visualizeMode == 'debug':
                exit(1)
                pass
        continue
    print('validation loss', np.array(epoch_losses).mean(0))

    model.train()
    return

def testBatch_unet(options, model, dataset):
    model.eval()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)


    data_iterator = tqdm(dataloader, total=len(dataset))
    for sampleIndex, sample in enumerate(data_iterator):
        images, img_path = sample[0].cuda(), sample[1][0]
        img_name = os.path.splitext(img_path)[0].split('/')[-1]

        corner_pred, icon_pred, room_pred = model(images)

        room_unet = np.load('../../UNets/sb_preds_1000/%s.npy' % img_name).transpose(1,2,0)
        #corner_hg = np.load('../../pytorch-pose/val_res_0725_mse/%s.npy' % img_name)[0].transpose(1,2,0)

        visualizeBatch(options, images.detach().cpu().numpy(), [('pred', {'corner': corner_pred.max(-1)[1].detach().cpu().numpy(), 
                                                                          'icon': icon_pred.max(-1)[1].detach().cpu().numpy(), 
                                                                          'room': room_pred.max(-1)[1].detach().cpu().numpy()})], batch_img_name = img_name)         

        for batchIndex in range(len(images)):
          corner_heatmaps = torch.sigmoid(corner_pred[batchIndex]).detach().cpu().numpy()
          #corner_heatmaps = corner_hg
          #print(corner_heatmaps.shape); exit(1)
          icon_heatmaps = torch.sigmoid(icon_pred[batchIndex]).detach().cpu().numpy()
          #room_heatmaps = torch.sigmoid(room_pred[batchIndex]).detach().cpu().numpy()
          room_heatmaps = torch.nn.functional.softmax(torch.from_numpy(room_unet), dim=-1).detach().cpu().numpy()

          reconstructFloorplan(corner_heatmaps[:, :, :NUM_WALL_CORNERS], corner_heatmaps[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 8],
                               corner_heatmaps[:, :, -4:], icon_heatmaps, room_heatmaps, output_prefix=options.test_dir + '/' + img_name + '_',
                               densityImage=None, gt_dict=None, gt=False, gap=-1, distanceThreshold=-1, lengthThreshold=-1, debug_prefix='test',
                               heatmapValueThresholdWall=None, heatmapValueThresholdDoor=None, heatmapValueThresholdIcon=None, enableAugmentation=True)
          continue
        continue
    return


def testBatch(options, model, dataset):
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)

    data_iterator = tqdm(dataloader, total=len(dataset))
    for sampleIndex, sample in enumerate(data_iterator):
        images, img_path = sample[0].cuda(), sample[1][0]
        img_name = os.path.splitext(img_path)[0].split('/')[-1]

        corner_pred, icon_pred, room_pred = model(images)

        '''
        visualizeBatch(options, images.detach().cpu().numpy(), [('pred', {'corner': corner_pred.max(-1)[1].detach().cpu().numpy(), 
                                                                          'icon': icon_pred.max(-1)[1].detach().cpu().numpy(), 
                                                                          'room': room_pred.max(-1)[1].detach().cpu().numpy()})], batch_img_name = img_name)         
        '''

        for batchIndex in range(len(images)):
          corner_heatmaps = torch.sigmoid(corner_pred[batchIndex]).detach().cpu().numpy()
          icon_heatmaps = torch.sigmoid(icon_pred[batchIndex]).detach().cpu().numpy()
          room_heatmaps = torch.sigmoid(room_pred[batchIndex]).detach().cpu().numpy()

          reconstructFloorplan(corner_heatmaps[:, :, :NUM_WALL_CORNERS], corner_heatmaps[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 8], corner_heatmaps[:, :, -4:], 
                               icon_heatmaps, room_heatmaps, output_prefix=options.test_dir + '/' + img_name + '_', densityImage=None, gt_dict=None, gt=False, gap=-1, 
                               distanceThreshold=-1, lengthThreshold=-1, debug_prefix='test', heatmapValueThresholdWall=None, heatmapValueThresholdDoor=None, 
                               heatmapValueThresholdIcon=None, enableAugmentation=True)
          continue
        continue
    model.train()
    return

def visualizeBatch(options, images, dicts, indexOffset=0, prefix='', batch_img_name=None):
    #cornerColorMap = {'gt': np.array([255, 0, 0]), 'pred': np.array([0, 0, 255]), 'inp': np.array([0, 255, 0])}
    #pointColorMap = ColorPalette(20).getColorMap()
    images = ((images.transpose((0, 2, 3, 1)) + 0.5) * 255).astype(np.uint8)
    for batchIndex in range(len(images)):
        image = images[batchIndex].copy()
        if batch_img_name:
          filename = options.test_dir + '/%s_image.png' % batch_img_name
        else:
          filename = options.test_dir + '/' + str(indexOffset + batchIndex) + '_image.png'
        cv2.imwrite(filename, image)
        for name, result_dict in dicts:
            for info in ['corner', 'icon', 'room']:
                cv2.imwrite(filename.replace('image', info + '_' + name), drawSegmentationImage(result_dict[info][batchIndex], blackIndex=0, blackThreshold=0.5))
                continue
            continue
        continue
    return

if __name__ == '__main__':
    args = parse_args()
    
    args.keyname = 'floorplan'
    #args.keyname += '_' + args.dataset

    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass
    
    args.checkpoint_dir = 'checkpoint/' + args.keyname
    args.test_dir = 'test/' + args.keyname

    print('keyname=%s task=%s started'%(args.keyname, args.task))

    main(args)
