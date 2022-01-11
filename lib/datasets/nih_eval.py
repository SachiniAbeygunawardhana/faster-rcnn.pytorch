# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
from csv import reader


def parse_rec(filename, imagename):
  """
  filename: path to annotations csv file
  imagename : image index
   """

  objects = []

  with open(filename, 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
      # Iterate over each row after the header in the csv
      for row in csv_reader:
        if row[0] == imagename:  # if row [image Index] == imagename
          obj_struct = {}
          obj_struct['label'] = row[1]
          x1 = float(row[2])
          y1 = float(row[3])
          x2 = x1 + float(row[4])
          y2 = y1 + float(row[5])
          obj_struct['bbox'] = [x1, y1, x2, y2]
          objects.append(obj_struct)

  return objects


def nih_ap(rec, prec):
  """
  Compute NIH AP given precision and recall.

  """
  # correct AP (average precision) calculation
  # first append sentinel values at the end
  mrec = np.concatenate(([0.], rec, [1.]))
  mpre = np.concatenate(([0.], prec, [0.]))

  # compute the precision envelope
  for i in range(mpre.size - 1, 0, -1):
    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

  # to calculate area under PR curve, look for points
  # where X axis (recall) changes value
  i = np.where(mrec[1:] != mrec[:-1])[0]

  # and sum (\Delta recall) * prec
  ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

  return ap


def nih_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
  """rec, prec, ap = voc_eval(detpath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              [use_07_metric])

  Top level function that does the NIH evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
      (the result file for each disease)
  annopath: Path to annotations
     should be the csv annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  (test.txt)
  classname: Disease Category name (e.g. Atelectasis)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)

  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]  # list of test images

  if not os.path.isfile(cachefile):
    # load annotations
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(annopath, imagename + '.png')  # return for each image
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
      pickle.dump(recs, f)

  # if there is a cachefile
  else:
    # load
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')  # okay

  # extract gt objects for this class
  class_recs = {}
  npos = 0

  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['label'] == classname]
    bbox = np.array([x['bbox'] for x in R])
    det = [False] * len(R)
    npos = npos + len(R)
    class_recs[imagename] = {'bbox': bbox,
                             'det': det}

  # read detections
  detfile = detpath.format(classname)
  with open(detfile, 'r') as f:
    lines = f.readlines()  # okay

  splitlines = [x.strip().split(' ') for x in lines]  # each detection for the class
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])  # bounding boxes

  nd = len(image_ids)
  tp = np.zeros(nd)  # true positives
  fp = np.zeros(nd)  # false positives

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['det'][jmax]:
          tp[d] = 1.
          R['det'][jmax] = 1
        else:
          fp[d] = 1.
      else:
        fp[d] = 1.

  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = nih_ap(rec, prec)

  return rec, prec, ap