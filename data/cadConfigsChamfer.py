from scipy.io import loadmat
import glob
import os
import pdb
import torch
import modules.primitives as primitives
import numpy as np
from torch.autograd import Variable

## Dataloader for Volumetric Primitives

class SimpleCadData(object):
  def __init__(self, params):
    self.gridSize = params.gridSize
    self.modelsDir = params.modelsDataDir
    self.modelIter = params.modelIter
    self.batchSize = params.batchSize
    self.modelSize = params.gridSize
    self.nSamplePoints = params.nSamplePoints
    self.gridBound = params.gridBound

    self.iter = 0
    self.startModelIndex = 0
    self.modelNames = []

    for filename in glob.iglob(self.modelsDir + '/*.mat'):
      self.modelNames.append(filename)


    ## Limit to 200 chairs
    # self.modelNames = self.modelNames[0:200]

    self.loadedVoxels = torch.Tensor(self.batchSize, 1, self.modelSize, self.modelSize, self.modelSize).fill_(0)
    self.loadedTsdfs = torch.Tensor(self.batchSize, 1, self.modelSize, self.modelSize, self.modelSize).fill_(0)
    self.loadedCPs = torch.Tensor(self.batchSize, 1, self.modelSize, self.modelSize, self.modelSize, 3).fill_(0)
    self.loadedSurfaceSamples = {}

    gridMin = -params.gridBound + params.gridBound / params.gridSize
    gridMax = params.gridBound - params.gridBound / params.gridSize

    meshGridInit = primitives.meshGrid([gridMin, gridMin, gridMin], [gridMax, gridMax, gridMax],
                                       [params.gridSize, params.gridSize, params.gridSize])
    meshGrid = meshGridInit.repeat(params.batchSize, 1, 1, 1, 1)
    self.gridPoints = meshGrid.view(params.batchSize, params.gridSize**3, 3).clone()
    self.shape_mats = dict()
    self.load_all_mats()
    self.load_torch_tensors()
    self.outSampleTsfds = torch.Tensor(self.batchSize, self.nSamplePoints).fill_(0).cuda()

  def load_all_mats(self):
    for ix in range(len(self.modelNames)):
      model_name = self.modelNames[ix]
      self.shape_mats[model_name] = loadmat(model_name)

  def load_torch_tensors(self):
    self.all_tsdfs = []
    self.all_volumes = []
    self.all_closetPoints = []
    self.all_surfaceSamples = []

    for model_file, shape in self.shape_mats.items():
      self.all_tsdfs.append(torch.from_numpy(shape['tsdf']).float())
      self.all_volumes.append(torch.from_numpy(shape['Volume']).float())
      self.all_closetPoints.append(torch.from_numpy(shape['closestPoints']).float())
      self.all_surfaceSamples.append(torch.from_numpy(shape['surfaceSamples']).float())

    self.all_tsdfs = torch.stack(self.all_tsdfs).cuda()
    self.all_volumes = torch.stack(self.all_volumes).unsqueeze(1).cuda()
    self.all_closetPoints = torch.stack(self.all_closetPoints).unsqueeze(1).cuda()
    self.all_surfaceSamples = torch.stack(self.all_surfaceSamples).cuda()

  def reloadShapes(self): #
    ids = []
    for ix in range(self.batchSize):
      self.startModelIndex = np.random.randint(0, len(self.modelNames))
      shape_dict = self.shape_mats[self.modelNames[self.startModelIndex]]
      ids.append(self.startModelIndex)

    ids = torch.LongTensor(ids).cuda()
    # pdb.set_trace()
    self.loadedVoxels = self.all_volumes[ids]
    self.loadedTsdfs = self.all_tsdfs[ids]
    self.loadedCPs = self.all_closetPoints[ids]
    self.loadedSurfaceSamples = self.all_surfaceSamples[ids]
    self.loadedShapes = self.loadedVoxels

  def forward(self):
    if (self.iter % self.modelIter == 0):
      self.reloadShapes()
    self.iter = self.iter + 1

    outSamplePoints = []
    # pdb.set_trace()
    for b in range(self.batchSize):
      nPointsTot = self.loadedSurfaceSamples[b].size(0)
      sample_ids = torch.LongTensor(np.random.randint(0, nPointsTot, self.nSamplePoints)).cuda() #0~nPointsTot smaplepoint 
      outSamplePoints.append(self.loadedSurfaceSamples[b][sample_ids])
    outSamplePoints = torch.stack(outSamplePoints)
    output = [self.loadedShapes, self.outSampleTsfds, outSamplePoints, self.loadedCPs]    #closet point
    return output

  def forwardTest(self):
    if self.iter % self.modelIter == 0:
      self.reloadShapes()
    self.iter  = self.iter + 1
    outTsfds = self.loadedTsdfs.view(self.batchSize, self.gridSize**3)
    outPoints = self.gridPoints.clone()
    return self.loadedShapes, outTsfds, outPoints


  def chamfer_forward(self, queryPoints):
    #query points is B x nQ x 3
    neighbourIds = self.pointClosestCellIndex(queryPoints).data
    # queryDiffs = queryPoints.clone()
    # queryDiffs.data.fill_(0)
    loadedCPs = Variable(self.loadedCPs.cuda())
    queryDiffs = []
    cps = []
    batch_size = queryPoints.size(0)

    for b in range(queryPoints.size(0)):
      inds = neighbourIds[b]
      inds = self.gridSize*self.gridSize*inds[:,0]  + self.gridSize*inds[:,1] + inds[:,2]
      cp = loadedCPs[b,0].view(-1,3)
      cp = cp[inds]
      voxels = Variable(self.loadedVoxels[b][0].view(-1)) #TODO
      voxels = voxels[inds]
      diff = (cp - queryPoints[b].view(-1,3)).pow(2).sum(1)
      queryDiffs.append((-voxels+1) * diff)
    queryDiffs = torch.stack(queryDiffs)

    outDists = queryDiffs
    self.queryDiffs = queryDiffs
    return outDists

  def pointClosestCellIndex(self, points):
    gridMin = -self.gridBound + self.gridBound / self.gridSize
    gridMax = self.gridBound - self.gridBound / self.gridSize
    inds = (points - gridMin) * self.gridSize / (2 * self.gridBound)
    inds = torch.round(torch.clamp(inds, min=0, max=self.gridSize-1)).long()
    return inds