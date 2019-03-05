import torch
import torch.nn as nn
from torch.autograd import Variable


class CuboidSurface(nn.Module):
  def __init__(self, nSamples, normFactor='None'):
    self.nSamples = nSamples
    self.samplesPerFace = nSamples // 3
    self.normFactor = normFactor


  def sample_wt_module(self, dims):
    # dims is bs x 1 x 3
    area = self.cuboidAreaModule(dims) # bs x 1 x 1
    dimsInv = dims.pow(-1)
    dimsInvNorm = dimsInv.sum(2).repeat(1, 1, 3)
    normWeights = 3 * (dimsInv / dimsInvNorm)   #[32, 32, 3]

    widthWt, heightWt, depthWt = torch.chunk(normWeights, chunks=3, dim=2)
    widthWt = widthWt.repeat(1, self.samplesPerFace, 1) #[32, 1600(32x50), 1]
    heightWt = heightWt.repeat(1, self.samplesPerFace, 1)
    depthWt = depthWt.repeat(1, self.samplesPerFace, 1)

    sampleWt = torch.cat([widthWt, heightWt, depthWt], dim=1)  #[32, 4800, 1]
    # sampleWt = sampleWt.squeeze(2)
    # area = area.squeeze(2)
    sampleWt = sampleWt[:,:150,:]
    finalWt = (1/self.samplesPerFace) * (sampleWt * area)   # [32, 4800, 1] [32, 150, 1]
    return finalWt


  def sample(self,dims, coeff):
    dims_rep = dims.repeat(1,self.nSamples, 1)
    return dims_rep * coeff

  def cuboidAreaModule(self, dims):
    width, height, depth = torch.chunk(dims, chunks=3, dim=2)

    wh = width * height
    hd = height * depth
    wd = width * depth

    surfArea = 2*(wh + hd + wd)
    areaRep = surfArea.repeat(1, self.nSamples, 1)
    return areaRep

  def sample_points_cuboid(self, primShapes):
    # primPred B x 1 x 3
    # output B x nSamples x 3, B x nSamples x 1
    bs = primShapes.size(0)
    ns = self.nSamples
    nsp = self.samplesPerFace

    data_type = primShapes.data.type()
    coeffBernoulli = torch.bernoulli(torch.Tensor(bs, nsp, 3).fill_(0.5)).type(data_type)
    coeffBernoulli = 2 * coeffBernoulli - 1   # makes entries -1 and 1

    coeff_w = torch.Tensor(bs, nsp, 3).type(data_type).uniform_(-1, 1)
    coeff_w[:, :, 0].copy_(coeffBernoulli[:,:,0].clone())

    coeff_h = torch.Tensor(bs, nsp, 3).type(data_type).uniform_(-1, 1)
    coeff_h[:, :, 1].copy_(coeffBernoulli[:,:,1].clone())

    coeff_d = torch.Tensor(bs, nsp, 3).type(data_type).uniform_(-1, 1)
    coeff_d[:, :, 2].copy_(coeffBernoulli[:,:,2].clone())

    coeff = torch.cat([coeff_w, coeff_h, coeff_d], dim=1)
    coeff = Variable(coeff)
    samples = self.sample(primShapes, coeff)  #[32, 150, 3]
    importance_weights = self.sample_wt_module(primShapes)
    return samples, importance_weights

def test_cuboid_surface():
  N = 1
  P = 1

  cuboidSampler = CuboidSurface(18)
  primShapes  = torch.Tensor(N, P, 3).fill_(0.5)

  samples, importance_weights = cuboidSampler.sample_points_cuboid(primShapes)

  pdb.set_trace()


if __name__ == "__main__":
  test_cuboid_surface()
