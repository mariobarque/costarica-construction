import torch
import torch.nn as nn


def evaluateSigmoid(W, X):
  NetWeights = torch.mm(X, W);
  sigmoid = nn.Sigmoid();
  Activation = sigmoid(NetWeights)
  return Activation;

def getDeltaWeightsForClass(W, Samples, alpha, target):
  ActivationSamples = evaluateSigmoid(W, Samples)
  Error = (target - ActivationSamples)
  DeltaWeights = Samples * Error
  DeltaWeightsWithAlpha = alpha * DeltaWeights
  error = torch.sum(torch.abs(Error))
  return DeltaWeightsWithAlpha, error


def trainLogisticRegression(samplesClass1, samplesClass2, numIter, alpha):
  samplesClass1 = torch.from_numpy(samplesClass1).float()
  samplesClass2 = torch.from_numpy(samplesClass2).float()

  dimensionsData = samplesClass1.size()
  W = torch.ones(dimensionsData[1],1)

  errorPerEpoch = []
  for i in range(0, numIter):
    (DeltaWeights1, error1) = getDeltaWeightsForClass(W, samplesClass1, alpha, target = 0)
    (DeltaWeights2, error2) = getDeltaWeightsForClass(W, samplesClass2, alpha, target = 1)
    error = error1 + error2
    errorPerEpoch.append(float(error.numpy()))
    WdeltaTotal = torch.sum(DeltaWeights1, dim = 0) + torch.sum(DeltaWeights2, dim = 0)
    W = W + WdeltaTotal.view(len(W), 1)

  return W.numpy().T[0], errorPerEpoch


def train(df, variable):
    classZero = df.loc[df[variable] == 0].drop(variable, 1)
    classOne = df.loc[df[variable] == 1].drop(variable, 1)

    numIterations = 10;
    alpha = 0.1;

    weights, errors = trainLogisticRegression(classZero.values, classOne.values, numIterations, alpha)
