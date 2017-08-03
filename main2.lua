require 'math'
require 'cutorch'
require 'nn'
require 'cunn'
rnnlib = require 'rnnlib'

tablex  = require 'pl.tablex'
stringx = require 'pl.stringx'
tnt     = require 'torchnet'
optim   = require 'optim'
data    = require 'data'
utils   = require 'utils'

torch.setheaptracking(true)

cmd = torch.CmdLine('-', '-')
cmd:option('-seed', 1111, 'Seed for the random generator')

cmd:option('-isz',     512, 'Dimension of input word vectors')
cmd:option('-nhid',    512, 'Number of hidden variables per layer')
cmd:option('-nlayer',  1,   'Number of layers')
cmd:option('-dropout', 0.5, 'Dropout probability')

cmd:option('-lr',            0.1,  'Learning rate')
cmd:option('-epsilon',       1e-5, 'Epsilon for Adagrad')
cmd:option('-initrange',     0.1,  'Init range')
cmd:option('-maxepoch',      30,   'Number of epochs')
cmd:option('-bptt',          20,   'Number of backprop through time steps')
cmd:option('-clip',          0.25, 'Threshold for gradient clipping')
cmd:option('-batchsize',     1,   'Batch size')
cmd:option('-testbatchsize', 16,   'Batch size for test')

cmd:option('-data',      'dataset/ptb', 'Path to the dataset directory')
cmd:option('-outdir',    '', 'Path to the output directory')
cmd:option('-threshold', 0,  'Threshold for <unk> words')

cmd:option('-cutoff', '2000',   'Cutoff for AdaptiveSoftMax')

cmd:option('-embed',  'BoW','topic embedding (BoW | W2V)')
cmd:option('-window',  10,   'topic window size')
cmd:option('-ntopic',  30,  'topic size')
cmd:option('-latent', 128,   'latent space size')
cmd:option('-alpha',   0.5, 'weights for LM, 1-alpha for Topic')
cmd:option('-beta',    0.5, 'weights between KL and Expection')
cmd:option('-device',  2,   'gpu usage index')

cmd:option('-usecudnn', false, '')
print(cmd)

config = cmd:parse(arg)

cutorch.setDevice(config.device)
torch.manualSeed(config.seed)
cutorch.manualSeed(config.seed)

--------------------------------------------------------------------------------
-- LOAD DATA
--------------------------------------------------------------------------------

trainfilename = paths.concat(config.data, 'train.txt')
validfilename = paths.concat(config.data, 'valid.txt')
testfilename  = paths.concat(config.data, 'test.txt')


dic = data.makedictionary(trainfilename)

collectgarbage()

ntoken = #dic.idx2word
bsz    = config.batchsize
tbsz   = config.testbatchsize
bptt   = config.bptt
topicvocab_size = #dic.tidx2word

train1, train2 = data.loadfile(trainfilename, dic)
valid1, valid2 = data.loadfile(validfilename, dic)
test1,  test2  = data.loadfile(testfilename,  dic)
batch = {
train = train1, 
valid = valid1, 
test  = test1,
}
batchtopic = {
train = train2,                                                             
valid = valid2,                                                             
test  = test2,
}
collectgarbage()

train = tnt.DatasetIterator(tnt.TopicRNNDataset(batch.train, batchtopic.train, dic, bsz,  bptt, config.window))
valid = tnt.DatasetIterator(tnt.TopicRNNDataset(batch.valid, batchtopic.valid, dic, tbsz, bptt, config.window))
test  = tnt.DatasetIterator(tnt.TopicRNNDataset(batch.test , batchtopic.test,  dic, tbsz, bptt, config.window))


VAE=require './vae/VAE.lua'
require './vae/KLDPenalty.lua'
require './vae/Sampler.lua'
require './vae/ACOSLayer.lua'
require './vae/weightedSumCriterion.lua'
dofile './Topic/cellTopicRNN.lua'
dofile './Topic/TopicRNN.lua'
dofile './Topic/topic_recurrentnetwork.lua'

topic_encoder = VAE.get_encoder(topicvocab_size, 256, config.latent)
topic_decoder = VAE.get_decoder(config.ntopic, topicvocab_size, 30)
KLD = nn.KLDPenalty(1, config.batchsize)

train = tnt.TopicRNNDataset(batch.train, batchtopic.train, dic, bsz,  bptt, config.window)
valid = tnt.TopicRNNDataset(batch.valid, batchtopic.valid, dic, tbsz, bptt, config.window)
topicmodel = nn.Sequential():add(topic_encoder):add(KLD):add(nn.Sampler()):add(nn.Linear(config.latent, config.ntopic)):add(nn.SoftMax()):add(topic_decoder):cuda()                                          
topiccrit = nn.BCECriterion():cuda()
--topiccrit = nn.weightedSumCriterion():cuda()
W, grad   = topicmodel:getParameters()

inputSize = topicmodel:get(6):get(1).weight:size(2)
outputSize= topicmodel:get(6):get(1).weight:size(1)
m1= nn.Sequential():add(nn.ConcatTable():add(nn.Sequential():add(nn.Identity()):add(nn.Replicate(inputSize, 2)):add(nn.Reshape(inputSize*inputSize, outputSize))):add(nn.Sequential():add(nn.Identity()):add(nn.Replicate(inputSize, 1)):add(nn.Reshape(inputSize*inputSize, outputSize))) ):add(nn.CosineDistance()):add(nn.ACOSLayer()) 
m2 = nn.ConcatTable():add(nn.Identity()):add(nn.Sequential():add(nn.Mean()):add(nn.Replicate(inputSize*inputSize)):add(nn.Reshape(inputSize*inputSize))):add(nn.Mean()) 
m3 = nn.ConcatTable():add(nn.SelectTable(-1)):add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CSubTable()):add(nn.Power(2)):add(nn.Mean()))
model2 = nn.Sequential():add(m1):add(m2):add(m3):cuda()
crit2 = nn.ParallelCriterion():add(nn.AbsCriterion(), -0.1):add(nn.AbsCriterion(), 0.1):cuda()
dofile("train_topic.lua")
print(topicmodel)
print(model2)
trainTopic(topicmodel, topiccrit, model2, crit2, W, grad, train, valid)

embed = topicmodel:get(6):get(1).weight
visualtopic(embed, dic)
