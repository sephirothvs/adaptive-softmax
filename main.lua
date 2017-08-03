-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require 'math'
require 'cutorch'
require 'nn'
require 'cunn'
rnnlib = require 'rnnlib'

local tablex  = require 'pl.tablex'
local stringx = require 'pl.stringx'
local tnt     = require 'torchnet'
local optim   = require 'optim'
local data    = require 'data'
local utils   = require 'utils'

torch.setheaptracking(true)

local cmd = torch.CmdLine('-', '-')
cmd:option('-seed', 1111, 'Seed for the random generator')

cmd:option('-isz',     512, 'Dimension of input word vectors')
cmd:option('-nhid',    512, 'Number of hidden variables per layer')
cmd:option('-nlayer',  1,   'Number of layers')
cmd:option('-dropout', 0.5, 'Dropout probability')

cmd:option('-lr',            0.1,  'Learning rate')
cmd:option('-epsilon',       1e-5, 'Epsilon for Adagrad')
cmd:option('-initrange',     0.1,  'Init range')
cmd:option('-maxepoch',      10,   'Number of epochs')
cmd:option('-bptt',          20,   'Number of backprop through time steps')
cmd:option('-clip',          0.25, 'Threshold for gradient clipping')
cmd:option('-batchsize',     16,   'Batch size')
cmd:option('-testbatchsize', 16,   'Batch size for test')

cmd:option('-data',      'dataset/ptb', 'Path to the dataset directory')
cmd:option('-outdir',    '', 'Path to the output directory')
cmd:option('-threshold', 0,  'Threshold for <unk> words')

cmd:option('-cutoff', '2000',   'Cutoff for AdaptiveSoftMax')

cmd:option('-embed',  'BoW','topic embedding (BoW | W2V)')
cmd:option('-window',  5,   'topic window size')
cmd:option('-ntopic',  30,  'topic size')
cmd:option('-alpha',   0.5, 'weights for LM, 1-alpha for Topic')
cmd:option('-beta',    1, 'weights between KL and Expection')
cmd:option('-device',  2,   'gpu usage index')

cmd:option('-usecudnn', false, '')
print(cmd)

local config = cmd:parse(arg)

cutorch.setDevice(config.device)
torch.manualSeed(config.seed)
cutorch.manualSeed(config.seed)

--------------------------------------------------------------------------------
-- SET LOGGER
--------------------------------------------------------------------------------

local logfile
if config.outdir ~= '' then
   paths.mkdir(config.outdir)
   logfile = io.open(paths.concat(config.outdir, 'log.txt'))
   print('Log file: ' .. paths.concat(config.outdir, 'log.txt'))
end

--------------------------------------------------------------------------------
-- LOAD DATA
--------------------------------------------------------------------------------

local trainfilename = paths.concat(config.data, 'train.txt')
local validfilename = paths.concat(config.data, 'valid.txt')
local testfilename  = paths.concat(config.data, 'test.txt')

local dic
dic = data.makedictionary(trainfilename)
--[[
if paths.filep(paths.concat(config.data, 'dic.txt')) then
   dic = data.loaddictionary(paths.concat(config.data, 'dic.txt'))
else
   dic = data.makedictionary(trainfilename)
   data.savedictionary(dic, paths.concat(config.data, 'dic.txt'))
end
--]]
collectgarbage()

local w2v
if config.embed == 'W2V' then
    if paths.filep(paths.concat(config.data, 'w2v.t7')) then
        w2v = torch.load(paths.concat(config.data, 'w2v.t7'))
    else
        w2vutils = require('utils/w2vutils.lua')
        local vocabsize = dic.idx2freq:size(1)
        w2v = torch.zeros(vocabsize, 300)
        for i = 1,vocabsize do
            w2v[i] = w2vutils:word2vec(dic.idx2word[i])
        end
        torch.save(paths.concat(config.data, 'w2v.t7'), w2v)
    end
end
collectgarbage()

local ntoken = #dic.idx2word
local bsz    = config.batchsize
local tbsz   = config.testbatchsize
local bptt   = config.bptt
local topicvocab_size = #dic.tidx2word

local train1, train2 = data.loadfile(trainfilename, dic)
local valid1, valid2 = data.loadfile(validfilename, dic)
local test1,  test2  = data.loadfile(testfilename,  dic)
local batch = {
    train = train1, 
    valid = valid1, 
    test  = test1,
}
local batchtopic = {
    train = train2,                                                             
    valid = valid2,                                                             
    test  = test2,
}

collectgarbage()

local train = tnt.DatasetIterator(tnt.TopicRNNDataset(batch.train, batchtopic.train, dic, bsz,  bptt, config.window))
local valid = tnt.DatasetIterator(tnt.TopicRNNDataset(batch.valid, batchtopic.valid, dic, tbsz, bptt, config.window))
local test  = tnt.DatasetIterator(tnt.TopicRNNDataset(batch.test , batchtopic.test,  dic, tbsz, bptt, config.window))

--------------------------------------------------------------------------------
-- MAKE MODEL
--------------------------------------------------------------------------------
VAE=require './vae/VAE.lua'
require './vae/KLDPenalty.lua'
require './vae/Sampler.lua'
dofile './Topic/cellTopicRNN.lua'
dofile './Topic/TopicRNN.lua'
dofile './Topic/topic_recurrentnetwork.lua'

local initrange = config.initrange or 0.1

local lut = nn.LookupTable(ntoken, config.isz)
lut.weight:uniform(-initrange, initrange)
lut:cuda()

local w2v_embed=nil
if config.embed == 'W2V' then
    w2v_embed = nn.LookupTable(ntoken, 300)
    w2v_embed.weight:copy(w2v)
    w2v_embed:cuda()
end

local rnn, hids = nn.TopicLSTM{
    inputsize = config.isz,
    hidsize   = config.nhid,
    nlayer    = config.nlayer,
    ntopic    = config.ntopic,
    usecudnn  = false,
}

local topic_encoder = VAE.get_encoder(topicvocab_size, 500, config.ntopic)
local topic_decoder = VAE.get_decoder(config.ntopic, topicvocab_size)
local KLD = nn.KLDPenalty((1-config.alpha)*config.beta, config.batchsize)


dofile 'train_topic.lua'


local cutoff = tablex.map(tonumber, stringx.split(config.cutoff, ','))
table.insert(cutoff, ntoken)

local decoder = nn.AdaptiveSoftMax(config.nhid, cutoff)
local crit1 = nn.AdaptiveLoss(cutoff)
local crit2 = nn.BCECriterion()
require './utils/TopicCriterion.lua'
local crit  = nn.TopicCriterion()
crit:add(crit1, (config.alpha)*2)   -- Language Model Loss
    :add(crit2, (1-config.alpha)*2) -- Topic Model Loss

onsample = function(state)
   state.inputlut = state.sample.input:cuda()
   lut:forward(state.inputlut)
   state.sample.input = {state.hid, lut.output, state.sample.doc:cuda()}

   local target = state.sample.target:cuda()
   state.sample.target = {target:view(target:nElement()), state.sample.word:cuda()}

   decoder:setTarget(state.sample.target[1])
end

-- [state, input_lm, input_topic] => [state, input_lm, topic_vector]
vae_encoder = nn.ParallelTable()
                :add(nn.Identity())
                :add(nn.Identity())
                :add(nn.Sequential()
                    :add(topic_encoder)
                    :add(KLD)
                    :add(nn.Sampler())
                    :add(nn.SoftMax()))
-- => [state, input_with_topic, topic_vector]
input_add_topic = nn.Sequential()
                :add(nn.NarrowTable(2,2))
                :add(nn.ParallelTable()
                    :add(nn.Identity())
                    :add(nn.Replicate(20)))
                :add(nn.JoinTable(3))
input_ready = nn.ConcatTable()
                :add(nn.SelectTable(1))
                :add(input_add_topic)
                :add(nn.SelectTable(-1))
-- => [rnn_output: n x t x (topic_dim + lstm_dim), topic_output: n x vocab]
rnn_preprocessing = nn.ParallelTable()
                :add(nn.Identity())
                :add(nn.Sequential():add(nn.Dropout(config.dropout)):add(nn.SplitTable(1)))
rnn_decoder = nn.Sequential()
                :add(nn.NarrowTable(1,2))
                :add(rnn_preprocessing)
                :add(rnn)
                :add(nn.SelectTable(2))
                :add(nn.SelectTable(-1))
                :add(nn.JoinTable(1))
                :add(nn.Dropout(config.dropout))
                :add(nn.Narrow(2,1,config.nhid)) --Maybe Concate the topic Vector
                :add(decoder)
output = nn.ConcatTable()
                :add(rnn_decoder)
                :add(nn.Sequential()
                    :add(nn.SelectTable(-1))
                    :add(topic_decoder))
local model = nn.Sequential()
            :add(vae_encoder)
            :add(input_ready)
            :add(output)

collectgarbage()
model:cuda()
crit:cuda()
print(model:get(1):get(3):get(2))

--[[
datas = {{{torch.rand(16,512):cuda(), torch.rand(16,512):cuda()}}, torch.rand(20,16,512):cuda(), torch.rand(16,5865):cuda()}
targets = {torch.ones(320):cuda(), torch.ones(16, 5865):cuda()} 
decoder:setTarget(targets[1])
output = model:forward(datas)
print(output)
print(targets)
print(crit:forward(output, targets))
--]]
--------------------------------------------------------------------------------
-- TORCHNET
--------------------------------------------------------------------------------
local timer     = tnt.TimeMeter{unit = true}
local logtimer  = tnt.TimeMeter()
local tottimer  = tnt.TimeMeter()

local trainloss_LM = tnt.AverageValueMeter()
local trainloss_topic = tnt.AverageValueMeter()
local trainloss_KL = tnt.AverageValueMeter()

local function runvalidation(network, criterion, iterator)
   local engine = tnt.SGDEngine()
   local meter  = tnt.AverageValueMeter()

   function engine.hooks.onStart(state)
      state.hid = rnn:initializeHidden(tbsz)
   end

   engine.hooks.onSample = onsample

   function engine.hooks.onForwardCriterion(state)
      meter:add(state.criterion.output)
      state.hid = tnt.utils.table.clone(rnn:getLastHidden())
   end

   engine:test{
      network   = network,
      iterator  = iterator,
      criterion = criterion,
   }

   return meter:value()
end

local engine = tnt.OptimEngine()

function engine.hooks.onStart(state)
   local eps = config.epsilon
   state.optim.paramVariance = state.gradParams:clone():fill(eps)
   state.optim.paramStd      = state.gradParams:clone()
   state.optim.lutVariance   = torch.Tensor(ntoken, 1):typeAs(lut.weight):fill(eps)
   state.hid = rnn:initializeHidden(bsz)
end

function engine.hooks.onStartEpoch(state)
    timer:reset()
    trainloss_LM:reset()
    trainloss_topic:reset()
    trainloss_KL:reset()
end

engine.hooks.onSample = onsample

function engine.hooks.onBackward(state)
   -- clip gradients
   if config.clip > 0 then
      local norm = state.gradParams:norm()
      if norm > config.clip then
         state.gradParams:div(math.max(norm, 1e-6) / config.clip)
      end
   end

   local gradinput = model.gradInput[2]:view(bptt * bsz, config.isz)
   local idx       = state.inputlut:view(bptt * bsz)
   local variance  = state.optim.lutVariance
   variance:indexAdd(1, idx, torch.pow(gradinput, 2):mean(2))
   gradinput:cdiv(torch.sqrt(variance:index(1, idx):expandAs(gradinput)))
   lut:accUpdateGradParameters(state.inputlut, model.gradInput[2], state.config.learningRate)
end

function engine.hooks.onUpdate(state)
   trainloss_LM:add(state.criterion:LmLoss())
   trainloss_topic:add(state.criterion:TopicLoss())
   trainloss_KL:add(state.network:get(1):get(3):get(2).loss)
   timer:incUnit()
   state.hid = tnt.utils.table.clone(rnn:getLastHidden())

   if logtimer:value() > 300 then
      local msbatch = timer:value() * 1000
      local train_LM_loss = trainloss_LM:value()
      local trainppl = math.exp(train_LM_loss)
      local train_topic_reconstr = trainloss_topic:value()
      local train_topic_KL = trainloss_KL:value()
      local validppl = math.exp(runvalidation(model, crit, valid))

      local str = string.format(
         '| epoch %2d | %8d samples | %7d ms/batch ' ..
         '| %5d min | train ppl %5.1f (%4.2f, %4.2f, %4.2f) | valid ppl %5.1f',
         state.epoch, state.t, msbatch,
         tottimer:value() / 60, trainppl, train_LM_loss,
         train_topic_reconstr, train_topic_KL, validppl)
      print(str)
      if logfile then
         logfile:write(str .. '\n')
         logfile:flush()
      end

      collectgarbage()
      model:training()
      trainloss_LM:reset()
      trainloss_topic:reset()
      trainloss_KL:reset()
      logtimer:reset()
      timer:reset()
   end
end

function engine.hooks.onEndEpoch(state)
   local msbatch = timer:value() * 1000
   local train_LM_loss = trainloss_LM:value()
   local trainppl = math.exp(train_LM_loss)
   local train_topic_reconstr = trainloss_topic:value()
   local train_topic_KL = trainloss_KL:value()
   local validppl = math.exp(runvalidation(model, crit, valid))
   local testppl  = math.exp(runvalidation(model, crit, test))

   local str = string.format(
      '| epoch %2d | %8d samples | %7d ms/batch | %5d min ' ..
      '| train ppl %5.1f (%4.2f, %4.2f %4.2f) | valid ppl %5.1f | test ppl %5.1f',
      state.epoch, state.t, msbatch, tottimer:value() / 60,
      trainppl, train_LM_loss, train_topic_reconstr, 
      train_topic_KL, validppl, testppl)
   print(str)
   if logfile then
      logfile:write(str .. '\n')
      logfile:flush()
   end

   collectgarbage()
   model:training()
   trainloss_LM:reset()
   trainloss_topic:reset()
   trainloss_KL:reset()
   logtimer:reset()
   timer:reset()

   if state.epoch >= 5 then
      state.config.learningRate = state.config.learningRate / 2
   end
end

tottimer:reset()
local config_opt = {
   learningRate = config.lr,
}

engine:train{
   network     = model,
   criterion   = crit,
   iterator    = train,
   optimMethod = optim.adagrad,
   maxepoch    = config.maxepoch,
   config      = config_opt,
}

--------------------------------------------------------------------------------
-- MODEL SAVING
--------------------------------------------------------------------------------

if config.outdir ~= '' then
   local model = nn.Sequential()
      :add(nn.ParallelTable()
              :add(nn.Identity())
              :add(nn.Sequential()
                      :add(lut)
                      :add(nn.Dropout(config.dropout))
                      :add(nn.SplitTable(1))
                  )
          )
      :add(rnn)
      :add(nn.SelectTable(2))
      :add(nn.SelectTable(-1))
      :add(nn.JoinTable(1))
      :add(nn.Dropout(config.dropout))
      :add(decoder)

   torch.save(paths.concat(config.outdir, 'model.t7'),
              {model = model, dic = dic, cutoff = cutoff, config = config})
end
