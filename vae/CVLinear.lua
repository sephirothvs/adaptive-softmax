require 'nn'
require 'cunn'
require 'ACOSLayer.lua'
local CVLinear, parent = torch.class('nn.CVLinear', 'nn.Module')

function CVLinear:__init(inputSize, outputSize, lambda)
    parent.__init(self)
    self.lambda = lambda
    self.loss = 0
    self.linear = nn.Linear(inputSize, outputSize, false)
    self.weight = self.linear.weight
    self.gradWeight = self.linear.gradWeight
    self.target = {torch.zeros(1), torch.zeros(1)}
    local m1= nn.Sequential():add(nn.ConcatTable()
                :add(nn.Sequential():add(nn.Identity()):add(nn.Replicate(inputSize, 2)):add(nn.Reshape(inputSize*inputSize, outputSize)))
                :add(nn.Sequential():add(nn.Identity()):add(nn.Replicate(inputSize, 1)):add(nn.Reshape(inputSize*inputSize, outputSize))) )
            :add(nn.CosineDistance()):add(nn.ACOSLayer())
    local m2 = nn.ConcatTable():add(nn.Identity())
                               :add(nn.Sequential():add(nn.Mean()):add(nn.Replicate(inputSize*inputSize)):add(nn.Reshape(inputSize*inputSize)))
                               :add(nn.Mean())
    local m3 = nn.ConcatTable():add(nn.SelectTable(-1))
                               :add(nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.CSubTable()):add(nn.Power(2)):add(nn.Mean()))
    self.model = nn.Sequential():add(m1):add(m2):add(m3)
    self.criterion = nn.ParallelCriterion():add(nn.AbsCriterion(), -self.lambda):add(nn.AbsCriterion(), self.lambda)        
end

function CVLinear:updateOutput(input) 
    self.inputs = input
    self.res = self.linear:forward(input)
    self.output:resize(self.res:size()):copy(self.res)
    self.f = self.model:forward(self.weight)
    self.loss = self.criterion:forward(self.f, self.target)
    return self.output
end

function CVLinear:updateGradInput(input, gradOutput)
    local grad = self.linear:backward(input, gradOutput)
    self.gradInput:resize(grad:size()):copy(grad)
    return self.gradInput
end

function CVLinear:accGradParameters(input, gradOutput)
    self.linear:accGradParameters(input, gradOutput)
    local df = self.criterion:backward(self.f, self.target)
    local grad = self.model:backward(self.weight, df)
    self.gradWeight:add(grad)
    self.grads = grad
end

function CVLinear:cuda(...)
    --for key,param in pairs(self) do
    --    print(key)
    --end
    self._type = 'torch.CudaTensor'
    self.output:cuda()
    self.gradInput:cuda()
    self.target = {torch.zeros(1):cuda(), torch.zeros(1):cuda()} 
    self.linear:cuda()
    self.weight = self.linear.weight
    self.gradWeight = self.linear.gradWeight
    self.model:cuda()
    self.criterion:cuda()
end
