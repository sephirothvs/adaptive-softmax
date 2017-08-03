local KLDPenalty, parent = torch.class('nn.KLDPenalty', 'nn.Module')

function KLDPenalty:__init(weight, batchsize)
    parent.__init(self)
    self.myweight = weight
    self.batchsizes = batchsize
    --print(self.myweight)
    --print(self.batchsizes)
end

function KLDPenalty:updateOutput(input)
    local mean, log_var = table.unpack(input)
    self.output = input
    local mean_sq = torch.pow(mean, 2)
    local KLDelements = log_var:clone()
    KLDelements:exp():mul(-1)
    KLDelements:add(-1, torch.pow(mean, 2))
    KLDelements:add(1)
    KLDelements:add(log_var)
    self.loss = -0.5 * torch.sum(KLDelements) * self.myweight
    return self.output
end

function KLDPenalty:updateGradInput(input, gradOutput)
    assert(#gradOutput == 2)
    local mean, log_var = table.unpack(input)
    self.gradInput = {}
    self.gradInput[1] = mean:clone():mul(self.myweight) + gradOutput[1]
    self.gradInput[2] = torch.exp(log_var):mul(-1):add(1):mul(-0.5):mul(self.myweight) + gradOutput[2]
    return self.gradInput
end
