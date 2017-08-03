local weightedSumCriterion, parent = torch.class('nn.weightedSumCriterion', 'nn.Criterion')

function weightedSumCriterion:__init()
    parent.__init(self)
    self.output = torch.zeros(1)
    self.target = torch.Tensor()
    self.gradInput = torch.Tensor()
end

function weightedSumCriterion:updateOutput(input, target)    
    if type(target) == 'userdata' then
        if torch.typename(input):find('torch%.Cuda.*Tensor') then
            self.target = torch.CudaTensor and target:cuda() or target:cuda()
        else
            self.target = target:double()
        end
    elseif torch.typename(input):find('torch%.Cuda.*Tensor') then
        self.target = torch.CudaTensor and target:cuda() or target
    else
        error("something went wrong!")
    end
    self.output = -torch.cmul(input, self.target):sum()/input:size(1)
    return self.output
end

function weightedSumCriterion:updateGradInput(input, target)
    if type(target) == 'userdata' then
        if torch.typename(input):find('torch%.Cuda.*Tensor') then
            self.target = torch.CudaTensor and target:cuda() or target:cuda()
        else
            self.target = target:double()
        end
    elseif torch.typename(input):find('torch%.Cuda.*Tensor') then
        self.target = torch.CudaTensor and target:cuda() or target
    else
        error("something went wrong!")  
    end
    self.gradInput:resizeAs(self.target):copy(-self.target):div(input:size(1))
    return self.gradInput
end
