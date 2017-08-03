require 'nn'
local TopicCriterion, parent = torch.class('nn.TopicCriterion', 'nn.Criterion')

function TopicCriterion:__init(repeatTarget)
    parent.__init(self)
    self.criterions = {}
    self.weights = {}
    self.gradInput = {}
    self.results = {}
    self.repeatTarget = repeatTarget
end

function TopicCriterion:add(criterion, weight)
    assert(criterion, 'no criterion provided')
    weight = weight or 1
    table.insert(self.criterions, criterion)
    table.insert(self.weights, weight)
    return self
end

function TopicCriterion:updateOutput(input, target)
    self.output = 0
    for i,criterion in ipairs(self.criterions) do
        local target = self.repeatTarget and target or target[i]
        local result = criterion:updateOutput(input[i],target)
        self.results[i] = result
        self.output = self.output + self.weights[i] * result
    end
    return self.output
end

function TopicCriterion:LmLoss()
    return self.results[1]
end

function TopicCriterion:TopicLoss()
    return self.results[2]
end

function TopicCriterion:updateGradInput(input, target)
    self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
    nn.utils.recursiveFill(self.gradInput, 0)
    for i,criterion in ipairs(self.criterions) do
        local target = self.repeatTarget and target or target[i]
        nn.utils.recursiveAdd(self.gradInput[i], self.weights[i], criterion:updateGradInput(input[i], target))
    end
    return self.gradInput
end

function TopicCriterion:type(type, tensorCache)
    self.gradInput = {}
    return parent.type(self, type, tensorCache)
end
