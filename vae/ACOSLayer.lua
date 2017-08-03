local ACOSLayer, parent = torch.class('nn.ACOSLayer', 'nn.Module')

function ACOSLayer:__init()
    parent.__init(self)
end

function ACOSLayer:updateOutput(input)
    self.output = torch.acos(input)
    self.datas = torch.pow(input, 2)
    return self.output
end

function ACOSLayer:updateGradInput(input, gradOutput)
    local res = 1 - torch.pow(torch.sqrt( 1 - self.datas )+1e-8, -1)
    self.gradInput:resize(input:size()):copy(torch.cmul(res, gradOutput))
    return self.gradInput
end
