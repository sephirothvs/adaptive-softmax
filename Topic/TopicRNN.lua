require 'nn'

local argcheck = require 'argcheck'
local rnnlib = require 'rnnlib.env'

nn.TopicLSTM = argcheck{
    { name = 'inputsize' , type = 'number'   ,                 },
    { name = 'hidsize'   , type = 'number'   ,                 },
    { name = 'nlayer'    , type = 'number'   ,                 },
    { name = 'ntopic'    , type = 'number'   ,                 },
    { name = 'hinitfun'  , type = 'function' , opt     = true  },
    { name = 'winitfun'  , type = 'function' , opt     = true  },
    { name = 'usecudnn'  , type = 'boolean'  , default = false },
    call = function(inputsize, hidsize, nlayer, ntopic, hinitfun, winitfun, usecudnn)
        local hids = { [0] = inputsize }
        for i = 1, nlayer do
            hids[i] = hidsize
        end
        local model = rnnlib.makeTopicRecurrent{
            cellfn      = rnnlib.cell.TopicLSTM,
            inputsize   = inputsize,
            hids        = hids,
            ntopic      = ntopic,
            hinitfun    = hintfun,
            winitfun    = winitfun,
        }
        if usecudnn then
            return nn.WrappedCudnnRnn(model, 'LSTM', hids, model.saveHidden)
        end
        return model, hids
    end
}
