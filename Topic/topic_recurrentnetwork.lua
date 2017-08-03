local argcheck = require 'argcheck'
local mutils   = require 'rnnlib.mutils'
local rnnlib   = require 'rnnlib.env'

rnnlib.makeTopicRecurrent = argcheck{
    doc = [[
<a name="makeRecurrent">
### rnnlib.makeTopicRecurrent(@ARGP)
@ARGT

Given a cell function, e.g. from 'rnnlib.cell', `makeRecurrent` constructs an
RNN whose outer layer is of type nn.SequenceTable and inner layers are of type
nn.RecurrentTable.

Be sure to call :training() or :evaluate() before performing training or
evaluation, since the hidden state logic depends on this. If you would rather
handle this manually, set .saveHidden = false.
]],

    { name = "cellfn"     , type = "function" ,                              },
    { name = "inputsize"  , type = "number"   ,                              },
    { name = "hids"       , type = "table"    ,                              },
    { name = "ntopic"     , type = "number"   ,                              },
    { name = "hinitfun"   , type = "function" , opt     = true               },
    { name = "winitfun"   , type = "function" , default = mutils.defwinitfun },
    { name = "savehidden" , type = "boolean"  , default = true               },
    call = function(cellfn, inputsize, hids, ntopic, hinitfun, winitfun, savehidden)
        hids[0] = inputsize
        local initfs = {}
        local nlayer = #hids

        local layers = {}
        for i = 1, nlayer do
            local c, f = cellfn(hids[i-1], hids[i], ntopic)
            layers[i] = nn.RecurrentTable{
                dim = 2,
                module = rnnlib.cell.gModule(c),
            }
            initfs[i] = f
        end
        local network = rnnlib.make(1, layers, hinitfun)
        return rnnlib.setupRecurrent{
            network = network,
            initfs = initfs,
            winitfun = winitfun,
            savehidden = savehidden,
        }
    end
}
