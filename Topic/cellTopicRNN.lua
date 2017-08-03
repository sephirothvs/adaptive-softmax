local _init = function(bsz, nhid, t, cache)
    local t = t or 'torch.Tensor'
    local tensor = cache or torch.Tensor()
    return tensor:resize(bsz, nhid):type(t):fill(0)
end

rnnlib.cell.TopicLSTM = function(nin, nhid, k)
    local make = function(prevch, input)
        -- prevh : { prevc : node, prevh : node}
        -- input : node
        local split = {prevch:split(2)}
        local prevc = split[1]
        local prevh = split[2]

        -- input : nbatch x (nin + k)
        input1 = nn.Narrow(2, 1, nin)(input):annotate{name="data"}
        input2 = nn.Narrow(2, nin+1, k)(input):annotate{name="topics"}

        -- the four gates are computed simulatenously
        local i2h1 = nn.Linear(nin,  4 * nhid, false)(input1):annotate{name="lstm_i2h_1"}
        local i2h2 = nn.Linear(k,    4 * nhid, false)(input2):annotate{name="lstm_i2h_2"}
        local i2h3 = nn.CMulTable()({i2h1, i2h2})
        local i2h  = nn.Linear(4 * nhid, 4 * nhid, false)(i2h3):annotate{name="lstm_i2h"}

        local h2h1 = nn.Linear(nhid, 4 * nhid, false)(prevh):annotate{name="lstm_h2h_1"}
        local h2h2 = nn.Linear(k,    4 * nhid, false)(input2):annotate{name="lstm_h2h_2"}
        local h2h3 = nn.CMulTable()({h2h1, h2h2})
        local h2h  = nn.Linear(4 * nhid, 4 * nhid, false)(h2h3):annotate{name="lstm_h2h"}

        -- the gates are separated
        local gates = nn.CAddTable()({i2h, h2h})
        -- assumes that input is of dimension nbatch x ngate * nhid
        gates = nn.SplitTable(2)(nn.Reshape(4, nhid)(gates))
        -- apply nonlinearities:
        local igate = nn.Sigmoid()(nn.SelectTable(1)(gates)):annotate{name="lstm_ig"}
        local fgate = nn.Sigmoid()(nn.SelectTable(2)(gates)):annotate{name="lstm_fg"}
        local cgate = nn.Tanh   ()(nn.SelectTable(3)(gates)):annotate{name="lstm_cg"}
        local ogate = nn.Sigmoid()(nn.SelectTable(4)(gates)):annotate{name="lstm_og"}
        -- c_{t+1} = fgate * c_t + igate * f(h_{t+1}, i_{t+1})
        local nextc = nn.CAddTable()({
            nn.CMulTable()({fgate, prevc}),
            nn.CMulTable()({igate, cgate})
        }):annotate{name="nextc"}
        -- h_{t+1} = ogate * c_{t+1}
        local nexth  = nn.CMulTable()({ogate, nn.Tanh()(nextc)}):annotate{name="lstm_nexth"}
        local nextch = nn.Identity ()({nextc, nexth}):annotate{name="lstm_nextch"}
        local output = nn.JoinTable(2)({nexth, input2})
        return nextch, output
    end

    local init = function(bsz, t, cache)
        return { 
            _init(bsz, nhid, t, cache and cache[1]),
            _init(bsz, nhid, t, cache and cache[2])
        }
    end
    return make, init
end
