--[[

topicmodel = nn.Sequential()
            :add(topic_encoder)
            :add(KLD)
            :add(nn.Sampler())
            :add(nn.SoftMax())
            :add(topic_decoder):cuda()
topiccrit = nn.BCECriterion():cuda()
--]]
require 'optim'
function trainTopic(modelt, criteriont, model2, crit2, W, grad, data, datav)
    local inputs_gpu  = torch.CudaTensor()
    local targets_gpu = torch.CudaTensor()
    local nTrain      = data:size()
    local targets = {torch.zeros(1):cuda(), torch.zeros(1):cuda()}
    local optimizer = optim.adam
    local optim_config = {
        learningRate = 1e-3,
        learningRateDecay = 0,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-8,
        weightDecay=0
    }
    local function feval(x)
        assert(x==W)
        grad:zero()
        local outputs = modelt:forward(inputs_gpu)
        local f = criteriont:forward(outputs, targets_gpu)
        local df_dw   = criteriont:backward(outputs, targets_gpu)
        modelt:backward(inputs_gpu, df_dw)
        
        --[[
        local weights = modelt:get(6):get(1).weight:t()     -- hard code
        local output2 = model2:forward(weights)
        local f2 = crit2:forward(output2, targets)
        local df2= crit2:backward(output2,targets)
        local dw = model2:backward(weights, df2)
        dw = dw:t()
        modelt:get(6):get(1).gradWeight:add(dw)  -- hard code
        --]]
        return f, grad
    end
    for t = 1, 10 do
        modelt:training()
        timer = torch.Timer()
        for j = 1, nTrain, 10 do
            local tmp = data:get(j)
            local inputs = tmp.doc
            local targets= tmp.word
            inputs_gpu:resize(inputs:size()):copy(inputs)
            targets_gpu:resize(targets:size()):copy(targets)
            optimizer(feval, W, optim_config)
        end
        local timespend = timer:time().real
        --print(t)
        local err1, err2, err3= evaluation(modelt, criteriont, model2, crit2, datav)
        print(string.format('Pre-Train Topic: Epoch %d | Valid Construction: %4.4f | Valid KL: %4.4f | Diversity Loss: %4.4f | Time: %4.4f s',t, err1, err2, err3, timespend))
    end
end

function evaluation(modelt, criteriont, model2, crit2, data)
    modelt:evaluate()
    local nTest = data:size()
    local err1=0
    local err2=0
    local err3=0
    local inputs_gpu = torch.CudaTensor()
    local targets_gpu = torch.CudaTensor()
    for k = 1, nTest do
        local tmp = data:get(k)
        local inputs = tmp.doc
        local targets= tmp.word
        inputs_gpu:resize(inputs:size()):copy(inputs)
        targets_gpu:resize(targets:size()):copy(targets)
        local outputs = modelt:forward(inputs_gpu)
        local f = criteriont:forward(outputs, targets_gpu)
        err1 = f + err1
        err2 = modelt:get(2).loss + err2
    end
    local weights = topicmodel:get(6):get(1).weight     -- hard code
    local f = model2:forward(weights)
    err3 = crit2:forward(f, {torch.zeros(1):cuda(), torch.zeros(1):cuda()})
    err1 = err1 / nTest
    err2 = err2 / nTest
    return err1, err2, err3
end


function visualtopic(embed, dic)
    -- embed: vocab x num_topic
    local dicts = dic.tidx2word
    local _, embeds = torch.sort(embed:t(), true)
    for i = 1, embeds:size(1) do
        filename = "Topic "..i..":"
        for j = 1, 15 do
            filename = filename ..' ' ..dicts[embeds[i][j]]
        end
        print(filename)
    end
end
