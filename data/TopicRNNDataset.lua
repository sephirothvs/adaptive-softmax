-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local tnt = require 'torchnet.env'

local TopicRNNDataset, _ = torch.class("tnt.TopicRNNDataset", "tnt.Dataset", tnt)

function TopicRNNDataset:__init(data, tdata, dic, bsz, bptt, window)
    local ntoken = data:nElement()
    local nbatch = math.ceil(ntoken / (bsz * bptt))
    local tsize  = nbatch * bptt * bsz
    local buffer = torch.LongTensor(tsize):fill(1)
    buffer:narrow(1, tsize - ntoken + 1, ntoken):copy(data)
    buffer = buffer:view(bsz, nbatch * bptt):t()
    local buffer2 = torch.LongTensor(tsize):fill(0)  -- 0 means <\s>
    buffer2:narrow(1, tsize - ntoken + 1, ntoken):copy(tdata)
    buffer2 = buffer2:view(bsz, nbatch*bptt):t()
    
    self.vocabsize = #dic.tidx2word
    self.nbatch = nbatch
    self.bsz    = bsz
    self.bptt   = bptt
    self.window = window
    self.data   = torch.LongTensor(nbatch * bptt + 1, bsz):fill(1)
    self.data:narrow(1, 2, nbatch * bptt):copy(buffer)
    self.tdata  = torch.LongTensor(nbatch * bptt + 1, bsz):fill(0)
    self.tdata:narrow(1, 2, nbatch * bptt):copy(buffer2)
    self.dic =dic

    self.input  = torch.LongTensor(self.bptt, self.bsz)
    self.target = torch.LongTensor(self.bptt, self.bsz)
    self.doc    = torch.LongTensor(self.bptt*self.window, self.bsz):zero()
end

function TopicRNNDataset:size()
    return self.nbatch
end

function TopicRNNDataset:get(i)
    local pos    = 1 + self.bptt * (i - 1)
    self.input:copy(self.data:narrow(1, pos, self.bptt))
    self.target:copy(self.data:narrow(1, pos+1, self.bptt))

    local tpos  = 1 + self.bptt*(i-1-self.window)
    local epos  = -1
    if tpos < 0 then epos = self.bptt * (self.nbatch - self.window + i-1) + 1 end
    if epos == -1 then
        self.doc:copy(self.tdata:narrow(1, tpos, self.bptt*self.window))
    else
        local tmp = self.tdata:narrow(1, epos, (self.window-i+1)*self.bptt)
        local tmp1 
        if self.bsz > 1 then
            tmp1 = torch.cat(tmp:narrow(2, self.bsz, 1), tmp:narrow(2, 1, self.bsz-1), 2)
        else
            tmp1 = tmp
        end
        if i-1>0 then
            local tmp2 = self.tdata:narrow(1, 1, (i-1) * self.bptt)
            tmp1 = torch.cat(tmp1, tmp2, 1)
        end
        self.doc:copy(tmp1)
    end
    
    topic_doc = torch.Tensor(self.vocabsize, self.bsz):zero()
    for i = 1, self.bsz do 
        subtensor = self.doc:select(2, i)
        if subtensor:sum() ~= 0 then
            subtensor = subtensor[torch.ge(subtensor, 1)]
            for j = 1, subtensor:size(1) do
                topic_doc[subtensor[j]][i] = topic_doc[subtensor[j]][i] + 1
            end
        end
    end
    topic_doc = topic_doc:t()  -- hard code
    topic_docy = torch.ge(topic_doc, 1)
    return {input = self.input, target = self.target, doc=topic_doc, word=topic_docy}
end
