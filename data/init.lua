-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local tds      = require 'tds'
local stringx  = require 'pl.stringx'
local tablex   = require 'pl.tablex'

--require 'data.RNNDataset'
require 'data.TopicRNNDataset'

local data = {}

data.addword = function(dic, word, count)
   local count = count or 1
   if dic.word2idx[word] == nil then
      dic.idx2word:insert(word)
      dic.word2idx[word] = #dic.idx2word
      dic.idx2freq[#dic.idx2word] = count
   else
      local idx = dic.word2idx[word]
      dic.idx2freq[idx] = dic.idx2freq[idx] + count
   end
   return dic
end

data.addtopicword = function(dic, word)
    --if #dic.tidx2word == 0 then
    --    dic.tidx2word:insert("<unk>")
    --    dic.tword2idx[word] = #dic.tidx2word
    --end
    if dic.tword2idx[word] == nil and dic.powerwords[dic.word2idx[word]]>0 then
        dic.tidx2word:insert(word)
        dic.tword2idx[word] = #dic.tidx2word
    end
    return dic
end

data.getidx = function(dic, word)
   return dic.word2idx[word] or dic.word2idx["<unk>"]
end

data.gettopicidx = function(dic, word)
    return dic.tword2idx[word] or 0
end

data.initdictionary = function()
   local dic = {
      idx2word = tds.Vec(),
      word2idx = tds.Hash(),
      idx2freq = {},
      powerwords={},
      tidx2word = tds.Vec(),
      tword2idx = tds.Hash(),
   }
   data.addword(dic, "</s>")
   data.addword(dic, "<unk>")
   return dic
end

data.savedictionary = function(dic, filename)
   print(filename)
   local fout = io.open(filename, 'w')
   for i = 1, #dic.idx2word do
      fout:write(i ..' '.. dic.idx2word[i] ..' '.. dic.idx2freq[i] ..' '..dic.powerwords[i]..'\n')
   end
   fout:close()
end

data.loaddictionary = function(filename)
   local dic = data.initdictionary()
   dic.powerwords = {}
   for line in io.lines(filename) do
      local tokens = stringx.split(line)
      local idx = tonumber(tokens[1])
      local freq = tonumber(tokens[3])
      local word = tokens[2]
      local power = tonumber(tokens[4])
      dic.idx2word[idx] = word
      dic.word2idx[word] = idx
      dic.idx2freq[idx] = freq
      dic.powerwords[idx] = power
   end
   dic.idx2freq = torch.Tensor(dic.idx2freq)
   dic.powerwords = torch.Tensor(dic.powerwords)
   return dic
end

data.makedictionary = function(filename)
   local dic = data.initdictionary()
   local lines = 0

   for line in io.lines(filename) do
      local words = stringx.split(line)
      tablex.map(function(w) data.addword(dic, w) end, words)
      lines = lines + 1

      if lines % 10000 == 0 then
         collectgarbage()
      end
   end
   dic.idx2freq[dic.word2idx["</s>"]] = lines
   dic.idx2freq[dic.word2idx["<unk>"]] = dic.idx2freq[2] ~= 0
      and dic.idx2freq[2] or 1 -- nonzero hack

   dic.idx2freq = torch.Tensor(dic.idx2freq)

   -- sort dic
   local freq, idxs = dic.idx2freq:sort(true)
   local newdic = data.initdictionary()
   for i = 1, idxs:size(1) do
      data.addword(newdic, dic.idx2word[idxs[i]], freq[i]) 
   end
   newdic.idx2freq = torch.Tensor(newdic.idx2freq) 
   collectgarbage()
   collectgarbage()
   
   newdic.powerwords = torch.ones(#newdic.idx2word)  -- remove stopword
   for line in io.lines('./dataset/stopwords.txt') do
      local words = stringx.split(line)
      if newdic.word2idx[words[1]] ~= nil then
         newdic.powerwords[newdic.word2idx[words[1]]] = 0
      end
   end
   newdic.powerwords:narrow(1, 1, #newdic.idx2word * 0.001):fill(0)
   newdic.powerwords[torch.le(newdic.idx2freq, 10)] = 0
   newdic.indexpowerwords = torch.cmul(newdic.powerwords, torch.cumsum(newdic.powerwords)):long()
   for i = 1, newdic.powerwords:size(1)  do
       data.addtopicword(newdic, dic.idx2word[i]) 
   end
   --newdic.tidx2word:insert("<unk>")                                         
   --newdic.tword2idx["<unk>"] = #dic.tidx2word
   print(string.format("| Dictionary size %d", #newdic.idx2word))
   return newdic
end

data.sortthresholddictionary = function(dic, threshold)
   local freq, idxs = dic.idx2freq:sort(true)
   local newdic = data.initdictionary()

   for i = 1, idxs:size(1) do
      if freq[i] <= threshold then
         break
      end
      data.addword(newdic, dic.idx2word[idxs[i]], freq[i])
   end
   newdic.idx2freq = torch.Tensor(newdic.idx2freq)

   collectgarbage()
   collectgarbage()
   print(string.format("| Dictionary size %d", #newdic.idx2word))

   return newdic
end

local function add_data_to_tensor(tensor, buffer)
   if tensor then
      if #buffer > 0 then
         return torch.cat(tensor, torch.LongTensor(buffer))
      else
         return tensor
      end
   else
      return torch.LongTensor(buffer)
   end
end

data.loadfile = function(filename, dic)
   local buffer = {}
   local tensor = nil

   local buffer2 = {}
   local tensor2 = nil

   for line in io.lines(filename) do
      local words = stringx.split(line)
      table.insert(words, "</s>")
      local idx = tablex.map(function(w) return data.getidx(dic, w) end, words)
      tablex.insertvalues(buffer, idx)
      local idx2 = tablex.map(function(w) return data.gettopicidx(dic, w) end, words)
      tablex.insertvalues(buffer2, idx2)

      if #buffer > 5000000 then
         tensor = add_data_to_tensor(tensor, buffer)
         buffer = {}
         tensor2 = add_data_to_tensor(tensor2, bnuffer2)
         buffer2 = {}
         collectgarbage()
      end
   end
   tensor = add_data_to_tensor(tensor, buffer)
   tensor2= add_data_to_tensor(tensor2, buffer2)

   print(string.format("| Load file %s: %d tokens",
                       filename, tensor:size(1)))
   collectgarbage()

   return tensor, tensor2
end

return data
