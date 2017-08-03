require 'torch'
require 'nn'
require 'cudnn'

local VAE = {}

function VAE.get_encoder(input_size, hidden_layer_size, topic_size)
    -- The Encoder
    local encoder = nn.Sequential()
    encoder:add(nn.Linear(input_size, hidden_layer_size))                       
    encoder:add(nn.Dropout(0.8))                                                
    encoder:add(nn.ReLU(true))
    encoder:add(nn.Linear(hidden_layer_size, hidden_layer_size))
    encoder:add(nn.Dropout(0.8))
    encoder:add(nn.ReLU(true))
    mean_logvar = nn.ConcatTable()
    mean_logvar:add(nn.Linear(hidden_layer_size, topic_size))
    mean_logvar:add(nn.Linear(hidden_layer_size, topic_size))
    encoder:add(mean_logvar)
    return encoder
end

function VAE.get_decoder(topic_size, vocab_size, word_dim)
    -- The Decoder
    local decoder = nn.Sequential()
    --decoder:add(nn.Linear(topic_size, word_dim))
    --decoder:add(nn.Linear(word_dim, vocab_size))
    decoder:add(nn.Linear(topic_size, vocab_size))
    decoder:add(nn.Sigmoid())
    --decoder:add(nn.LogSoftMax())
    return decoder
end

return VAE
