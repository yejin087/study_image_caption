import json
import torch
from pytorch_transformers import *
import logging
import argparse
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
import math
import numpy
import collections
import random
from tqdm import tqdm

## Model

class NoContext(BertModel):
    def __init__(self, config):

        super(BertModel, self).__init__(config)
        self.bert = BertModel(config)
        self.hidden_dim = 768


        self.dropout = torch.nn.Dropout(0.5)
        self.second_last_layer = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.last_layer = torch.nn.Linear(self.hidden_dim, 2)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, event_1, event_2, entities):
        event_1_representation, _ = self.bert(event_1.unsqueeze(0)) #[1, seq_len, 768]
        event_1_representation = torch.mean(event_1_representation, dim=1) # [1, 768]
        event_2_representation, _ = self.bert(event_2.unsqueeze(0))
        event_2_representation = torch.mean(event_2_representation, dim=1)
        overall_representation = torch.cat([event_1_representation, event_2_representation], dim=1) # [1, 768*2]
        overall_representation = self.dropout(overall_representation)
        prediction = self.last_layer(self.second_last_layer(overall_representation))
        return prediction


class NoAttention(BertModel):
    def __init__(self, config):
        super(NoAttention, self).__init__(config)
        self.bert = BertModel(config)

        self.embedding_size = 300
        self.hidden_dim = 768

        self.dropout = torch.nn.Dropout(0.5)
        self.second_last_layer = torch.nn.Linear(self.hidden_dim * 2 + self.embedding_size, self.hidden_dim)
        self.last_layer = torch.nn.Linear(self.hidden_dim, 2)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, event_1, event_2, entities):
        event_1_representation, _ = self.bert(event_1.unsqueeze(0))
        event_1_representation = torch.mean(event_1_representation, dim=1) # [1, 768]
        event_2_representation, _ = self.bert(event_2.unsqueeze(0))
        event_2_representation = torch.mean(event_2_representation, dim=1)
        entity_representation = torch.mean(entities.unsqueeze(0), dim=1) # [1, 300]
        overall_representation = torch.cat([event_1_representation, event_2_representation, entity_representation],
                                           dim=1) #[1, 1836]
        overall_representation = self.dropout(overall_representation)
        prediction = self.last_layer(self.second_last_layer(overall_representation))
        return prediction

class ResNetAsContext(BertModel):
    def __init__(self, config):
        super(ResNetAsContext, self).__init__(config)
        self.bert = BertModel(config)

        self.hidden_dim = 768
        self.compress_size = 200

        self.dropout = torch.nn.Dropout(0.5)
        self.compress = torch.nn.Linear(2048, self.compress_size)
        self.second_last_layer = torch.nn.Linear(self.hidden_dim * 2 + self.compress_size, self.hidden_dim)
        # self.second_last_layer = torch.nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        self.last_layer = torch.nn.Linear(self.hidden_dim, 2)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, event_1, event_2, resnet_representation):
        event_1_representation, _ = self.bert(event_1.unsqueeze(0))
        event_1_representation = torch.mean(event_1_representation, dim=1)
        event_2_representation, _ = self.bert(event_2.unsqueeze(0))
        event_2_representation = torch.mean(event_2_representation, dim=1)

        overall_representation = torch.cat([event_1_representation, event_2_representation, self.compress(resnet_representation.unsqueeze(0))], dim=1)


        prediction = self.last_layer(self.second_last_layer(overall_representation))
        return prediction

class BERTCausal(BertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.bert = BertModel(config)
        self.embedding_size = 300
        self.hidden_dim = 200

        self.dropout = torch.nn.Dropout(0.5)
        self.second_last_layer = torch.nn.Linear(768*2 + self.embedding_size * 2, self.hidden_dim)

        self.attention_to_entity = torch.nn.Linear(768 + self.embedding_size, 1)
        self.attention_to_word = torch.nn.Linear(768 + self.embedding_size, 1)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def cross_attention(self, eventuality_representation, entity_representations):

        number_of_words = eventuality_representation.size(0)
        number_of_objects = entity_representations.size(0)
        event_raw_representation = torch.mean(eventuality_representation, dim=0)  # 768

        event_raw_representation = event_raw_representation.repeat(number_of_objects, 1)  # 10*768

        event_attention = self.attention_to_entity(
            torch.cat([event_raw_representation, entity_representations], dim=1))  # 10 * 1

        context_representation = torch.mean(entity_representations * event_attention.repeat(1, self.embedding_size),
                                            dim=0)  # 1 * 300


        context_representation_for_attention = context_representation.repeat(number_of_words, 1)


        word_attention = self.attention_to_word(
            torch.cat([eventuality_representation, context_representation_for_attention], dim=1))  # num_words * 1

        event_representation = torch.mean(eventuality_representation * word_attention.repeat(1, 768),
                                          dim=0)

        return event_representation.unsqueeze(0), context_representation.unsqueeze(0)

    def forward(self, event_1, event_2, entities):
        event_1_representation = self.bert(event_1.unsqueeze(0))
        event_1_weighted_representation, event_1_context_representation = self.cross_attention(event_1_representation[0].squeeze(),entities)
        event_2_representation = self.bert(event_2.unsqueeze(0))
        event_2_weighted_representation, event_2_context_representation = self.cross_attention(event_2_representation[0].squeeze(),entities)

        overall_representation = torch.cat(
            [event_1_weighted_representation, event_2_weighted_representation, event_1_context_representation,
             event_2_context_representation], dim=1)
             
        overall_representation = self.dropout(overall_representation)

        prediction = self.second_last_layer(overall_representation)
        return prediction



class GPTCausal(GPT2Model):
    def __init__(self, config):
        super(GPT2Model, self).__init__(config)
        self.lm = GPT2Model(config)
        self.embedding_size = 300
        self.hidden_dim = 200

        self.dropout = torch.nn.Dropout(0.5)
        self.second_last_layer = torch.nn.Linear(768*2 + self.embedding_size * 2, self.hidden_dim)


        self.attention_to_entity = torch.nn.Linear(768 + self.embedding_size, 1)
        self.attention_to_word = torch.nn.Linear(768 + self.embedding_size, 1)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def cross_attention(self, eventuality_representation, entity_representations):

        number_of_words = eventuality_representation.size(0)
        number_of_objects = entity_representations.size(0)
        event_raw_representation = torch.mean(eventuality_representation, dim=0)  # 768

        event_raw_representation = event_raw_representation.repeat(number_of_objects, 1)  # 10*768

        event_attention = self.attention_to_entity(
            torch.cat([event_raw_representation, entity_representations], dim=1))  # 10 * 1

        context_representation = torch.mean(entity_representations * event_attention.repeat(1, self.embedding_size),
                                            dim=0)  # 1 * 300


        context_representation_for_attention = context_representation.repeat(number_of_words, 1)


        word_attention = self.attention_to_word(
            torch.cat([eventuality_representation, context_representation_for_attention], dim=1))  # num_words * 1

        event_representation = torch.mean(eventuality_representation * word_attention.repeat(1, 768),
                                          dim=0)

        return event_representation.unsqueeze(0), context_representation.unsqueeze(0)

    def forward(self, event_1, event_2, entities):
        event_1_representation = self.lm(event_1.unsqueeze(0))
        event_1_weighted_representation, event_1_context_representation = self.cross_attention(
            event_1_representation[0].squeeze(),
            entities)
        event_2_representation = self.lm(event_2.unsqueeze(0))
        event_2_weighted_representation, event_2_context_representation = self.cross_attention(
            event_2_representation[0].squeeze(),
            entities)

        overall_representation = torch.cat(
            [event_1_weighted_representation, event_2_weighted_representation, event_1_context_representation,
             event_2_context_representation], dim=1)
        overall_representation = self.dropout(overall_representation)

        prediction = self.second_last_layer(overall_representation)
        return prediction