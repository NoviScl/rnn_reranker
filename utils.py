import json
import os
import random

from tqdm import tqdm
import glob
import os

import torch

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class GraphRetrieverConfig:

    def __init__(self,
                 example_limit: int,
                 task: str,
                 max_seq_length: int,
                 max_select_num: int,
                 max_para_num: int,
                 tfidf_limit: int,

                 train_file_path: str,
                 use_redundant: bool,
                 use_multiple_redundant: bool,
                 max_redundant_num: int,

                 dev_file_path: str,
                 beam: int,
                 min_select_num: int,
                 no_links: bool,
                 pruning_by_links: bool,
                 expand_links: bool,
                 eval_chunk: int,
                 tagme: bool,
                 topk: int,
                 db_save_path: str):

        # General
        self.example_limit = example_limit

        self.open = False
        
        self.task = task
        assert task in ['hotpot_distractor', 'hotpot_open',
                        'squad', 'nq', 'ambigqa',
                        None]

        if task == 'hotpot_open' or (train_file_path is None and task in ['squad', 'nq']):
            self.open = True
        
        self.max_seq_length = max_seq_length
        
        self.max_select_num = max_select_num

        self.max_para_num = max_para_num

        self.tfidf_limit = tfidf_limit
        assert self.tfidf_limit is None or type(self.tfidf_limit) == int
        
        # Train
        self.train_file_path = train_file_path

        self.use_redundant = use_redundant

        self.use_multiple_redundant = use_multiple_redundant
        if self.use_multiple_redundant:
            self.use_redundant = True

        self.max_redundant_num = max_redundant_num
        assert self.max_redundant_num is None or self.max_redundant_num > 0 or not self.use_multiple_redundant
        
        # Eval
        self.dev_file_path = dev_file_path
        assert self.train_file_path is not None or self.dev_file_path is not None or task is None

        self.beam = beam

        self.min_select_num = min_select_num
        assert self.min_select_num >= 1 and self.min_select_num <= self.max_select_num

        self.no_links = no_links
        
        self.pruning_by_links = pruning_by_links
        if self.no_links:
            self.pruning_by_links = False

        self.expand_links = expand_links
        if self.no_links:
            self.expand_links = False
        
        self.eval_chunk = eval_chunk

        self.tagme = tagme

        self.topk = topk

        self.db_save_path = db_save_path

    def __str__(self):
        configStr = '\n\n' \
                    '### RNN graph retriever configurations ###\n' \
                    '@@ General\n' \
                    '- Example limit: ' + str(self.example_limit) + '\n' \
                    '- Task: ' + str(self.task) + '\n' \
                    '- Open: ' + str(self.open) + '\n' \
                    '- Max seq length: ' + str(self.max_seq_length) + '\n' \
                    '- Max select num: ' + str(self.max_select_num) + '\n' \
                    '- Max paragraph num (including links): ' + str(self.max_para_num) + '\n' \
                    '- Limit of the initial TF-IDF pool: ' + str(self.tfidf_limit) + '\n' \
                    '\n' \
                    '@@ Train\n' \
                    '- Train file path: ' + str(self.train_file_path) + '\n' \
                    '- Use redundant: ' + str(self.use_redundant) + '\n' \
                    '- Use multiple redundant: ' + str(self.use_multiple_redundant) + '\n' \
                    '- Max redundant num: ' + str(self.max_redundant_num) + '\n' \
                    '\n' \
                    '@@ Eval\n' \
                    '- Dev file path: ' + str(self.dev_file_path) + '\n' \
                    '- Beam size: ' + str(self.beam) + '\n' \
                    '- Min select num: ' + str(self.min_select_num) + '\n' \
                    '- No links: ' + str(self.no_links) + '\n' \
                    '- Pruning by links (and top 1): ' + str(self.pruning_by_links) + '\n' \
                    '- Exapnd links (for NQ): ' + str(self.expand_links) + '\n' \
                    '- Eval chunk: ' + str(self.eval_chunk) + '\n' \
                    '- Tagme: ' + str(self.tagme) + '\n' \
                    '- Top K: ' + str(self.topk) + '\n' \
                    '- DB save path: ' + str(self.db_save_path) + '\n' \
                    '#########################################\n'

        return configStr


class InputExample(object):

    def __init__(self, guid, q, c, s_g):

        self.guid = guid
        self.question = q
        self.context = c
        self.short_gold = s_g

        # paragraph index -> title
        self.title_order = []

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_masks, segment_ids, output_masks, num_paragraphs, num_steps, ex_index = None):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.output_masks = output_masks
        self.num_paragraphs = num_paragraphs
        self.num_steps = num_steps

        self.ex_index = ex_index

def expand_links(context, all_linked_paras_dic, all_paras):
    for context_title in context:
        # Paragraphs from the same article
        raw_context_title = context_title.split('_')[0]

        if context_title not in all_linked_paras_dic:
            all_linked_paras_dic[context_title] = {}

        for title in all_paras:
            if title == context_title or title in all_linked_paras_dic[context_title]:
                continue
            raw_title = title.split('_')[0]
            if raw_title == raw_context_title:
                all_linked_paras_dic[context_title][title] = all_paras[title]

        
class DataProcessor:

    def get_train_examples(self, graph_retriever_config):

        examples = []

        assert graph_retriever_config.train_file_path is not None

        file_name = graph_retriever_config.train_file_path

        if os.path.exists(file_name):
            examples += self._create_examples(file_name, graph_retriever_config, "train")
        else:
            file_list = list(glob.glob(file_name+'*'))
            for file_name in file_list:
                examples += self._create_examples(file_name, graph_retriever_config, "train")

        assert len(examples) > 0
        return examples
        
    def get_dev_examples(self, graph_retriever_config):

        examples = []

        assert graph_retriever_config.dev_file_path is not None

        file_name = graph_retriever_config.dev_file_path

        if os.path.exists(file_name):
            examples += self._create_examples(file_name, graph_retriever_config, "dev")
        else:
            file_list = list(glob.glob(file_name+'*'))
            for file_name in file_list:
                examples += self._create_examples(file_name, graph_retriever_config, "dev")

        assert len(examples) > 0
        return examples

    '''
    Read training examples from a json file
    * file_name: the json file name
    * graph_retriever_config: the graph retriever's configuration
    * task: a task name like "hotpot_open"
    * set_type: "train" or "dev"

    clsi: Note that I've deleted support for other datasets to make the codebase more concise.
    '''
    def _create_examples(self, file_name, graph_retriever_config, set_type):

        task = graph_retriever_config.task
        jsn = json.load(open(file_name, 'r'))
        
        examples = []

        '''
        Limit the number of examples used.
        This is mainly for sanity-chacking new settings.
        '''
        if graph_retriever_config.example_limit is not None:
            random.shuffle(jsn)
            jsn = sorted(jsn, key = lambda x: x['q_id'])
            jsn = jsn[:graph_retriever_config.example_limit]

        '''
        For each question, we just keep k retrieved paragraphs as candidates (k=10).
        ''' 
        logger.info('#### Loading examples... from {} ####'.format(file_name))
        for (_, data) in enumerate(tqdm(jsn)):

            guid = data['q_id']
            question = data['question']
            context = data['context'] # {context title: paragraph}
            short_gold = data['short_gold'] # [title 1, title 2] (Both are gold)
            
            examples.append(InputExample(guid = guid,
                                         q = question,
                                         c = context,
                                         s_g = short_gold))

        # if set_type == 'dev':
        #     examples = sorted(examples, key = lambda x: len(x.all_paras))
        logger.info('Done!')
        
        return examples

def tokenize_question(question, tokenizer):
    tokens_q = tokenizer.tokenize(question)
    tokens_q = ['[CLS]'] + tokens_q + ['[SEP]']

    return tokens_q

def tokenize_paragraph(p, tokens_q, max_seq_length, tokenizer):
    tokens_p = tokenizer.tokenize(p)[:max_seq_length-len(tokens_q)-1]
    tokens_p = tokens_p + ['[SEP]']

    padding = [0] * (max_seq_length - len(tokens_p) - len(tokens_q))

    input_ids_ = tokenizer.convert_tokens_to_ids(tokens_q + tokens_p)
    input_masks_ = [1] * len(input_ids_)
    segment_ids_ = [0] * len(tokens_q) + [1] * len(tokens_p)

    input_ids_ += padding
    input_masks_ += padding
    segment_ids_ += padding

    assert len(input_ids_) == max_seq_length
    assert len(input_masks_) == max_seq_length
    assert len(segment_ids_) == max_seq_length

    return input_ids_, input_masks_, segment_ids_

def convert_examples_to_features(examples, max_seq_length, max_para_num, graph_retriever_config, tokenizer, train = False):
    """Loads a data file into a list of `InputBatch`s."""

    if not train and graph_retriever_config.db_save_path is not None:
        max_para_num = graph_retriever_config.max_context_size
        graph_retriever_config.max_para_num = max(graph_retriever_config.max_para_num, max_para_num)
    
    max_steps = graph_retriever_config.max_select_num
    
    DUMMY = [0] * max_seq_length
    features = []

    logger.info('#### Converting examples to features... ####')
    for (ex_index, example) in enumerate(tqdm(examples, desc='Example')):
        tokens_q = tokenize_question(example.question, tokenizer)
        
        ##############
        # Short gold #
        ##############
        title2index = {}
        input_ids = []
        input_masks = []
        segment_ids = []

        # Append gold and non-gold paragraphs from context
        if train and graph_retriever_config.use_redundant and len(example.redundant_gold) > 0:
            if graph_retriever_config.use_multiple_redundant:
                titles_list = example.short_gold + [redundant[0] for redundant in example.all_redundant_gold] + list(example.context.keys())
            else:
                titles_list = example.short_gold + [example.redundant_gold[0]] + list(example.context.keys())
        else:
            titles_list = example.short_gold + list(example.context.keys())
        titles_list = [str(t) for t in titles_list] # in case some are not in str format yet
        for p in titles_list:

            if len(input_ids) == max_para_num:
                break 

            # Avoid appending gold paragraphs as negative
            # clsi: so negatives should always be after positives
            if p in title2index:
                continue

            # fullwiki eval
            # Gold paragraphs are not always in context
            if not train and graph_retriever_config.open and p not in example.context:
                continue
            
            title2index[p] = len(title2index)
            example.title_order.append(p)
            p = example.context[p]

            input_ids_, input_masks_, segment_ids_ = tokenize_paragraph(p, tokens_q, max_seq_length, tokenizer)
            input_ids.append(input_ids_)
            input_masks.append(input_masks_)
            segment_ids.append(segment_ids_)
            
        assert len(input_ids) <= max_para_num
        
        num_paragraphs = len(input_ids)
        num_steps = len(example.short_gold)+1 # 1 for EOE

        ## clsi: perhaps add a truncation?
        if train:
            assert num_steps <= max_steps, print (num_steps, max_steps)
        
        output_masks = [([1.0] * len(input_ids) + [0.0] * (max_para_num - len(input_ids) + 1)) for _ in range(max_para_num + 2)]

        ## ignore open settings stuff for ambigqa
        if (not train) and graph_retriever_config.open:
            assert len(example.context) == num_paragraphs_no_links
            for i in range(len(output_masks[0])):
                if i >= num_paragraphs_no_links:
                    output_masks[0][i] = 0.0
        
        for i in range(len(input_ids)):
            output_masks[i+1][i] = 0.0            

        if train:
            size = num_steps-1

            for i in range(size):
                for j in range(size):
                    if i != j:
                        output_masks[i][j] = 0.0

            for i in range(size):
                output_masks[size][i] = 0.0
                        
            for i in range(max_steps):
                if i > size:
                    for j in range(len(output_masks[i])):
                        output_masks[i][j] = 0.0

            # # Use REDUNDANT setting
            # # Avoid treating the redundant paragraph as a negative example at the first step
            # if graph_retriever_config.use_redundant and len(example.redundant_gold) > 0:
            #     if graph_retriever_config.use_multiple_redundant:
            #         for redundant in example.all_redundant_gold:
            #             output_masks[0][title2index[redundant[0]]] = 0.0
            #     else:
            #         output_masks[0][title2index[example.redundant_gold[0]]] = 0.0
                    
        padding = [DUMMY] * (max_para_num - len(input_ids))
        input_ids += padding
        input_masks += padding
        segment_ids += padding

        # print ("num_para", num_paragraphs)
        # print ("num_steps", num_steps)
        # print ("output_masks", output_masks)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_masks=input_masks,
                              segment_ids=segment_ids,
                              output_masks = output_masks,
                              num_paragraphs = num_paragraphs,
                              num_steps = num_steps,
                              ex_index = ex_index))

        if not train or not graph_retriever_config.use_redundant or len(example.redundant_gold) == 0:
            continue


        # ##################
        # # Redundant gold #
        # ##################
        # for redundant_gold in example.all_redundant_gold:
        #     hist = set()
        #     input_ids_r = []
        #     input_masks_r = []
        #     segment_ids_r = []

        #     # Append gold and non-gold paragraphs from context
        #     for p in redundant_gold + list(example.context.keys()):

        #         if len(input_ids_r) == max_para_num:
        #             break

        #         #assert p in title2index
        #         if p not in title2index:
        #             assert p not in redundant_gold
        #             continue

        #         if p in hist:
        #             continue
        #         hist.add(p)

        #         index = title2index[p]
        #         input_ids_r.append(input_ids[index])
        #         input_masks_r.append(input_masks[index])
        #         segment_ids_r.append(segment_ids[index])

        #     # Open-domain setting (mainly for HotpotQA fullwiki)
        #     if graph_retriever_config.open:

        #         for p in title2index:

        #             if len(input_ids_r) == max_para_num:
        #                 break

        #             if p in hist:
        #                 continue
        #             hist.add(p)

        #             index = title2index[p]
        #             input_ids_r.append(input_ids[index])
        #             input_masks_r.append(input_masks[index])
        #             segment_ids_r.append(segment_ids[index])

        #     assert len(input_ids_r) <= max_para_num

        #     num_paragraphs_r = len(input_ids_r)
        #     num_steps_r = len(redundant_gold)+1

        #     assert num_steps_r <= max_steps

        #     output_masks_r = [([1.0] * len(input_ids_r) + [0.0] * (max_para_num - len(input_ids_r) + 1)) for _ in range(max_para_num + 2)]

        #     size = num_steps_r-1

        #     for i in range(size):
        #         for j in range(size):
        #             if i != j:
        #                 output_masks_r[i][j] = 0.0

        #         if i > 0:
        #             output_masks_r[i][0] = 1.0

        #     for i in range(size): #size-1
        #         output_masks_r[size][i] = 0.0

        #     for i in range(max_steps):
        #         if i > size:
        #             for j in range(len(output_masks_r[i])):
        #                 output_masks_r[i][j] = 0.0

        #     padding = [DUMMY] * (max_para_num - len(input_ids_r))
        #     input_ids_r += padding
        #     input_masks_r += padding
        #     segment_ids_r += padding

        #     features.append(
        #             InputFeatures(input_ids=input_ids_r,
        #                           input_masks=input_masks_r,
        #                           segment_ids=segment_ids_r,
        #                           output_masks = output_masks_r,
        #                           num_paragraphs = num_paragraphs_r,
        #                           num_steps = num_steps_r,
        #                           ex_index = None))

        #     if not graph_retriever_config.use_multiple_redundant:
        #         break

    logger.info('Done!')
    return features

def save(model, output_dir, suffix):
    logger.info('Saving the checkpoint...')
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, "pytorch_model_"+suffix+".bin")

    status = True
    try:
        torch.save(model_to_save.state_dict(), output_model_file)
    except:
        status = False

    if status:
        logger.info('Successfully saved!')
    else:
        logger.warn('Failed!')
        
    return status

def load(output_dir, suffix):
    file_name = 'pytorch_model_' + suffix +'.bin'
    output_model_file = os.path.join(output_dir, file_name)
    return torch.load(output_model_file)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x
