######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################
import theano
import numpy as np
import os
import operator
from math import log, log10, exp, pow
from copy import deepcopy
import sys
import random
import time
import itertools
import pickle as pk
from ast import literal_eval
import gc

from navnnsds import NNSDS

from utils.tools import setWordVector
from utils.nlp import normalize
from utils.bleu import sentence_bleu_4

from loader.NavDataReader import *
from loader.GentScorer import *

from ConfigParser import SafeConfigParser

from api.Interact import Interact

theano.gof.compilelock.set_lock_status(False)

class NNDial(object):
    '''
    Main interface class for the model. This class takes charge of save/load
    hyperparameters from the config file and trained models. It delegates the
    data preprocessing to DataReader module and delegates the learning to
    NNSDS module. It implements training based on early stopping and testing
    and interactive interfaces.
    '''
    #######################################################################
    # all variables that needs to be save and load from model file, indexed
    # by their names
    #######################################################################
    learn_vars  = ['self.lr','self.lr_decay','self.stop_count','self.l2',
                   'self.seed','self.min_impr','self.debug','self.llogp',
                   'self.grad_clip','self.valid_logp','self.params',
                   'self.cur_stop_count','self.learn_mode']
    file_vars   = ['self.corpusfile', 'self.semidictfile',
                   'self.ontologyfile','self.modelfile']
    data_vars   = ['self.split','self.percent','self.shuffle','self.lengthen']
    gen_vars    = ['self.topk','self.beamwidth','self.verbose',
                   'self.repeat_penalty','self.token_reward']
    n2n_vars    = ['self.enc','self.trk','self.dec']
    enc_vars    = ['self.vocab_size','self.input_hidden']
    dec_vars    = ['self.output_hidden','self.seq_wvec_file','self.dec_struct']
    trk_vars    = ['self.trkinf','self.trkreq','self.belief','self.inf_dimensions',
                   'self.req_dimensions','self.trk_enc','self.trk_wvec_file',
                   'self.task_size']
    ply_vars    = ['self.policy','self.latent']

    def __init__(self,config=None,opts=None):

        if config==None and opts==None:
            print "Please specify command option or config file ..."
            return

        # config parser
        parser = SafeConfigParser()
        parser.read(config)

        # model file name
        self.modelfile = parser.get('file','model')
        # get current mode from command argument
        if opts:
            if opts.mode == 'test-oracle':
                opts.mode = 'test'
            self.mode = opts.mode  # mode: trk|enc|dec|all
        # loading pretrained model if any
        if os.path.isfile(self.modelfile):
            # if model file already exists
            if not opts:  self.loadNet(parser,None)
            else:         self.loadNet(parser,opts.mode)
        else: # init network from scratch
            self.initNet(config,opts)
            self.initBackupWeights()

    def initNet(self,config,opts=None):

        print '\n\ninit net from scratch ... '

        # config parser
        parser = SafeConfigParser()
        parser.read(config)

        # Setting default learn from config file
        self.debug          = parser.getboolean('learn','debug')
        if self.debug:
            print 'loading model settings from config file ...'
        self.lr             = parser.getfloat('learn','lr')
        self.lr_decay       = parser.getfloat('learn','lr_decay')
        self.stop_count     = parser.getint('learn','stop_count')
        self.cur_stop_count = parser.getint('learn','cur_stop_count')
        self.l2             = parser.getfloat('learn','l2')
        self.seed           = parser.getint('learn','random_seed')
        self.min_impr       = parser.getfloat('learn','min_impr')
        self.llogp          = parser.getfloat('learn','llogp')
        self.grad_clip      = parser.getfloat('learn','grad_clip')

        # Setting file paths
        # self.dbfile         = parser.get('file','db')
        self.ontologyfile   = parser.get('file','ontology')
        self.corpusfile     = parser.get('file','corpus')
        self.semidictfile   = parser.get('file','semidict')

        # setting data manipulations
        self.split          = literal_eval(parser.get('data','split'))
        self.lengthen       = parser.getint('data','lengthen')
        self.shuffle        = parser.get('data','shuffle')
        self.percent        = parser.get('data','percent')

        # Setting generation specific parameters
        self.verbose        = parser.getint('gen','verbose')
        self.topk           = parser.getint('gen','topk')
        self.beamwidth      = parser.getint('gen','beamwidth')
        self.repeat_penalty = parser.get('gen','repeat_penalty')
        self.token_reward   = parser.getboolean('gen','token_reward')

        # setting n2n components
        self.enc            = parser.get('n2n','encoder')
        self.trk            = parser.get('n2n','tracker')
        self.dec            = parser.get('n2n','decoder')

        # setting encoder structure
        self.input_hidden   = parser.getint('enc','ihidden')

        # setting policy  structure
        self.policy         = parser.get('ply','policy')
        self.latent         = parser.getint('ply','latent')\
                              if self.policy=='latent' else 0

        # setting decoder structure
        self.output_hidden  = parser.getint('dec','ohidden')
        self.seq_wvec_file  = parser.get('dec','wvec')
        self.dec_struct     = parser.get('dec','struct')
        self.use_snapshot   = parser.getboolean('dec','snapshot')

        # setting tracker structure
        self.trkinf         = parser.getboolean('trk','informable')
        self.trkreq         = parser.getboolean('trk','requestable')
        self.belief         = parser.get('trk','belief')
        self.trk_enc        = parser.get('trk','trkenc')
        self.trk_wvec_file  = parser.get('trk','wvec')

        # setting learnable parameters
        self.learn_mode     = parser.get('mode','learn_mode')

        # set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        np.set_printoptions(precision=4)

        # setting data reader, processors, and lexicon
        # 1. split the corpus data into (train, valid, test)
        # 2. prepared all possible slot names & values
        #    (informables and requestables, with delexicalation
        #     and lexicalization using semidict and mappings)
        # 3. format the DB by creating indexes and hash tables
        # 4. Load the vocabulary from the DB & ontology
        # 5. Load the vocabulary from the dialogue data itself
        # 6. Load the dialogue
        # 7. Load the semantics
        # 8. Parse goals
        self.reader = DataReader(
            self.corpusfile, self.semidictfile, self.ontologyfile,  # remove 2nd db file
            self.split, self.lengthen, self.percent,
            self.shuffle, self.trk_enc, self.verbose, opts.mode, self.policy,  # att
            self.latent
        )

        # network size according to dataset
        self.vocab_size     = len(self.reader.vocab)
        self.inf_dimensions = self.reader.infoseg
        self.req_dimensions = self.reader.reqseg
        self.task_size = len(self.reader.refvs) - 1

        # logp for validation set
        self.valid_logp = 0.0

        # start setting networks
        self.ready()

    def ready(self):

    #################################################################
    ################### THEANO CONFIGURATION ########################
    #################################################################

        # initialise network model
        if self.debug:
            print 'setting network structures using theano variables ...'
        self.model = NNSDS(self.enc, self.dec, self.policy,
            self.trk, self.trkinf, self.trkreq, self.belief, self.trk_enc,
            self.use_snapshot, self.dec_struct, self.vocab_size,
            self.input_hidden, self.output_hidden,
            self.inf_dimensions, self.req_dimensions,
            self.task_size, self.grad_clip,
            self.learn_mode, len(self.reader.snapshots), self.latent)

        # setput theano variables
        self.model.config_theano()
        if self.debug:
            numofparams, trainable = self.model.numOfParams()
            print '\t\tnumber of parameters : %8d' % numofparams
            print '\t\tnumber of training parameters : %8d' % trainable

    #################################################################
    ############################ END ################################
    #################################################################
    def testNet(self, oracle=False):

        # testing generation
        np.random.seed(self.seed)
        if self.debug:
            print 'generating dialogue responses for trained network ...'

        # evaluator
        bscorer = BLEUScorer()
        # parallel_corpus = []
        best_corpus = []

        # load testing data
        testset = self.reader.iterate(mode=self.mode)  # valid or test

        # statistics for calulating semi performance
        stats = self._statsTable()
        start_time = time.time()

        # gate stats
        gstats = np.zeros((4))
        num_sent = 0.0

        # for each dialog
        for cnt in range(len(testset)):
            # initial state
            if self.verbose>0:
                print '='*25 + ' Dialogue '+ str(cnt) +' '+ '='*28
            #print '##############################################################'
            # read one example
            source, source_len, masked_source, masked_source_len,\
            target, target_len, masked_target, masked_target_len,\
            snapshot, change, goal, inf_trk_label, req_trk_label,\
            db_degree, new_task, srcfeat, tarfeat,\
            ref_trk_label, ref_mentions, refsrcfeat, reftarfeat,\
            finished, utt_group = testset[cnt]

            # initial selection
            selected_venue  = -1
            venue_offered   = None
            prev_correct = False
            task_i = 0
            ref_correct = True
            prev_ref_correct = True

            # initial belief
            #print 'initial belief size:', self.inf_dimensions[-1]
            flatten_belief_tm1 = np.zeros((self.inf_dimensions[-1]))
            for i in range(len(self.inf_dimensions)-1):
                flatten_belief_tm1[self.inf_dimensions[i+1]-1] = 1.0
                # print 'setting belief of', self.reader.infovs[self.inf_dimensions[i+1]-1], ' to 1'
                # for every if s = none, set to 1 initially, for all informable

            # initial ref target feat
            # ref_mentions_t = np.zeros((self.task_size + 1))
            #refmentions_t[-1] = 1.0
            #reftarfeat_tm1 = np.zeros((self.task_size + 1))

            reqs = []
            generated_utt_tm1 = ''  # previous masked generated utterance
            # for each turn
            for t in range(len(source)):
                if self.verbose>0:
                    print '-'*28 + ' Turn '+ str(t) +' '+ '-'*28
                # extract source and target sentence for that turn
                source_t        = source[t][:source_len[t]]
                masked_source_t = masked_source[t][:masked_source_len[t]]
                masked_target_t = masked_target[t][:masked_target_len[t]]
                # this turn features
                srcfeat_t   = srcfeat[t]
                refsrcfeat_t = refsrcfeat[t]
                ref_mentions_t = ref_mentions[t]

                # previous target
                masked_target_tm1, target_tm1, starpos_tm1, vtarpos_tm1, offer = \
                    self.reader.extractSeq(generated_utt_tm1,type='target')
                tarfeat_tm1 = [starpos_tm1,vtarpos_tm1]
                if t == 0:
                    _, reftarfeat_tm1 = self.reader.extractRef(generated_utt_tm1, ref_mentions_t)
                #else:
                #    reftarfeat_tm1 = reftarfeat[t-1]
                # _, reftarfeat_tm1 = self.reader.extractRef(generated_utt_tm1, ref_mentions_t)
                #ref_mentions_t = self.reader.updateRef(generated_utt_tm1)

                # utterance preparation
                source_utt = ' '.join([self.reader.vocab[w] for w in source_t])
                masked_source_utt= ' '.join([self.reader.vocab[w]
                        for w in masked_source_t])
                masked_target_utt= ' '.join([self.reader.vocab[w]
                        for w in masked_target_t])

                # read and understand user sentence
                masked_intent_t = self.model.read(masked_source_t) # bidirectional encode context

                # task reference tracking
                task_ref_t, ref_belief = self.model.track_ref(ref_mentions_t,
                                                              masked_source_t,
                                                              masked_target_tm1,
                                                              refsrcfeat_t,
                                                              reftarfeat_tm1)

                # belief tracking
                full_belief_t, belief_t = self.model.track(
                        flatten_belief_tm1, masked_source_t, masked_target_tm1,
                        srcfeat_t, tarfeat_tm1)
                flatten_belief_t = np.concatenate(full_belief_t,axis=0)

                # add task ref into belief for attention scoring
                if self.policy == 'attention':
                    belief_t.append(ref_belief)
                    # belief_t = belief_t.concat(task_ref_t)

                # search DB
                # db_degree_t, query = self._searchDB(flatten_belief_t)
                # based in flatten_belief, retrieve db_degree_t
                # which is a list of [0, 1, 0, ... x combinations  | 6 OHE vector ]
                # but now our db_degree is just [0, 1] or [1, 0]
                # being                          no, and yes
                # suppose to use API to search, given flatten belief t
                # but for now, just use the same degree as given in the dataset
                db_degree_t = db_degree[t]
                # score table
                scoreTable = self._genScoreTable(full_belief_t)
                # generation
                generated,sample_t,_ = self.model.talk(
                        masked_intent_t,belief_t, db_degree_t,
                        masked_source_t, masked_target_t, scoreTable)

                # choose venue
                #venues = [i for i, e in enumerate(db_degree_t[:-6]) if e != 0 ]
                # suppose to get the venues from the api list
                venues = ['test']
                # keep the current venue
                if selected_venue in venues: pass
                else: # choose the first match as default index
                    if len(venues)!=0:  selected_venue = random.choice(venues)
                    # no matched venues
                    else: selected_venue = None

                # lexicalise generated utterance
                generated_utts = []
                for gen in generated:
                    generated_utt = ' '.join([self.reader.vocab[g] for g in gen[0]])
                    generated_utts.append(generated_utt)
                generated_utt = generated_utts[0]   # ??? why not just get the first one?

                # calculate semantic match rate
                twords = [self.reader.vocab[w] for w in masked_target_t]
                for gen in generated:
                    gwords = [self.reader.vocab[g] for g in gen[0]]
                    for gw in gwords:
                        if gw.startswith('[VALUE_') or gw.startswith('[SLOT_'):
                            if gw in twords: # match target semi token
                                stats['approp'][0] += 1.0
                            stats['approp'][1] += 1.0
                    #gstats += np.mean( np.array(gen[2][1:]),axis=0 )
                    num_sent += 1

                # update history belief - but only for informables
                flatten_belief_tm1 = flatten_belief_t[:self.inf_dimensions[-1]]

                # for calculating success: check requestable slots match
                #requestables = ['phone','address','postcode','food','area','pricerange']
                requestables = self.reader.s2v['requestable'].keys()
                for requestable in requestables:
                    if '[VALUE_'+requestable.upper()+']' in generated_utt and prev_ref_correct:
                        reqs.append(self.reader.reqs.index(requestable+'=exist'))
                prev_ref_correct = ref_correct
                # check offered venue
                if '[VALUE_PLACE]' in generated_utt and selected_venue!=None:
                    #venue_offered = self.reader.db2inf[selected_venue]
                    venue_offered = 'test'

                ############################### debugging ############################
                if self.verbose>0:
                    print 'User Input   :\t%s'% source_utt
                    print 'Masked Input :\t%s'% masked_source_utt
                    # slot_features, value_features = srcfeat_t
                    # print slot_features, len(slot_features)
                    # for i, sf in enumerate(slot_features):
                    #     if sf[0] != -1:
                    #         print 'slot present :\t%s' % masked_source_utt.split()[sf[0]], 'for:%s' % self.reader.allvs[i]
                    # #print value_features, len(value_features)
                    # for i, vf in enumerate(value_features):
                    #     if vf[0] != -1:
                    #         print 'value present :\t%s' % masked_source_utt.split()[vf[0]], 'for:%s' % self.reader.allvs[i]
                    # assert(False)
                    # print

                if self.trk=='rnn' and self.trkinf==True:
                    if self.verbose>1:
                        print 'Belief Tracker :'
                        print '  | %25s%13s%35s|' % ('','Informable','')
                        print '  | %25s\t%5s\t%35s |' % ('Prediction','Prob.','Ground Truth')
                        print '  | %25s\t%5s\t%35s |' % ('------------','-----','------------')
                    all_correct = True
                    for i in range(len(self.inf_dimensions)-1):
                        bn = self.inf_dimensions[i]
                        bidx = np.argmax(np.array(full_belief_t[i]))+bn
                        yidx = np.argmax(np.array(inf_trk_label[t][bn:self.inf_dimensions[i+1]]))+bn
                        psem = self.reader.infovs[bidx]
                        ysem = self.reader.infovs[yidx]
                        prob = full_belief_t[i][np.argmax(np.array(full_belief_t[i]))]
                        #print '%20s\t%.3f\t%20s' % (psem,prob,ysem)
                        if self.verbose>1:
                            print '  | %25s\t%.3f\t%35s |' % (psem,prob,ysem)

                        # counting stats
                        slt,val = ysem.split('=')
                        if 'none' not in ysem:
                            if psem==ysem: # true positive
                                stats['informable'][slt][0] += 1.0
                            else: # false negative
                                stats['informable'][slt][1] += 1.0
                                if ysem != 'any':
                                    all_correct = False
                        else:
                            if psem==ysem: # true negative
                                stats['informable'][slt][2] += 1.0
                            else: # false positive
                                stats['informable'][slt][3] += 1.0
                                if ysem != 'any':
                                    all_correct = False

                print '  | %25s%13s%35s|' % ('','Task Reference','')
                print '  | %25s\t%5s\t%35s |' % ('Prediction','Prob.','Ground Truth')
                print '  | %25s\t%5s\t%35s |' % ('------------','-----','------------')
                bidx = np.argmax(np.array(task_ref_t))
                yidx = np.argmax(ref_trk_label[t])
                psem = self.reader.refvs[bidx]
                ysem = self.reader.refvs[yidx]
                prob = task_ref_t[bidx]
                print '  | %25s\t%.3f\t%35s |' % (psem,prob,ysem)
                if oracle:
                    ref_correct = True
                else:
                    ref_correct = ysem == psem
                if 'none' not in ysem:
                    if psem==ysem:
                        stats['informable']['task_reference'][0] += 1.0
                    else:
                        stats['informable']['task_reference'][1] += 1.0
                else:
                    if psem==ysem: # true negative
                        stats['informable']['task_reference'][2] += 1.0
                    else: # false positive
                        stats['informable']['task_reference'][3] += 1.0

                if self.trk=='rnn' and self.trkreq==True:
                    if self.verbose>1:
                        print '  | %25s%13s%35s|' % ('','Requestable','')
                        print '  | %25s\t%5s\t%35s |' % ('Prediction','Prob.','Ground Truth')
                        print '  | %25s\t%5s\t%35s |' % ('------------','-----','------------')
                    infbn = 3 if self.trkinf else 0
                    for i in range(len(self.req_dimensions)-1):
                        bn = self.req_dimensions[i]
                        ysem = self.reader.reqs[np.argmax(np.array(\
                                req_trk_label[t][bn:self.req_dimensions[i+1]+bn]))+bn]
                        psem = self.reader.reqs[ \
                            np.argmax(np.array(full_belief_t[infbn+i])) +\
                            self.req_dimensions[i] ]
                        prob = np.max(np.array(full_belief_t[infbn+i]))
                        if self.verbose>1:
                            print '  | %25s\t%.3f\t%35s |' % (psem,prob,ysem)

                        # counting stats
                        slt,val = ysem.split('=')
                        if slt+'=exist'==ysem:  # truth is exits
                            if psem==ysem and ref_correct: # true positive
                                stats['requestable'][slt][0] += 1.0
                            else: # false negative
                                stats['requestable'][slt][1] += 1.0
                        else:  # truth: does not exists
                            if psem==ysem: # true negative
                                stats['requestable'][slt][2] += 1.0
                            else: # false positive
                                stats['requestable'][slt][3] += 1.0

                    # offer change tracker
                    # bn = self.req_dimensions[-1]
                    # psem = 0 if full_belief_t[-1][0]>=0.5 else 1
                    # ysem = np.argmax(change_label[t])
                    # if ysem==0:
                    #     if psem==ysem:
                    #         stats['requestable']['change'][0] += 1.0
                    #     else:
                    #         stats['requestable']['change'][1] += 1.0
                    # else:
                    #     if psem==ysem:
                    #         stats['requestable']['change'][2] += 1.0
                    #     else:
                    #         stats['requestable']['change'][3] += 1.0
                    #prdtvenue = 'venue=change' if psem==0 else 'venue=not change'
                    #truevenue = 'venue=change' if ysem==0 else 'venue=not change'
                    #prob      = full_belief_t[-1][0] if psem==0 else 1-full_belief_t[-1][0]
                    #if self.verbose>1:
                    #    print '  | %16s\t%.3f\t%20s |' % (prdtvenue,prob,truevenue)

                if self.verbose>0:
                    match_number = str(db_degree_t[1])
                    #match_number = np.argmax(np.array(db_degree_t[-6:]))
                    #match_number = str(match_number) if match_number<5 else '>5'
                    print
                    print 'DB Match     : %s' % match_number
                    print
                    print 'Generated    : %s' % generated_utts[0]
                    for g in generated_utts[1:]:
                        print '             : %s'% g
                    print
                    print 'Ground Truth : %s' % masked_target_utt
                    print
                #raw_input()
                ############################### debugging ############################
                generated_utt_tm1 = masked_target_utt
                reftarfeat_tm1 = reftarfeat[t]

                # parallel_corpus.append([generated_utts,[masked_target_utt]])
                # not used at all
                best_corpus.append([[generated_utt],[masked_target_utt]])

                new = new_task[t]
                if new == 1:
                    new = True
                else:
                    new = False
                print 'New Label : %s' % new_task[t]
                if new:
                    if venue_offered != None:
                        if prev_correct:
                            stats['vmc'] += 1.0

                        truth_req = goal[task_i][1].nonzero()[0].tolist()
                        # truth_req is a list of all req types: 0 if exist, 1 if not exist
                        # reqs is a list of indexes of requests that values exist in the system turn
                        for req in reqs:
                            if req in truth_req:
                                stats['success_tp'] += 1.0
                            else:
                                stats['success_fp'] += 1.0
                        for req in truth_req:
                            if req not in reqs:
                                stats['success_fn'] += 1.0

                        #if set(reqs).issuperset(set(goal[task_i][1].nonzero()[0].tolist())):
                        if set(reqs).issuperset(set(truth_req)):
                            stats['success'] += 1.0

                    task_i += 1
                    stats['vmc_total'] += 1.0
                    venue_offered = None
                prev_correct = all_correct

            # at the end of the dialog, calculate goal completion rate
            # for now, just calculate the last task
            # TODO: do an average for all tasks; need to add in the 'new' param
            # if venue_offered != None: # and finished:
            #     #stats['vmc'] += 1.0
            #     truth_req = goal[-1][1].nonzero()[0].tolist()
            #     # truth_req is a list of all req: 0 if exist, 1 if not exist
            #     # reqs is a list of indexes that exist in the dialogue
            #     for req in reqs:
            #         if req in truth_req:
            #             stats['success_tp'] += 1.0
            #         else:
            #             stats['success_fp'] += 1.0
            #     for req in truth_req:
            #         if req not in reqs:
            #             stats['success_fn'] += 1.0
            #
            #     if set(reqs).issuperset(set(goal[-1][1].nonzero()[0].tolist())):
            #         stats['success'] += 1.0
               # if set(venue_offered).issuperset(set(goal[0].nonzero()[0].tolist())):
               #     stats['vmc'] += 1.0
               #     if set(reqs).issuperset(set(goal[1].nonzero()[0].tolist())):
               #         stats['success'] += 1.0

        precision = stats['success_tp'] / (stats['success_tp'] + stats['success_fp'])
        recall =  stats['success_tp'] / (stats['success_tp'] + stats['success_fn'])
        success_f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # evaluation result
        print 80*'#'
        print 35*'#' + '  Metrics ' + 35*'#'
        print 80*'#'
        #print 'Venue Match Rate     : %.1f%%' % (100*stats['vmc']/float(len(testset)))
        print 'Venue Match Rate     : %.1f%%' % (100*stats['vmc']/stats['vmc_total'])
        print 'Task Success Rate    : %.1f%%' % (100*stats['success']/stats['vmc_total'])
        print 'Request Rate         : %.2f%%' % (100*recall)
        print 'Success F1           : %.2f%%' % (100*success_f1)
        if self.dec!='none':
            print 'BLEU                 : %.4f' % (bscorer.score(best_corpus))
            print 'Semantic Match       : %.1f%%' % (100*stats['approp'][0]/stats['approp'][1])
        print 35*'#' + ' Trackers ' + 35*'#'
        print '---- Informable  '+ 63*'-'
        #infslots = ['area','food','pricerange']
        infslots = self.reader.s2v['informable'].keys() + ['task_reference']
        joint = [0.0 for x in range(4)]
        for i in range(len(infslots)):
            s = infslots[i]
            joint = [joint[i]+stats['informable'][s][i] for i in range(len(joint))]
            tp, fn, tn, fp = stats['informable'][s]
            p = tp/(tp+fp)*100
            r = tp/(tp+fn)*100
            total = tp+tn+fp+fn
            ac= (tp+tn)/(total)*100
            print '%12s :\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t| %d \t' %\
                (s, p, r, 2*p*r/(p+r), ac, total)
        tp, fn, tn, fp = joint
        p = tp/(tp+fp)*100
        r = tp/(tp+fn)*100
        total = tp+tn+fp+fn
        ac= (tp+tn)/(total)*100
        print 80*'-'
        print '%12s :\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t| %d \t|' %\
                ('joint', p, r, 2*p*r/(p+r), ac, total)
        print '---- Requestable '+ 63*'-'
        reqslots = self.reader.s2v['requestable'].keys()
        #reqslots = ['area','food','pricerange','address','postcode','phone']#,'change']
        joint = [0.0 for x in range(4)]
        for i in range(len(reqslots)):
            s = reqslots[i]
            joint = [joint[i]+stats['requestable'][s][i] for i in range(len(joint))]
            tp, fn, tn, fp = stats['requestable'][s]
            p = tp/(tp+fp)*100
            r = tp/(tp+fn)*100
            total = tp+tn+fp+fn
            ac= (tp+tn)/(total)*100
            print '%12s :\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t| %d \t|' %\
                (s, p, r, 2*p*r/(p+r), ac, total)
        tp, fn, tn, fp = joint
        p = tp/(tp+fp)*100
        r = tp/(tp+fn)*100
        total = tp+tn+fp+fn
        ac= (tp+tn)/(total)*100
        print 80*'-'
        print '%12s :\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t| %2.2f%%\t| %d\t' %\
                ('joint', p, r, 2*p*r/(p+r), ac, total)
        print 80*'-'
        print '%12s :\t| %7s\t| %7s\t| %7s\t| %7s\t| %7s\t |' %\
                ('Metrics', 'Prec.', 'Recall', 'F-1', 'Acc.', 'Total')
        print 80*'#'


    def trainNet(self):

        #if self.debug:
        print 'start network training ...'

        ######## training with early stopping #########
        epoch = 0

        while True:
            # training phase
            tic = time.time()
            epoch += 1
            train_logp = 0.0
            num_dialog = 0.0
            while True:
                data = self.reader.read(mode='train')
                # end of dataset
                if data==None:
                    break
                # read one example
                source, source_len, masked_source, masked_source_len,\
                target, target_len, masked_target, masked_target_len,\
                snapshot, change, goal, inf_trk_label, req_trk_label,\
                db_degree, new_task, srcfeat, tarfeat,\
                ref_trk_label, ref_mentions, refsrcfeat, reftarfeat,\
                finished, utt_group = data

                # TODO: improve, default parameters for success
                success_rewards = [0. for i in range(len(source))]
                sample = np.array([0 for i in range(len(source))],dtype='int32')

                # set regularization
                loss, prior_loss, posterior_loss, base_loss, \
                posterior, sample, reward, baseline, debugs =\
                        self.model.train(
                            source, target, source_len, target_len,
                            masked_source, masked_target,
                            masked_source_len, masked_target_len,
                            utt_group, snapshot, success_rewards, sample,
                            change, inf_trk_label, req_trk_label, db_degree,
                            srcfeat, tarfeat,
                            ref_trk_label, ref_mentions, refsrcfeat, reftarfeat,
                            self.lr, self.l2)
                if self.policy=='latent':
                    train_logp+=-np.sum(loss)-0.1*np.sum(prior_loss)
                else:
                    train_logp+=-np.sum(loss)

                num_dialog+=1

                if self.debug and num_dialog%1==0:
                    print 'Finishing %8d dialog in epoch %3d\r' % \
                            (num_dialog,epoch),
                    sys.stdout.flush()

            sec = (time.time()-tic)/60.0
            if self.debug:
                print 'Epoch %3d, Alpha %.6f, TRAIN entropy:%.2f, Time:%.2f mins,' %\
                        (epoch, self.lr, -train_logp/log10(2)/num_dialog, sec),
                sys.stdout.flush()

            # validation phase
            self.valid_logp = 0.0
            num_dialog = 0.0
            while True:
                data = self.reader.read(mode='valid')
                # end of dataset
                if data==None:
                    break
                # read one example
                source, source_len, masked_source, masked_source_len,\
                target, target_len, masked_target, masked_target_len,\
                snapshot, change, goal, inf_trk_label, req_trk_label,\
                db_degree, new_task, srcfeat, tarfeat,\
                ref_trk_label, ref_mentions, refsrcfeat, reftarfeat,\
                finished, utt_group = data

                # TODO: improve, default parameters for success
                success_rewards = [0. for i in range(len(source))]
                sample = np.array([0 for i in range(len(source))],dtype='int32')

                # validating
                loss, prior_loss, _ = self.model.valid(
                        source, target, source_len, target_len,
                        masked_source, masked_target,
                        masked_source_len, masked_target_len,
                        utt_group, snapshot, success_rewards, sample,
                        change, inf_trk_label, req_trk_label, db_degree,
                        srcfeat, tarfeat,
                        ref_trk_label, ref_mentions, refsrcfeat, reftarfeat)
                if self.policy=='latent':
                    self.valid_logp += -np.sum(loss)-0.1*np.sum(prior_loss)
                else:
                    self.valid_logp += -np.sum(loss)

                num_dialog  += 1

            if self.debug:
                print 'VALID entropy:%.2f'%-(self.valid_logp/log10(2)/num_dialog)

            # decide to throw/keep weights
            if self.valid_logp < self.llogp:   # if better than previous
                self.getBackupWeights()        # save params
            else:
                self.setBackupWeights()        # else get best params
            self.saveNet()  # no need to save, only save when done (optimize)

            # learning rate decay
            if self.cur_stop_count>=self.stop_count:
                self.lr *= self.lr_decay

            # early stopping
            if self.valid_logp*self.min_impr<self.llogp:
                if self.cur_stop_count<self.stop_count:
                    self.lr *= self.lr_decay
                    self.cur_stop_count += 1
                else:
                    self.saveNet()
                    print 'Training completed.'
                    break

            self.llogp = self.valid_logp    # save lowest logp loss

            # garbage collection
            tic = time.time()
            cnt = gc.collect()
            sec = (time.time()-tic)/60.0
            print 'Garbage collection:\t%4d objs\t%.2f mins' % (cnt,sec)

    # sampling dialogue during training to get task success information
    def sampleDialog(self, data):
        # unzip the dialogue
        source, source_len, masked_source, masked_source_len,\
        target, target_len, masked_target, masked_target_len,\
        snapshot, change, goal, inf_trk_label, req_trk_label,\
        db_degree, new_task, srcfeat, tarfeat,\
        ref_trk_label, ref_mentions, refsrcfeat, reftarfeat,\
        finished, utt_group = data

        # for calculating success: check requestable slots match
        #requestables = ['phone','address','postcode']
        requestables = self.reader.s2v['requestable'].keys()   #
        offer_per_turn  = []
        request_per_turn= []
        target_sents = []
        # for each turn
        for t in range(len(masked_target)):
            sent_t = [self.reader.vocab[w] for w in
                    masked_target[t][:masked_target_len[t]]][1:-1]
            target_sents.append(sent_t)
            # decide offer or not
            if '[VALUE_PLACE]' in sent_t: offer_per_turn.append(True)
            else: offer_per_turn.append(False)
            # compute requestable matches
            requests = []
            for requestable in requestables:
                if '[VALUE_'+requestable.upper()+']' in sent_t:
                    requests.append(self.reader.reqs.index(requestable+'=exist'))
            request_per_turn.append(requests)

        # compute original success
        all_requests = np.sum(np.array([np.array(x[1]) for x in goal]))
        original_success = sum(offer_per_turn)>0 and \
                set(np.hstack(np.array(request_per_turn)).tolist()).issuperset(
                set(all_requests.nonzero()[0].tolist()))

        success_rewards = []
        samples= []
        gens = []
        # sample a sentence for each dialogue
        for t in range(len(masked_target)):
            # read
            masked_intent_t = self.model.read( masked_source[t][:masked_target_len[t]] )
            full_belief_t, belief_t = self.model.genPseudoBelief(
                    inf_trk_label[t], req_trk_label[t], change[t])
            # talk
            forced_sample = None if utt_group[t]==self.latent-1 else utt_group[t]
            generated, sample_t, prob_t = self.model.talk(
                    masked_intent_t,belief_t, db_degree[t],
                    masked_source[t][:masked_source_len[t]],
                    masked_target[t][:masked_target_len[t]],
                    None, forced_sample)
            sent_t = [self.reader.vocab[w] for w in generated[0][0][1:-1]]
            gens.append(' '.join(sent_t))

            # decide offer or not
            offer = deepcopy(offer_per_turn)
            if '[VALUE_PLACE]' in sent_t: offer[t] = True
            else: offer[t] = False
            # compute requestable matches
            requests = []
            for requestable in requestables:
                if '[VALUE_'+requestable.upper()+']' in sent_t:
                    requests.append(self.reader.reqs.index(requestable+'=exist'))
            req = deepcopy(request_per_turn)
            req[t] = requests
            all_requests = np.sum(np.array([np.array(x[1]) for x in goal]))
            success = sum(offer)>0 and \
                    set(np.hstack(np.array(req)).tolist()).issuperset(
                    set(all_requests.nonzero()[0].tolist()))
            bleu = sentence_bleu_4(gens[t].split(),[target_sents[t]])
            success_rewards.append(success-original_success+0.5*bleu-0.1)
            samples.append(sample_t[0])

        return np.array(success_rewards,dtype='float32'), samples, gens


    def trainNetRL(self):

        if self.debug:
            print 'start network RL training ...'

        ######## training with early stopping #########
        epoch = 0

        while True:
            # training phase
            tic = time.time()
            epoch += 1
            train_logp = 0.0
            num_dialog = 0.0
            # sampling from top 5
            self.model.policy.setSampleMode('prior',5)
            while True:
                self.model.loadConverseParams()
                data = self.reader.read(mode='train')
                # end of dataset
                if data==None:
                    break
                # read one example
                source, source_len, masked_source, masked_source_len,\
                target, target_len, masked_target, masked_target_len,\
                snapshot, change, goal, inf_trk_label, req_trk_label,\
                db_degree, new_task, srcfeat, tarfeat,\
                ref_trk_label, ref_mentions, refsrcfeat, reftarfeat,\
                finished, utt_group = data

                # sampling and compute success rate
                success_rewards, sample, gens =self.sampleDialog(data)

                # set regularization
                prior_loss, sample, prior = self.model.trainRL(
                            source, target, source_len, target_len,
                            masked_source, masked_target,
                            masked_source_len, masked_target_len,
                            utt_group, snapshot, success_rewards, sample,
                            change, inf_trk_label, req_trk_label, db_degree,
                            srcfeat, tarfeat,
                            ref_trk_label, ref_mentions, refsrcfeat, reftarfeat,
                            self.lr, self.l2)
                train_logp+=-np.sum(prior_loss)

                num_dialog+=1

                if self.debug and num_dialog%1==0:
                    print 'Finishing %8d dialog in epoch %3d\r' % \
                            (num_dialog,epoch),
                    sys.stdout.flush()

            sec = (time.time()-tic)/60.0
            if self.debug:
                print 'Epoch %3d, Alpha %.6f, TRAIN entropy:%.2f, Time:%.2f mins' %\
                        (epoch, self.lr, -train_logp/log10(2)/num_dialog, sec)
                sys.stdout.flush()
            self.saveNet()

            # force to stop after 3 epochs
            if epoch>=3:
                self.saveNet()
                print 'Training completed.'
                break

    # for interactive use
    def reply(self,user_utt_t,generated_tm1,selected_venue_tm1,
            venue_offered_tm1,flatten_belief_tm1):

        # initial belief
        if flatten_belief_tm1==[]:
            flatten_belief_tm1 = np.zeros((self.inf_dimensions[-1]))
            for i in range(len(self.inf_dimensions)-1):
                flatten_belief_tm1[self.inf_dimensions[i+1]-1] = 1.0

        # extract/index sequence and ngram features
        user_utt_t = normalize(user_utt_t)

        # extract sequencial features
        masked_source_t, source_t, ssrcpos, vsrcpos, _ = \
                self.reader.extractSeq(user_utt_t,type='source')
        masked_target_tm1, target_tm1, starpos, vtarpos, offer = \
                self.reader.extractSeq(generated_tm1,type='target')

        # position specific features
        srcfeat_t   = [ssrcpos,vsrcpos]
        tarfeat_tm1 = [starpos,vtarpos]

        # read sentences
        masked_intent_t = self.model.read( masked_source_t )
        full_belief_t, belief_t = self.model.track(
                flatten_belief_tm1, masked_source_t, masked_target_tm1,
                srcfeat_t, tarfeat_tm1 )
        flatten_belief_t = np.concatenate(full_belief_t,axis=0)
        # search
        db_degree_t, query = self._searchDB(flatten_belief_t)
        # additional scoring table
        scoreTable = self._genScoreTable(full_belief_t)
        # generation
        generated, sample_t,_ = self.model.talk(
                masked_intent_t, belief_t, db_degree_t, scoreTable)

        # lexicalise generated utterance
        generated_utts = []
        for gen in generated:
            generated_utt = ' '.join([self.reader.vocab[g] for g in gen[0]])
            generated_utts.append(generated_utt)
        gennerated_utt = random.choice(generated_utts)

        # choose venue and substitute slot tokens
        venues = [i for i, e in enumerate(db_degree_t[:-6]) if e != 0 ]
        if selected_venue_tm1 in venues: # previous venue still matches
            # keep the current venue
            selected_venue_t = selected_venue_tm1
            venue_offered_t= venue_offered_tm1
            # request alternative -> change offered venue
            if full_belief_t[-1][0]>=0.5:
              selected_venue_t = random.choice(venues)
              venue_offered_t = random.choice(
                        self.reader.idx2ent[selected_venue_t])
        else:
            if len(venues)!=0:
                # random choose from matched venues
                selected_venue_t= random.choice(venues)
                venue_offered_t = random.choice(
                        self.reader.idx2ent[selected_venue_t])
            else:
                # no matched venues
                selected_venue_t= -1
                venue_offered_t = {}

        # lexicalisation
        for cn in range(len(generated_utts)):
            words = generated_utts[cn].split()
            # count special tokens
            infslots = ['area','food','pricerange']
            specialtoks = {
                '[VALUE_FOOD]' :[0,
                    deepcopy(self.reader.s2v['informable']['food'])],
                '[VALUE_PRICERANGE]':[0,
                    deepcopy(self.reader.s2v['informable']['pricerange'])],
                '[VALUE_AREA]' :[0,
                    deepcopy(self.reader.s2v['informable']['area'])]}
            for w in words:
                if specialtoks.has_key(w):
                    specialtoks[w][0] += 1
            # replace slot tokens
            for i in range(len(words)):
                w = words[i]
                slot = w.split('_')[-1].replace(']','').lower()
                if w.startswith('[SLOT_'): # lexicalise slot token from semi dict
                    words[i] = random.choice(self.reader.semidict[slot])
                elif w.startswith('[VALUE_') and venue_offered_t=='None': # OOD situation
                    words[i] = self.reader.infovs[query[infslots.index(slot)]].split('=')[-1]
                elif w=='[VALUE_COUNT]': # replace count token
                    words[i] = str(sum(db_degree_t[:-6]))
                elif specialtoks.has_key(w) and specialtoks[w][0]>1:
                    choice = random.choice(specialtoks[w][1])
                    words[i] = choice
                    specialtoks[w][1].remove(choice)
                elif w[0]=='[' and w[-1]==']': # general case
                    if venue_offered_t=={}: # if no match, choose believe state to answer
                        if slot in infslots:
                            idx = infslots.index(slot)
                            words[i] = self.reader.infovs[query[idx]].split('=')[-1]
                    else: # if match, replaced with venue token
                        words[i] = venue_offered_t[slot].lower()
            # merge -s token
            j = 0
            while j<len(words)-1:
                if words[j+1]=='-s':
                    words[j]+='s'
                    del words[j+1]
                else: j+=1
            generated_utts[cn] = ' '.join(words)
        generated_utt = random.choice(generated_utts)

        # update belief state
        belief_tm1 = belief_t[:self.inf_dimensions[-1]]

        # packing
        response = {'belief_t'  : flatten_belief_t[:self.inf_dimensions[-1]].tolist(),
                    'generated' : generated_utt,
                    'selected_venue': selected_venue_t,
                    'venue_offered' : venue_offered_t   }

        ############################### debugging ############################
        print 'User Input :\t%s'% user_utt_t
        if self.trk=='rnn' and self.trkinf==True:
            print 'Belief Tracker :'
            print '  | %16s%13s%20s|' % ('','Informable','')
            print '  | %16s\t%5s\t%20s |' % ('Prediction','Prob.','Ground Truth')
            print '  | %16s\t%5s\t%20s |' % ('------------','-----','------------')
            for i in range(len(self.inf_dimensions)-1):
                psem = self.reader.infovs[ \
                        np.argmax(np.array(full_belief_t[i])) +\
                        self.inf_dimensions[i] ]
                prob = np.max(np.array(full_belief_t[i]))
                print '  | %16s\t%.3f |' % (psem,prob)
                #print full_belief_t[i]
        if self.trk=='rnn' and self.trkreq==True:
            infbn = 3 if self.trkinf else 0
            print '  | %16s%13s%20s|' % ('','Requestable','')
            print '  | %16s\t%5s\t%20s |' % ('Prediction','Prob.','Ground Truth')
            print '  | %16s\t%5s\t%20s |' % ('------------','-----','------------')
            for i in range(0,len(self.req_dimensions)-1):
                psem = self.reader.reqs[ \
                        np.argmax(np.array(full_belief_t[infbn+i])) +\
                        self.req_dimensions[i] ]
                prob = np.max(np.array(full_belief_t[infbn+i]))
                print '  | %16s\t%.3f |' % (psem,prob)

            # offer change tracker
            psem = 0 if full_belief_t[-1][0]>=0.5 else 1
            prdtvenue = 'venue=change' if psem==0 else 'venue=not change'
            prob      = full_belief_t[-1][0] if psem==0 else 1-full_belief_t[-1][0]
            print '%20s\t%.3f' % (prdtvenue,prob)

        match_number = np.argmax(np.array(db_degree_t[-6:]))
        match_number = str(match_number) if match_number<5 else '>5'
        print
        print 'DB Match     : %s' % match_number
        print
        print 'Generated    : %s' % generated_utts[0]
        for g in generated_utts[1:]:
          print '             : %s'% g
        print '--------------------------'
        print

        ############################### debugging ############################
        return response

    # Interactive interface
    def dialog(self):
        # interactive dialog interface
        print "="*40 + "\n\tStarting Interaction\n" + "="*40
        iact = Interact()
        turnNo = 0

        # initial state
        belief_tm1      = []
        generated_tm1   = ''
        venue_offered   = {}
        selected_venue  = -1
        while True:

            # asking user utterance
            sent = iact.prompt()
            if iact.quit(sent):
                break

            # chat one turn
            response = self.reply(sent,generated_tm1,
                    selected_venue,venue_offered,belief_tm1)

            # updating
            belief_tm1      = response['belief_t']
            generated_tm1   = response['generated']
            venue_offered   = response['venue_offered']
            selected_venue  = response['selected_venue']

            # deciding quit
            if iact.quit(generated_tm1):
                break

    # search database function
    def _searchDB(self,b):

        query = []
        q = []
        db_logic = []
        # formulate query for search
        if self.trkinf==True:
            for i in range(len(self.inf_dimensions)-1):
                b_i = b[self.inf_dimensions[i]:self.inf_dimensions[i+1]]
                idx = np.argmax(np.array(b_i)) + self.inf_dimensions[i]
                # find the index with the largest belief
                # ignore dont care case
                s2v = self.reader.infovs[idx]
                if '=any' not in s2v and '=none' not in s2v:
                    query.append(idx)
                q.append(idx)
            # search through db by query
            for entry in self.reader.db2inf:
                if set(entry).issuperset(set(query)):
                    db_logic.append(1)
                else:
                    db_logic.append(0)
            # form db count features
            dbcount = sum(db_logic)
            if dbcount<=3:
                dummy = [0 for x in range(6)]
                dummy[dbcount] = 1
                db_logic.extend(dummy)
            elif dbcount<=5:
                db_logic.extend([0,0,0,0,1,0])
            else:
                db_logic.extend([0,0,0,0,0,1])
        else:
            db_logic = [0,0,0,0,0,1]
        # db_logic is just a 1-hot encoding of 6 vector size
        # q is a list of index in self.infovs for every infomable that the belief
        # has the highest probbility for
        return db_logic, q

    def initBackupWeights(self):
        self.params = self.model.getParams()

    def setBackupWeights(self):
        self.params = self.model.getParams()

    def getBackupWeights(self):
        self.model.setParams( self.params )

    def saveNet(self):
        if self.debug:
            print 'saving net to file ... '

        self.setBackupWeights()
        bundle={
            'file'  :dict( [(name,eval(name)) for name in self.file_vars]),
            'learn' :dict( [(name,eval(name)) for name in self.learn_vars]),
            'data'  :dict( [(name,eval(name)) for name in self.data_vars] ),
            'gen'   :dict( [(name,eval(name)) for name in self.gen_vars]  ),
            'n2n'   :dict( [(name,eval(name)) for name in self.n2n_vars]  ),
            'enc'   :dict( [(name,eval(name)) for name in self.enc_vars]  ),
            'dec'   :dict( [(name,eval(name)) for name in self.dec_vars]  ),
            'ply'   :dict( [(name,eval(name)) for name in self.ply_vars]  ),
            'trk'   :dict( [(name,eval(name)) for name in self.trk_vars]  ),
        }
        pk.dump(bundle, open(self.modelfile, 'wb'))

    def loadNet(self,parser,mode='test'):

        print '\n\nloading net from file %s ... ' % self.modelfile
        bundle = pk.load(open(self.modelfile, 'rb'))

        # load learning variables from config
        if mode=='adjust' or mode=='rl':
            self.lr             = parser.getfloat('learn','lr')
            self.llogp          = parser.getfloat('learn','llogp')
            self.stop_count = parser.getint('learn','stop_count')
            self.cur_stop_count = parser.getint('learn','cur_stop_count')
            self.l2             = parser.getfloat('learn','l2')
            self.split          = literal_eval(parser.get('data','split'))
            self.trk_enc        = parser.get('trk','trkenc')
            self.seed           = parser.getint('learn','random_seed')
        else: # load learning variables from model
            self.lr             = bundle['learn']['self.lr']
            self.llogp          = bundle['learn']['self.llogp']
            self.cur_stop_count = bundle['learn']['self.cur_stop_count']
            self.stop_count     = bundle['learn']['self.stop_count']
            self.l2             = bundle['learn']['self.l2']
            self.split          = bundle['data']['self.split']
            self.trk_enc        = bundle['trk']['self.trk_enc']
            self.seed           = bundle['learn']['self.seed']

        # these we can just load from model
        self.lr_decay       = bundle['learn']['self.lr_decay']
        self.min_impr       = bundle['learn']['self.min_impr']
        self.grad_clip      = bundle['learn']['self.grad_clip']
        self.debug          = bundle['learn']['self.debug']
        self.valid_logp     = bundle['learn']['self.valid_logp']

        # model parameters
        self.params         = bundle['learn']['self.params']

        # load data files from model
        self.corpusfile     = bundle['file']['self.corpusfile']
        # self.dbfile         = bundle['file']['self.dbfile']
        self.ontologyfile   = bundle['file']['self.ontologyfile']
        self.semidictfile   = bundle['file']['self.semidictfile']
        # always load model file name from config
        self.modelfile      = parser.get('file','model')

        # setting data manipulations from config
        self.split          = literal_eval(parser.get('data','split'))
        self.lengthen       = parser.getint('data','lengthen')
        self.shuffle        = parser.get('data','shuffle')
        self.percent        = parser.get('data','percent')

        # Note: always load generation variables from config
        self.topk           = parser.getint('gen','topk')
        self.beamwidth      = parser.getint('gen','beamwidth')
        self.verbose        = parser.getint('gen','verbose')
        self.repeat_penalty = parser.get('gen','repeat_penalty')
        self.token_reward   = parser.getboolean('gen','token_reward')
        self.alpha          = parser.getfloat('gen','alpha')

        # load encoder decoder structures from config
        self.enc            = parser.get('n2n','encoder')
        self.dec            = parser.get('n2n','decoder')
        # load tracker setting from model, cannot change
        self.trk            = bundle['n2n']['self.trk']
        # load encoder variables from model
        self.input_hidden   = bundle['enc']['self.input_hidden']
        # load vocab size from model
        self.vocab_size     = bundle['enc']['self.vocab_size']

        # load policy parameters from config
        self.policy         = parser.get('ply','policy')
        self.latent         = parser.getint('ply','latent')\
                              if self.policy=='latent' else 0

        # load decoder variables from config
        self.output_hidden  = parser.getint('dec','ohidden')
        self.seq_wvec_file  = parser.get('dec','wvec')
        self.dec_struct     = parser.get('dec','struct')
        self.use_snapshot   = parser.getboolean('dec','snapshot')

        # load tracker variables from model
        self.trkinf         = bundle['trk']['self.trkinf']
        self.trkreq         = bundle['trk']['self.trkreq']
        self.inf_dimensions = bundle['trk']['self.inf_dimensions']
        self.req_dimensions = bundle['trk']['self.req_dimensions']
        self.belief         = parser.get('trk','belief')
        self.trk_wvec_file  = parser.get('trk','wvec')
        self.task_size      = bundle['trk']['self.task_size']

        # always load learning mode from config
        self.learn_mode = parser.get('mode','learn_mode')

        # set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        np.set_printoptions(precision=4)

        # load dataset
        self.reader = DataReader(
            self.corpusfile, self.semidictfile, self.ontologyfile,  # removed 2nd db file
            self.split, self.lengthen, self.percent,
            self.shuffle, self.trk_enc, self.verbose, mode, self.policy,
            self.latent)

        # setup model parameters
        self.model = NNSDS(self.enc, self.dec, self.policy,
            self.trk, self.trkinf, self.trkreq, self.belief, self.trk_enc,
            self.use_snapshot, self.dec_struct, self.vocab_size,
            self.input_hidden, self.output_hidden,
            self.inf_dimensions, self.req_dimensions, self.task_size, self.grad_clip,
            self.learn_mode, len(self.reader.snapshots), self.latent)

        # load weights
        self.getBackupWeights()
        self.model.loadConverseParams()
        # continue training
        if mode=='train' or mode=='adjust' or mode=='rl':
            self.model.config_theano()
            if self.debug:
                numofparams, trainable = self.model.numOfParams()
                print '\t\tnumber of parameters : %8d' % numofparams
                print '\t\tnumber of training parameters : %8d' % trainable

        if self.dec!='none': # if decoder exists
            if self.policy=='latent':
                # set testing mode sampling to only top-1
                if self.mode=='test' or self.mode=='valid':
                    self.model.policy.setSampleMode('prior',1)
                # set interaction mode sampling to top-5
                elif self.mode=='interact':
                    self.model.policy.setSampleMode('prior',5)
                # set RL mode sampling to top-5
                elif self.mode=='rl':
                    self.model.policy.setSampleMode('prior',5)
                    self.topk, self.beamwidth = 1, 2

            # config decoder
            self.model.decoder.setDecodeConfig(
                self.verbose, self.topk, self.beamwidth, self.reader.vocab,
                self.repeat_penalty, self.token_reward, self.alpha)


    def _statsTable(self):

                    # Metrics',  'Prec.', 'Recall', 'F-1', 'Acc.')
        return {'informable':{
                    'place_search_keyword': [10e-9, 10e-4, 10e-4, 10e-4],
                    'order'      : [10e-9, 10e-4, 10e-4, 10e-4],
                    'search_place_ratings'      : [10e-9, 10e-4, 10e-4, 10e-4],
                    'task_reference': [10e-9, 10e-4, 10e-4, 10e-4]
            },  'requestable':{
                    'place_ratings': [10e-9, 10e-4, 10e-4, 10e-4],
                    'place_address'      : [10e-9, 10e-4, 10e-4, 10e-4],
                    'waypoints'      : [10e-9, 10e-4, 10e-4, 10e-4],
                    'distance'  : [10e-9, 10e-4, 10e-4, 10e-4],
                    'duration'   : [10e-9, 10e-4, 10e-4, 10e-4],
                    'open_now'     : [10e-9, 10e-4, 10e-4, 10e-4],
            },
            'vmc': 10e-7, 'vmc_total': 10e-7, 'success': 10e-7, 'approp': [10e-7,10e-7],
            'success_tp': 10e-7, 'success_fp': 10e-7, 'success_fn': 10e-7
        }

    def _genScoreTable(self, sem_j):
        scoreTable = {}
        # requestable tracker scoreTable
        if self.trk=='rnn' and self.trkreq==True:
            infbn = 3 if self.trkinf else 0   # 3 because, =value, =any, =none
            for i in range(len(self.req_dimensions)-1):
                bn = self.req_dimensions[i]
                # prediction for this req tracker
                psem = self.reader.reqs[ \
                    np.argmax(np.array(sem_j[infbn+i])) +\
                    self.req_dimensions[i] ]
                #print psem
                # slot & value
                s,v = psem.split('=')
                if s=='place': # skip name slot
                    continue
                # assign score, if exist, +reward
                score = -0.05 if v=='none' else 0.2
                # slot value indexing
                vidx = self.reader.vocab.index('[VALUE_'+s.upper()+']')
                sidx = self.reader.vocab.index('[SLOT_'+s.upper()+']')
                scoreTable[sidx] = score
                scoreTable[vidx] = score # reward [VALUE_****] if generate
        # informable tracker scoreTable
        if self.trk=='rnn' and self.trkinf==True:
            for i in range(len(self.inf_dimensions)-1):
                bn = self.inf_dimensions[i]
                # prediction for this inf tracker
                psem = self.reader.infovs[np.argmax(np.array(sem_j[i]))+bn]
                #print psem
                # slot & value
                s,v = psem.split('=')
                # if none, discourage gen. if exist, encourage gen
                score = -0.5 if (v=='none' or v=='any') else 0.05
                # slot value indexing
                vidx = self.reader.vocab.index('[VALUE_'+s.upper()+']')
                sidx = self.reader.vocab.index('[SLOT_'+s.upper()+']')
                if not scoreTable.has_key(sidx) or scoreTable[sidx]<=0.0:
                    scoreTable[sidx] = 0.0 # less encourage for [SLOT_****]
                if not scoreTable.has_key(vidx) or scoreTable[vidx]<=0.0:
                    scoreTable[vidx] = score # encourage [SLOT_****]

        return scoreTable
