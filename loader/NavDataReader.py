#######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################
import os
import re
import sys
import simplejson as json
import math
import operator
import random
from pprint import pprint
import itertools
import numpy as np
from copy import deepcopy
from pprint import pprint

from utils.nlp import normalize
from utils.tools import findSubList
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

digitpat = re.compile('\d+')

class DataSplit(object):
    # data split helper , for split dataset into train/valid/test
    def __init__(self,split):
        self.split = split
        self.sum = sum(split)
    def train_valid(self,data):
        # split the dataset into train+valid
        e = int(len(data)*float(sum(self.split[:2]))/float(self.sum))
        return data[:e]
    def train(self,train_valid):
        # split training from train+valid
        e = int(len(train_valid)*\
                float(self.split[0])/float((sum(self.split[:2]))))
        return train_valid[:e]
    def valid(self,train_valid):
        # split validation from train+valid
        s = len(self.train(train_valid))
        return train_valid[s:]
    def test(self,data):
        # split the dataset into testing
        s = len(self.train_valid(data))
        return data[s:]

class DataReader(object):
    inputvocab = []
    outputvocab= []
    ngrams = {}
    idx2ngs = []

    def __init__(self,
            corpusfile, semifile, s2vfile,  # remove 2nd db file
            split, lengthen, percent, shuffle,
            trkenc, verbose, mode, att=False, latent_size=1):

        self.att = True if att=='attention' else False
        self.dl  = latent_size
        self.data  = {'train':[],'valid':[],'test':[]} # container for data
        self.mode = 'train'     # mode  for accessing data
        self.index = 0          # index for accessing data

        # data manipulators
        self.split  = DataSplit(split)  # split helper
        self.trkenc = trkenc
        self.lengthen = lengthen
        self.shuffle= shuffle

        # NLTK stopword module
        self.stopwords = set(stopwords.words('english'))
        for w in ['!',',','.','?','-s','-ly','</s>','s']:
            self.stopwords.add(w)

        # loading files
        #self.db       = self.loadjson(dbfile)  # TODO: remove db and parse it tgt with dialog
        self.s2v      = self.loadjson(s2vfile)
        self.semidict = self.loadjson(semifile)
        self.dialog   = self.loadjson(corpusfile)

        # producing slot value templates and db represetation
        self.prepareSlotValues()
        #self.structureDB()    # TODO: remove db and parse it together with dialog

        # load dialog
        self.loadVocab()
        if mode!='sds':
            self.loadDialog()
            self.loadSemantics()

        # goal
        self.parseGoal()

        # split dataset
        if mode!='sds':
            self._setupData(percent)

        if verbose : self._printStats()

    def loadDialog(self):

        # index words and make it suitable for NN input
        self.sourceutts = []
        self.targetutts = []
        self.masked_sourceutts = []
        self.masked_targetutts = []
        self.sourcecutoffs = []
        self.targetcutoffs = []
        self.masked_sourcecutoffs = []
        self.masked_targetcutoffs = []

        # delexicalised positions
        self.delsrcpos = []
        self.deltarpos = []

        # task reference
        self.refsrcfeat = []
        self.reftarfeat = []
        self.refmentions = []

        # finished dialogs
        self.finished = []
        self.newtask = []

        # venue specific - offered/changing
        self.offers = []
        self.changes= []

        # snapshot vectors
        self.snapshot_vecs = []

        # for each dialogue
        dcount = 0.0
        tcount = 0.0

        # for VAE initialisation
        self.sentGroupIndex = []
        groupidx = 0

        for d in self.dialog:

            # consider finished flag
            if d.has_key('finished'):
                self.finished.append(d['finished'])
            else:
                self.finished.append(True)

            # print loading msgs
            dcount += 1.0
            print '\tloading dialog from file ... finishing %.2f%%\r' %\
                (100.0*float(dcount)/float(len(self.dialog))),
            sys.stdout.flush()

            # container for each turn
            sourceutt   = []   # same
            targetutt   = []   # stores the list of target utterance
            m_sourceutt = []   # same
            m_targetutt = []   # stores the list of masked target utterance
            utt_group   = []   # stores self.sentGroup(groupidx), which is always incremented by 1
                               # by the order of the first sent-group found in loadVodab

            srcpos = []        # same
            tarpos = []        # contains list of tuple (slot_pos, value_pos) for every response

            maxtar = -1     # maximum length of original target
            maxsrc = -1     # maximum length of original source
            maxmtar= -1     # maximum length of masked target
            maxmsrc= -1     # maximum length of masked source
            maxfeat= -1     # maximum length of slot or value mentions in a response

            # task reference
            refsrcpos = []
            reftarpos = []
            refmaxfeat = 1
            refmention = []


            cur_ref_mention = [0] * len(self.refvs)+ [1]

            offers  = []      # list of tuple (1,0) - offer or (0,1)- no offer
            changes = []      # list of tuple (1,0) - change or (0,1)- no change
            prevoffer = []    # list of previous offers
            offered = False
            new = []  # true or false / 1 or 0

            snapshot_vecs = []   # list of snapshot vectors that contain snapshots
                                 # order by the last word in the masked target first
                                 # for every turn

            # for each turn in a dialogue
            for t in range(len(d['dial'])):
                tcount += 1
                turn = d['dial'][t]
                # extract system side sentence feature
                #sent = turn['sys']['sent']
                #mtar, tar, spos, vpos, venues \
                #    = self.extractSeq(sent,type='target')

                mtar, tar, spos, vpos, venues = self.extractSeq(turn['sys']['tokens'], slotpos=turn['sys']['slotpos'],\
                    type='target',index=True, split=True, debug=True)
                # by default, index=True, normalize=False
                # Output:
                # mtar: masked indexes [i1, i2, i3, ..., iN]
                # tar: original indexe [i1, i2, i3, ..., iN]
                # spos: [list of slot positions according to [self.infovs + self.regs]]
                # vpos: [list of value positions, ""]
                # names: list of names mentioned

                # update ref mention
                cur_ref_mention, refpos = self.extractRef(turn['sys']['tokens'], cur_ref_mention, slotpos=turn['sys']['slotpos'], index=True, split=True)

                # cur_ref_mention = [x for x in cur_ref_mention] # copy the last one
                # for name in names:
                #     if 'ref=%s' % name not in self.refvs:
                #         print name, 'not in self.refvs'
                #         print ' '.join(turn['sys']['tokens'])
                #         print d['dialogue_id']
                #     cur_ref_mention[self.refvs.index('ref=%s' % name)] = 1
                #     cur_ref_mention[-1] = 0

                refmention.append(cur_ref_mention)

                # handling positional features
                for f in refpos:            # list of list of index of slot positions
                    if len(f)>refmaxfeat:    # updates the maximum number of slot mentions
                        refmaxfeat = len(f)
                reftarpos.append(refpos)

                # store sentence group
                utt_group.append(self.sentGroup[groupidx])
                groupidx += 1

                # changing offer label
                # if at least some name is mentioned and first name is not prevoffer
                # not matching in previous offers
                if len(venues)!=0 and venues[0] not in prevoffer: # not matching
                    if prevoffer==[]: # new offer
                        change = [0,1]
                    else: # changing offer
                        change = [1,0]
                    prevoffer = venues   # TODO: shall we include prevoffer as all other offered?
                                         # have to investigate what's the need for this first
                else:
                    change = [0,1]

                # what does the tuple [x1, x2] represent?
                # it seems that if it's a new offer or no names mentioened, or first venue is in pref offer
                # then change is = [0,1]
                # but it's a new name mentioned , then change = [1,0]
                # it's a one-hot encoding?
                # x1 : change?, x2: no change?
                changes.append(change)

                # offer label
                if offered or len(venues)!=0: # offer has happened
                    offer = [1,0]
                    offered = True
                    # now offer = [1,0]
                    # OHE also? where x1: offer? x2: no offer?
                else:
                    offer = [0,1]
                    # No offer
                offers.append(offer)

                # delexicalised
                if len(mtar)>maxtar:    # if len(masked indexes) > -1:
                    maxtar = len(mtar)  # updates maxtar to be the longest masked index sequence
                m_targetutt.append(mtar)

                # extract snapshot vectors
                snapshot_vec = [[0.0 for x in range(len(self.snapshots))]]
                # it's another one-hot-encoding of all snapshots prepared in loadVodab
                # if offered, then set value of OFFERED index = 1
                # if changed, then set value of CHANGEd index = 1

                # add offer and change to snapshot vector
                if offer==[1,0] : snapshot_vec[0][
                        self.snapshots.index('OFFERED') ] = 1.0
                if change==[1,0]: snapshot_vec[0][
                        self.snapshots.index('CHANGED') ] = 1.0

                # attentive snapshot
                for w in mtar[::-1]:   # loop through the mapped utterance backwards
                    ssvec = deepcopy(snapshot_vec[0])   # copy the snapshot vector
                    if self.vocab[w] in self.snapshots:  # if word in snaphots
                        ssvec[ self.snapshots.index(     # then set the copy of the vector to 1
                            self.vocab[w]) ] = 1.0
                    snapshot_vec.insert(0,ssvec)   # insert the snapshot vector a copy of it at the start
                # decide changing snapshot or not
                if self.att==True:
                    snapshot_vecs.append(snapshot_vec[:-1])
                    # if attention, append a reference of the list except the last one (as it's the template)
                    # at the end
                    # e.g. [ssvev1, ssvev2, ..., ssvecN] , where there are N words
                else:  # normal
                    snapshot_vecs.append([deepcopy(snapshot_vec[0])
                        for x in snapshot_vec[:-1]])
                    # e.g. [ssvev1, ssvec1, ..., ssvec1], where there are N words
                    # where ssvec1 = [[0, 0, 1, 0, ....], [0, 0, 1, 0, ....], ... x N]
                    # each entry is a list representing a word snapshot from the back
                    # where the inner list is the size of self.snapshots

                # handling positional features
                for f in spos:            # list of list of index of slot positions
                    if len(f)>maxfeat:    # updates the maximum number of slot mentions
                        maxfeat = len(f)
                for f in vpos:            # list of list of value positions
                    if len(f)>maxfeat:    # updates the maximum number of value mentions
                        maxfeat = len(f)
                tarpos.append([spos,vpos])
                # [spos: [[], [1, 2], [], [], ... x self.infovs+self.reqs],
                #  vpos: [[], [1, 2], [], [], ... x self.infovs+self.reqs]]
                # print 'sys output   :\t%s' % ' '.join(self.vocab[x] for x in tar)
                # print 'masked target:\t%s' % ' '.join(self.vocab[x] for x in mtar)
                # for i, sf in enumerate(spos):
                #     if len(sf) > 0:
                #         print 'slot present:\t%s' % self.vocab[mtar[sf[0]]], 'for: %s' % self.allvs[i]
                # for i, vf in enumerate(vpos):
                #     if len(vf) > 0:
                #         print 'value present:\t%s' % self.vocab[mtar[vf[0]]], 'for: %s' % self.allvs[i]
                # print


                # non delexicalised
                if len(tar)>maxmtar:
                    maxmtar = len(tar)    # udpates maximum length of original
                targetutt.append(tar)

                # usr responses
                #sent = turn['usr']['transcript']
                #msrc, src, spos, vpos, _ = self.extractSeq(sent,type='source')

                msrc, src, spos, vpos, _ = self.extractSeq(turn['usr']['tokens'], slotpos=turn['usr']['slotpos'],\
                    type='source',index=True, split=True, debug=True)

                # repeats the same as sys
                # except that now, no need for changes, offers, snapshot vector,

                # update refsrcpos
                names, refpos = self.extractRef(turn['usr']['tokens'], slotpos=turn['usr']['slotpos'],\
                    index=True, split=True)
                # handling positional features
                for f in refpos:            # list of list of index of slot positions
                    if len(f)>refmaxfeat:    # updates the maximum number of slot mentions
                        refmaxfeat = len(f)
                refsrcpos.append(refpos)

                # update task reference - semantics?
                # delexicalised
                if len(msrc)>maxsrc:
                    maxsrc = len(msrc)
                m_sourceutt.append(msrc)

                # handling positional features
                for f in spos:
                    if len(f)>maxfeat:
                        maxfeat = len(f)
                for f in vpos:
                    if len(f)>maxfeat:
                        maxfeat = len(f)
                srcpos.append([spos,vpos])

                # print 'user input   :\t%s' % ' '.join(self.vocab[x] for x in src)
                # print 'masked source:\t%s' % ' '.join(self.vocab[x] for x in msrc)
                # for i, sf in enumerate(spos):
                #     if len(sf) > 0:
                #         print 'slot present:\t%s' % self.vocab[msrc[sf[0]]], 'for: %s' % self.allvs[i]
                # for i, vf in enumerate(vpos):
                #     if len(vf) > 0:
                #         print 'value present:\t%s' % self.vocab[msrc[vf[0]]], 'for: %s' % self.allvs[i]
                # print

                # non delexicalised
                if len(src)>maxmsrc:
                    maxmsrc = len(src)
                sourceutt.append(src)

                if d['dial'][t]['new']:
                    new.append(1)
                else:
                    new.append(0)

            # sentence group
            self.sentGroupIndex.append(utt_group)

            # offers
            self.changes.append(changes)
            self.newtask.append(new)
            self.offers.append(offers)

            # padding for snapshots
            # e.g. [ssvec 1, ssvec2, ssvec3, ssvec1, ssvec1]
            # e.g. maxtar = 5
            for i in range(len(m_targetutt)):
                snapshot_vecs[i].extend(
                        [snapshot_vecs[i][0]]*(maxtar-len(m_targetutt[i])))

            # padding unk tok
            m_sourcecutoff = []
            m_targetcutoff = []
            for i in range(len(m_targetutt)):
                m_targetcutoff.append(len(m_targetutt[i]))
                m_targetutt[i].extend(
                        [self.vocab.index('<unk>')]*(maxtar-len(m_targetutt[i])) )
            for i in range(len(m_sourceutt)):
                m_sourcecutoff.append(len(m_sourceutt[i]))
                m_sourceutt[i].extend(
                        [self.vocab.index('<unk>')]*(maxsrc-len(m_sourceutt[i])) )

            # non delexicalised version
            # Padding maximum wrong is it??
            # Shouldn't it be maxtar and maxsrc, and otherwise?

            sourcecutoff = []
            targetcutoff = []
            for i in range(len(targetutt)):
                targetcutoff.append(len(targetutt[i]))
                targetutt[i].extend(   # TODO: optimize the code; save index
                        [self.vocab.index('<unk>')]*(maxmtar-len(targetutt[i])) )

            for i in range(len(sourceutt)):
                sourcecutoff.append(len(sourceutt[i]))
                sourceutt[i].extend(
                        [self.vocab.index('<unk>')]*(maxmsrc-len(sourceutt[i])) )

            # padding positional features
            # tarpos:
            # [
            #   [spos: [[], [1, 2], [], [], ... x self.infovs+self.reqs],
            #    vpos: [[], [1, 2], [], [], ... x self.infovs+self.reqs]]
            #   x T ...
            # ]

            # Padding semantic info
            #print 'maxfeat:', maxfeat
            for i in range(len(tarpos)): # for every dialog
                for j in range(len(tarpos[i])): # for spos, and vpos
                    for k in range(len(tarpos[i][j])):  # for every index
                        tarpos[i][j][k].extend([-1]*(maxfeat-len(tarpos[i][j][k])))
            for i in range(len(srcpos)):
                for j in range(len(srcpos[i])):
                    for k in range(len(srcpos[i][j])):
                        srcpos[i][j][k].extend([-1]*(maxfeat-len(srcpos[i][j][k])))

            # Padding task reference feat
            #assert(refmaxfeat > 0)
            for i in range(len(reftarpos)):  # for every turn
                for j in range(len(reftarpos[i])):  # for every location
                    reftarpos[i][j].extend([-1]*(refmaxfeat-len(reftarpos[i][j])))
            for i in range(len(refsrcpos)):
                for j in range(len(refsrcpos[i])):
                    refsrcpos[i][j].extend([-1]*(refmaxfeat-len(refsrcpos[i][j])))

            # task reference matrix
            self.refsrcfeat.append(refsrcpos)
            self.reftarfeat.append(reftarpos)
            self.refmentions.append(refmention)

            # entire dialogue matrix
            self.sourceutts.append(sourceutt)
            self.targetutts.append(targetutt)
            self.sourcecutoffs.append(sourcecutoff)
            self.targetcutoffs.append(targetcutoff)

            self.masked_sourceutts.append(m_sourceutt)
            self.masked_targetutts.append(m_targetutt)
            self.masked_sourcecutoffs.append(m_sourcecutoff)
            self.masked_targetcutoffs.append(m_targetcutoff)

            self.snapshot_vecs.append(snapshot_vecs)

            # positional information
            self.delsrcpos.append(srcpos)
            self.deltarpos.append(tarpos)

    def loadSemantics(self):

        # sematic labels
        self.info_semis = []
        self.req_semis  = []
        self.db_logics = []
        self.ref_semis = []

        sumvec      = np.array([0 for x in range(self.infoseg[-1])])
        # for each dialogue
        dcount = 0.0
        for dx in range(len(self.dialog)):  # for every dialog
            d = self.dialog[dx]
            # print loading msgs
            dcount += 1.0
            print '\tloading semi labels from file ... finishing %.2f%%\r' %\
                (100.0*float(dcount)/float(len(self.dialog))),
            sys.stdout.flush()

            # container for each turn
            info_semi   = []
            req_semi    = []
            semi_idxs   = []
            db_logic    = []
            ref_semi    = []

            # for each turn in a dialogue
            for t in range(len(d['dial'])):
                turn = d['dial'][t]

                # read informable semi
                #reqnones = ['pricerange=none','food=none','area=none']
                #infones = [str(x)+'=none' for x in self.s2v['informable']] # initially everything is None
                if d['dial'][t]['new'] or len(info_semi) == 0:
                #if len(info_semi) == 0:
                    semi = sorted([str(x)+'=none' for x in self.s2v['informable']])
                else:
                    semi = deepcopy(info_semi[-1])  # copy the last one

                #semi = sorted(infones) \
                #        if len(info_semi)==0 else deepcopy(info_semi[-1])       # TODO: update informable semi

                for da in turn['usr']['slu']:  # updates semi based on usr slu
                    for s2v in da['slots']:
                        # skip invalid slots
                        if len(s2v)!=2 or s2v[0]=='slot' or s2v[0]=='place':   # means it's a request

                            continue                        # ignore
                        s,v = s2v
                        v = str(v).lower()  # added this

                        toreplace = None
                        for sem in semi:
                            if s in sem:
                                toreplace = sem
                                break
                        if toreplace:
                            semi.remove(toreplace)
                        semi.append(s+'='+v)

                # if goal changes not venue changes  # TODO: can goal change?
                if self.changes[dx][t]==[1,0]:  # if it changes
                    if info_semi[-1] != sorted(semi):   # check the last one, and the current is it the same
                        self.changes[dx][t] = [0,1]     # if different, then set the change back to no change

                info_semi.append(sorted(semi))
                # info_semi: [[pricerange=none, food=none, area=none], x T...]

                # indexing semi and DB
                vec = [0 for x in range(self.infoseg[-1])]
                constraints = []
                for sem in semi:
                    if 'name=' in sem:   # name shouldn't be in there
                        continue
                    sem_idx = self.infovs.index(sem)
                    vec[sem_idx] = 1
                    if sem_idx not in self.dontcare:
                        constraints.append(sem_idx)

                # vec = [0, 1, 0, ....  x len(self.infovs)]
                # constraints = [<indexes of constraints>...]
                #                 reference to self.infovs
                semi_idxs.append(vec)
                sumvec += np.array(vec)
                infosemi = semi

                # check db match
                #match = [len(filter(lambda x: x in constraints, sub)) \
                #        for sub in self.db2inf]
                        # TODO: find a workaround
                        # obtain a list of matches for all combinations
                        # [3, 2, 1, 0, 1, 0, 0 ,...]

                #venue_logic = [int(x>=len(constraints)) for x in match]
                        # vector of 1s and 0
                        # [0, 1, 0, 1, 0, 1, ...]
                        # 1 if x in match is >= len(constraints)
                vcount = 0
                #for midx in range(len(venue_logic)):
                #    if venue_logic[midx]==1:
                #        vcount += len(self.idx2db[midx])
                        # TODO: find a workaround db
                        # increment with the len of the number of places in
                        # that combination / all matched combinations
                        # vcount = total matched venues
                if 'places' in turn['state']:
                    venue_logic = [0, 1]  # have place
                else:
                    venue_logic = [1, 0]  # no place
                    #vcount = len(turn['state']['places'])
                # if vcount<=3:
                #     dummy = [0 for x in range(6)]   # ?? Why 6?
                #     dummy[vcount] = 1
                #     venue_logic.extend(dummy)
                #
                #     # venue_logic: [0, 1, 0, 1, 0, ...., 0, 0, 0, 0, 0, 1]
                # elif vcount<=5:
                #     venue_logic.extend([0,0,0,0,1,0])
                # else:
                #     venue_logic.extend([0,0,0,0,0,1])

                # [0, 1, 2, 3] number of matches
                #venue_logic = [0 for x in range(4)]  # maximum only 3 locations
                #venue_logic[vcount] = 1
                #venue_logic.extend(dummy)


                db_logic.append(venue_logic)
                # db_logic: [ [0, 0, 1, 0, ..., 0,0,0,0,0,1], ... xT]

                # read requestable semi
                semi = sorted(self.s2v['requestable'].keys())
                ref_vec = [0] * len(self.refvs)
                ref_vec[-1] = 1
                #semi =  sorted(['food','pricerange','area'])+\
                #        sorted(['phone','address','postcode'])  # TODO: remove informables as requestables
                for da in turn['usr']['slu']:
                    for s2v in da['slots']:
                        if s2v[0]=='slot':  # it is a request action
                            for i in range(len(semi)):
                                if s2v[1]==semi[i]:
                                    semi[i] += '=exist'
                        if s2v[0]=='place':  # it is a request action with place
                            for i in range(len(self.refvs)-1):
                                if s2v[1].lower()==self.refvs[i].split('=')[1]:
                                    ref_vec[i] = 1
                                    ref_vec[-1] = 0
                                    break
                ref_semi.append(ref_vec)

                for i in range(len(semi)):
                    if '=exist' not in semi[i]:
                        semi[i] += '=none'
                # semi = ['food=none', pricerange=none', 'area=exist', ...]
                vec = [0 for x in range(self.reqseg[-1])]
                for sem in semi:
                    vec[self.reqs.index(sem)] = 1
                req_semi.append(vec)

            self.info_semis.append(semi_idxs)
            self.req_semis.append(req_semi)
            self.db_logics.append(db_logic)
            self.ref_semis.append(ref_semi)
        print

    def extractRef(self, sent, split=False, index=True, slotpos=None, debug=False):
        vocab = self.vocab
        if not split:
            words = [x.lower() for x in sent.split()]
        else:
            words = [x.lower() for x in sent]

        # indexing, non-delexicalised
        if index:
            idx  = map(lambda w: vocab.index(w) if w in vocab else 0, words)
        else:
            idx = words

        # delexicalise all
        sent = self.delexicalise(' '.join(words),slotpos,type,mode='all',sep='$',debug=debug)
        words= sent.split()

        refsltpos = [[] for x in self.refvs[:-1]]
        names = []
        for i in range(len(words)):
            if '::' not in words[i]:
                continue
            # handling offer changing
            if words[i].startswith('[VALUE_PLACE]'):  # change name to place
                name = words[i].replace('[VALUE_PLACE]::','')

                # remove pos identifier
                tok, ID = words[i].split("::")
                ID = ID.replace('$',' ')  # revert back the spaces of the actual value
                # record position
                names.append(ID)
                #print tok, ID  # remove outer brackets

                for j in range(len(self.refvs)):
                    s,v = self.refvs[j].split('=')
                    if v == ID:
                        #print 'matched!', v, '==', ID
                        refsltpos[j].append(i)
                        #print 'updated index %s with value %s:' % (j, i), refsltpos
                        break

        cur_ref_mention = [x for x in cur_ref_mention] # copy the last one
        for name in names:
            if 'ref=%s' % name not in self.refvs:
                print name, 'not in self.refvs'
                print ' '.join(turn['sys']['tokens'])
                print d['dialogue_id']
            cur_ref_mention[self.refvs.index('ref=%s' % name)] = 1
            cur_ref_mention[-1] = 0
        
        return cur_ref_mention, refsltpos

    def extractSeq(self,sent,type='source',normalise=False,index=True, split=False, slotpos=None, debug=False):

        # setup vocab
        if type=='source':  vocab = self.vocab
        elif type=='target':vocab = self.vocab

        # standardise sentences
        #if normalise:
        #    sent = normalize(sent)

        #if debug:
        #    print sent

        # preporcessing
        if not split:
            words = [x.lower() for x in sent.split()]
        else:
            words = [x.lower() for x in sent]
        if type=='source':
            if len(words)==0: words = ['<unk>']
        elif type=='target':
            words = ['</s>'] + words + ['</s>']
        words = [x.lower() for x in words]  # added lower case here

        # indexing, non-delexicalised
        if index:
            idx  = map(lambda w: vocab.index(w) if w in vocab else 0, words)
        else:
            idx = words

        #print words

        # delexicalise all
        sent = self.delexicalise(' '.join(words),slotpos,type,mode='all', debug=debug)
        # convert all values found in self.values into the format of
        # [SLOT_<name>]::supervalue('-' as space) or
        # [VALUE]_<name>]::supervalue('-' as space)
        #sent = re.sub(digitpat,'[VALUE_COUNT]',sent)
        words= sent.split()

        #print words

        # formulate delex positions
        allvs = self.infovs+self.reqs   # all value-slot pairs
        sltpos = [[] for x in allvs]
        valpos = [[] for x in allvs]
        names = []
        for i in range(len(words)):
            if '::' not in words[i]:
                continue
            # handling offer changing
            if words[i].startswith('[VALUE_PLACE]'):  # change name to place
                name = words[i].replace('[VALUE_PLACE]::','')
                names.append(name)
            # remove pos identifier
            tok, ID = words[i].split("::")
            words[i] = tok   # overwrite with just the [SLOT/VALUE]
            # record position
            #print tok, ID  # remove outer brackets
            splits = tok[1:-1].lower().split('_')
            # remove the square bracket, then lower case it, then split by '_'
            mytok = splits[0]
            # if mytok == 'slot':
            #     sov = '_'.join(splits[1:])
            # else:
            #     sov = splits[1]
            #mytok,sov = splits[0], '_'.join(splits[1:])
            #mytok,sov = tok[1:-1].lower().split('_')
            # mytok is either SLOT or VALUE
            # sov is one of the INF or REQ keys

            ID = ID.replace('-',' ')  # revert back the spaces of the actual value
            #mylist = sltpos if mytok=='slot' else valpos  # unused

            # go through every allvs
            for j in range(len(allvs)):  # allvs = self.infovs+self.reqs   # all value-slot pairs
                s,v = allvs[j].split('=')
                comp = s if mytok=='slot' else v
                if comp==ID:
                    #print 'comparing:', comp, '   with    ', ID
                    if mytok=='slot':  # if slot match
                        sltpos[j].append(i)
                    else:              # if value match
                        valpos[j].append(i)
            # print sent
            # print 'slotpos:', sltpos[-len(self.reqs):]
            # print 'valpos:', valpos[-len(self.reqs):]
            # print

        # indexing, delexicalised
        if index:
            midx = map(lambda w: vocab.index(w) if w in vocab else 0, words)
        else:
            midx = words

        #     delexicalise index, lexicalised index,
        #     index of slot positions, index of value positions
        #     list of names
        return midx, idx, sltpos, valpos, names

    def delexicalise(self,utt,slotpos=None,type=None,mode='all',sep='-',debug=False):
        inftoks =   ['[VALUE_'+s.upper()+']' for s in self.s2v['informable'].keys()] + \
                    ['[SLOT_' +s.upper()+']' for s in self.s2v['informable'].keys()] + \
                    ['VALUE_ANY]','[VALUE_PLACE]'] + \
                    ['[SLOT_' +s.upper()+']' for s in self.s2v['requestable'].keys()]
        reqtoks =   ['[VALUE_'+s.upper()+']' for s in self.s2v['requestable'].keys()]
        # TODO: remove requestable as inf tokens?
        # TODO: standardize Dont care?

        if utt.startswith('* gibberish *'):
            utt = '<unk>'

        if debug:
            print 'before lexicalise:', utt

        # delexicalise based on slu values first
        # reason: ?
        if slotpos is not None:
            utt = utt.split()
            originals = []     # searches
            slot_replacements = []
            value_replacements = []
            for slot in slotpos:  # slot is a dict{'slot', 'start', 'exclusive_end'}
                # if it is a name of a place, it has 'value' key
                if slot['slot'].startswith('order'):
                    #type = slot['slot'][6:-1]  # order(before/after)
                    name = 'order'
                else:
                    name = slot['slot']
                value = '[VALUE_' + name.upper() + ']'
                start, end = slot['start'], slot['exclusive_end']
                if start is None:
                    continue
                if type == 'target':  # means it's a system response, and have </s>
                    start += 1
                    end += 1
                if start < len(utt):
                    originals.append(' '.join(utt[start:end]))
                    if slot['slot'] == 'place':
                        #utt = utt[:start+1] + utt[end:]
                        #utt[start] = value + '::' + slot['value'].replace(' ', sep)
                        value_replacements.append(slot['value'].replace(' ', sep))
                    else:
                        value_replacements.append(' '.join(utt[start:end]).replace(' ', sep))
                        #originals.append(' '.join(utt[start:end]).replace(' ', sep))
                    slot_replacements.append(value + '::')
                else:
                    print '%s start index %s is out of range' % (value, start)
                    print ' '.join(utt)
                        
                # special case for [SLOT_SEARCH_PLACE_RATINGS]
                if type == 'source' and name == 'search_place_ratings':
                    value = '[SLOT_' + 'search_place_ratings'.upper() + ']'
                    start = None
                    end = None
                    for i, token in enumerate(utt):
                        if token == 'ratings':
                            start = i
                            end = i + 1
                    if start is None:
                        continue
                    #assert(start)
                    #assert(end)
                    if start < len(utt):
                        originals.append(' '.join(utt[start:end]))
                        value_replacements.append(' '.join(utt[start:end]))
                        slot_replacements.append(value + '::')
                    else:
                        print '%s start index %s is out of range' % (value, start)
                        print ' '.join(utt)

            utt = ' '.join(utt)
            for i in range(len(originals)):
                utt = (' '+utt+' ').replace(' '+originals[i]+' ', ' '+slot_replacements[i]+value_replacements[i]+' ')
                utt = utt[1:-1]

        # Problem with
        # ORDER, which the order is not explicitly stated in the utterance
        # so we need to preprocess the order first to contain the actual value
        if debug:
            print 'intermediate lexicalise:', utt

        # for every value in self.values  ( which is a list of all values )
        # find the tokens to replace, which is [SLOT/VALUE_X]::exact-value
        # do a string match based on value
        for i in range(len(self.values)):
            # informable mode, preserving location information
            if mode=='informable'and self.slots[i] in inftoks:
                tok = self.slots[i]+'::'+(self.supervalues[i]).replace(' ',sep)
                utt = (' '+utt+' ').replace(' '+self.values[i]+' ',' '+tok+' ')
                utt = utt[1:-1]
            # requestable mode
            elif mode=='requestable' and self.slots[i] in reqtoks:
                utt = (' '+utt+' ').replace(' '+self.values[i]+' ',' '+self.slots[i]+' ')
                utt = utt[1:-1]
            elif mode=='all':
                tok = self.slots[i]+'::'+(self.supervalues[i]).replace(' ',sep) \
                        if self.slots[i] in inftoks else self.slots[i]
                utt = (' '+utt+' ').replace(' '+self.values[i]+' ',' '+tok+' ')
                utt = utt[1:-1]
        #utt = re.sub(digitpat,'[VALUE_COUNT]',utt)
        if debug:
            print 'after lexicalise:', utt
            print
        return utt

    def prepareSlotValues(self):

        print '\tprepare slot value templates ...'
        # put db requestable values into s2v

        # Originally, go through every DB, and add each value
        # into the requestables, and other
        #for e in self.db:
        #    for s,v in e.iteritems():
        #        if self.s2v['requestable'].has_key(s):
        #            self.s2v['requestable'][s].append(v.lower())
        #        if self.s2v['other'].has_key(s):
        #            self.s2v['other'][s].append(v.lower())
        print '\t\treading inf and req values from dialogues  ...'
        for data in self.dialog:
            dialog = data['dial']
            for turn in dialog:
                route_states = turn['state']['route']
                for state in route_states:
                    slot, value = state['slot'], state['value']
                    if slot in self.s2v['requestable']:
                        if not isinstance(value, list):
                            self.s2v['requestable'][slot].append(str(value).lower())
                        #else:
                            # for waypoints
                        #    self.s2v['requestable'][slot].append(' ==> '.join([x.lower() for x in value]))
                        # if isinstance(value, list):
                        #     value = ' ==> '.join(value)  # waypoints
                        # if value != '':
                        #     self.s2v['requestable'][slot].append(value.lower())
                if 'places' in turn['state']:
                    place_states = turn['state']['places']
                    for place in place_states:
                        #self.s2v['other']['place'].append(place['place'].lower())
                        for state in place['state']:
                            slot, value = state['slot'], state['value']
                            value = str(value)
                            if slot in self.s2v['requestable']:
                                self.s2v['requestable'][slot].append(value.lower())

        # sort and set values
        print '\t\tget unique and sort them ...'
        for s,vs in self.s2v['informable'].iteritems():
            self.s2v['informable'][s] = sorted(list(set(vs)))
        for s,vs in self.s2v['requestable'].iteritems():
            self.s2v['requestable'][s] = sorted(list(set(vs)))
        for s,vs in self.s2v['other'].iteritems():
            self.s2v['other'][s] = sorted(list(set(vs)))

        # Debug s2v
        print
        print '\t\t\tDebugging self.s2v values'
        for inf, values in self.s2v['informable'].iteritems():
            print '\t\t\t', inf, ':', ', '.join(values[:3]), '...', 'len:', len(values)
        for req, values in self.s2v['requestable'].iteritems():
            print '\t\t\t', req, ':', ', '.join(values[:3]), '...', 'len:', len(values)
        for ot, values in self.s2v['other'].iteritems():
            print '\t\t\t', ot, ':', ', '.join(values[:3]), '...', 'len:', len(values)
        print

        # make a 1-on-1 mapping for delexicalisation
        print '\t\tcreate 1-1 mapping for delexicatization ...'
        self.supervalues = []
        self.values = []
        self.slots  = []
        print '\t\t\tcreate 1-1 mapping for informables ...'
        for s,vs in self.s2v['informable'].iteritems():
             # adding slot delexicalisation
            self.supervalues.extend([s for x in self.semidict[s]])
            #self.values.extend([normalize(x) for x in self.semidict[s]])
            self.values.extend([x for x in self.semidict[s]])
            self.slots.extend(['[SLOT_'+s.upper()+']' for x in self.semidict[s]])
            # adding value delexicalisation
            for v in vs:
                self.supervalues.extend([v for x in self.semidict[v]])
                #self.values.extend([normalize(x) for x in self.semidict[v]])
                self.values.extend([x for x in self.semidict[v]])
                self.slots.extend(['[VALUE_'+s.upper()+']' for x in self.semidict[v]])

        print '\t\t\tcreate 1-1 mapping for requestables and other ...'
        for s,vs in self.s2v['requestable'].items() + self.s2v['other'].items():
            # adding value delexicalisation
            #self.values.extend([normalize(v) for v in vs])
            self.values.extend([v for v in vs])
            self.supervalues.extend([v for v in vs])
            self.slots.extend(['[VALUE_'+s.upper()+']' for v in vs])
            # adding slot delexicalisation
            self.supervalues.extend([s for x in self.semidict[s]])
            #self.values.extend([normalize(x) for x in self.semidict[s]])
            self.values.extend([x for x in self.semidict[s]])
            self.slots.extend(['[SLOT_'+s.upper()+']' for x in self.semidict[s]])

        #print '\t\t\tcreate 1-1 mapping for dont care values ...'
        # incorporate dontcare values
        #self.values.extend([normalize(v) for v in self.semidict['any']])
        self.values.extend([v for v in self.semidict['any']])
        self.supervalues.extend(['any' for v in self.semidict['any']])
        self.slots.extend(['[VALUE_ANY]' for v in self.semidict['any']])

        # sorting according to length
        print '\t\t\tsorting values, super values and slots according to length ...'
        self.values, self.supervalues, self.slots = zip(*sorted(\
                zip(self.values,self.supervalues,self.slots),\
                key=lambda x: len(x[0]),reverse=True))

        print
        print '\t\t\tDebugging delexicalized labels'
        print '\t\t\tself.values:', ', '.join(self.values[:3]), '...', 'len:', len(self.values)
        print '\t\t\tself.supervalues:', ', '.join(self.supervalues[:3]), '...', 'len:', len(self.supervalues)
        print '\t\t\tself.slots:', ', '.join(self.slots[:3]), '...', 'len:', len(self.slots)
        print

        #for i, value in enumerate(self.supervalues):
        #    print self.slots[i], self.supervalues[i], '\t', self.values[i]

        # for generating semantic labels
        self.infovs = []
        self.infoseg = [0]
        self.reqs = []
        self.reqseg = [0]
        self.dontcare = []
        self.refvs = []

        for place in sorted(self.s2v['other']['place']):
            self.refvs.append('ref=%s' % place)
        self.refvs.append('ref=none')

        print '\t\t\tgenerating informables semantic labels ...'
        for s in sorted(self.s2v['informable'].keys()):
            self.infovs.extend([s+'='+v for v in self.s2v['informable'][s]])
            self.infovs.append(s+'=any')    # changed dontcare to any
            self.infovs.append(s+'=none')
            self.infoseg.append(len(self.infovs))
            # dont care values and none values
            self.dontcare.append(len(self.infovs)-1)
            self.dontcare.append(len(self.infovs)-2)
        #for s in sorted(self.s2v['informable'].keys()):   # TODO: remove informables as requestables too?
        #    self.reqs.extend([s+'=exist',s+'=none'])
        #    self.reqseg.append(len(self.reqs))
        for s in sorted(self.s2v['requestable'].keys()):
            self.reqs.extend([s+'=exist',s+'=none'])
            self.reqseg.append(len(self.reqs))

        self.allvs = self.infovs + self.reqs

        # Debug
        print
        print '\t\t\tDebugging semantic labels'
        print '\t\t\tself.refvs:', self.refvs
        print '\t\t\tself.infovs:', self.infovs  #', '.join(self.infovs[:3]), '...', ', '.join(self.infovs[-3:]), 'len:', len(self.infovs)
        print '\t\t\tself.infoseg:', self.infoseg
        print '\t\t\tself.dontcare:', self.dontcare
        print '\t\t\tself.reqs:', self.reqs  #', '.join(self.reqs[:3]), '...', ', '.join(self.reqs[-3:]), 'len:', len(self.reqs)
        print '\t\t\tself.reqseg:', self.reqseg
        print

        # for ngram indexing
        print '\t\t\tgenerating ngram indexes ...'
        self.ngs2v = []
        for s in sorted(self.s2v['informable'].keys()):
            self.ngs2v.append( (s, self.s2v['informable'][s] + ['any','none']) )
        for s in sorted(self.s2v['informable'].keys()):
            self.ngs2v.append( (s,['exist','none']) )
        for s in sorted(self.s2v['requestable'].keys()):
            self.ngs2v.append( (s,['exist','none']) )

    def loadjson(self,filename):
        with open(filename) as data_file:
            #for i in range(5):
            #    data_file.readline()
            data = json.load(data_file)
        return data

    def _printStats(self):
        print '\n==============='
        print 'Data statistics'
        print '==============='
        print 'Train    : %d' % len(self.data['train'] )
        print 'Valid    : %d' % len(self.data['valid'] )
        print 'Test     : %d' % len(self.data['test']  )
        print '==============='
        print 'Voc      : %d' % len(self.vocab)
        if self.trkenc=='ng':
            print 'biGram:  : %d' % len(self.bigrams)
            print 'triGram: : %d' % len(self.trigrams)
        if self.trkenc=='ng':
            print 'All Ngram: %d' % len(self.ngrams)
        #print '==============='
        #print 'Venue    : %d' % len(self.db2inf)  # TODO: remove?
        #print '==============='

    def _setupData(self,percent):

        # zip corpus
        if self.trkenc=='ng':
            trksrc = self.ngram_source
            trktar = self.ngram_target
        else:
            trksrc = self.delsrcpos
            trktar = self.deltarpos

        corpus = [  self.sourceutts,        self.sourcecutoffs,
                    self.masked_sourceutts, self.masked_sourcecutoffs,
                    self.targetutts,        self.targetcutoffs,
                    self.masked_targetutts, self.masked_targetcutoffs,
                    self.snapshot_vecs,
                    self.changes,   self.goals,
                    self.info_semis,        self.req_semis,
                    np.array(self.db_logics), self.newtask,
                    trksrc,                 trktar,
                    self.ref_semis,         self.refmentions,
                    self.refsrcfeat,        self.reftarfeat,
                    self.finished,          self.sentGroupIndex]
        corpus = zip(*corpus)

        # split out train+valid
        train_valid = self.split.train_valid(corpus)

        # cut dataset according to percentage
        percent = float(percent)/float(100)
        train_valid = train_valid[:int(len(train_valid)*percent)]

        # split into train/valid/test
        self.data['train'] = self.split.train(train_valid)
        self.data['valid'] = self.split.valid(train_valid)
        self.data['test']  = self.split.test(corpus)


        # Print the shapes of every data piece
        sample_data = corpus[0]
        labels = ['source', 'source_len', 'masked_source', 'masked_source_len',
                  'target', 'target_len', 'masked_target', 'masked_target_len',
                  'snapshot', 'changes', 'goals', 'inf_labels', 'req_labels', 'db_logics', 'new_task',
                  'srcfeat', 'tarfeat', 'ref_labels', 'ref_mentions', 'refsrcfeat', 'reftarfeat',
                  'finished', 'sent_group']
        assert(len(labels) == len(sample_data))
        for i in range(len(sample_data)):
            print labels[i] + ':', np.array(sample_data[i]).shape

    def read(self,mode='train'):
        ## default implementation for read() function
        if self.mode!=mode:
            self.mode = mode
            index = 0

        # end of data , reset index & return None
        if self.index>=len(self.data[mode]):
            data = None
            self.index = 0

            if mode!='test': # train or valid, do shuffling
                if self.shuffle=='static': # just shuffle current set
                    random.shuffle(self.data[mode])
                elif self.shuffle=='dynamic':
                    # shuffle train + valid together
                    train_valid = self.data['train']+self.data['valid']
                    random.shuffle(train_valid)
                    self.data['train'] = self.split.train(train_valid)
                    self.data['valid'] = self.split.valid(train_valid)
            return data

        # 1 dialog at a time
        data = deepcopy(list(self.data[mode][self.index]))
        lengthen_idx = 1
        while   lengthen_idx<self.lengthen and \
                self.index+lengthen_idx<len(self.data[mode]):
            #lengthen the data by combining two data points
            nextdata = deepcopy(list(self.data[mode][self.index+lengthen_idx]))
            data = self.lengthenData(data,nextdata,mode)
            lengthen_idx += 1
        self.index += lengthen_idx
        return data

    def lengthenData(self,data,addon,mode):
        #for t in range(len(data[10])):
        #    print np.nonzero(np.array(data[10][t]))
        for i in range(len(data)): # for every data matrix
            if isinstance(data[i],list):
                idx = [0,2,4,6]
                if i in idx: # sequences, need padding
                    maxleng = max(len(data[i][0]),len(addon[i][0]))
                    for t in range(len(data[i])): # for each turn
                        data[i][t].extend([0]*(maxleng-len(data[i][t])))
                    for t in range(len(addon[i])): # for each turn
                        addon[i][t].extend([0]*(maxleng-len(addon[i][t])))
                idx = [8]
                if i in idx: # snapshot vectors
                    maxleng = max(len(data[i][0]),len(addon[i][0]))
                    for t in range(len(data[i])): # turn
                        data[i][t].extend([[-1 for cnt in \
                            range(len(data[i][t][0]))]]*(maxleng-len(data[i][t])))
                    for t in range(len(addon[i])):# turn
                        addon[i][t].extend([[-1 for cnt in \
                            range(len(addon[i][t][0]))]]*(maxleng-len(addon[i][t])))
                idx = [15,16]
                if i in idx: # srcfeat and tarfeat
                    maxleng = max(len(data[i][0][0][0]),len(addon[i][0][0][0]))
                    for t in range(len(data[i])): # turn
                        for x in range(len(data[i][t])): # slot or value
                            for sv in range(len(data[i][t][x])): # each value
                                data[i][t][x][sv].extend([-1]*\
                                    (maxleng-len(data[i][t][x][sv])))
                    for t in range(len(addon[i])):# turn
                        for x in range(len(addon[i][t])):# slot or value
                            for sv in range(len(addon[i][t][x])):# each value
                                addon[i][t][x][sv].extend([-1]*\
                                    (maxleng-len(addon[i][t][x][sv])))
                idx = [19, 20]
                if i in idx: # refsrcfeat and reftarfeat
                    maxleng = max(len(data[i][0][0]),len(addon[i][0][0]))
                    for t in range(len(data[i])): # turn
                        for sv in range(len(data[i][t])): # each value
                            data[i][t][sv].extend([-1]*\
                                (maxleng-len(data[i][t][sv])))
                    for t in range(len(addon[i])):# turn
                        for sv in range(len(addon[i][t])):# each value
                            addon[i][t][sv].extend([-1]*\
                                (maxleng-len(addon[i][t][sv])))
                data[i] = addon[i] + data[i]

        # propagte tracker labels
        for t in range(len(data[11])):
            for s in range(len(self.infoseg[:-1])):
                if t!=0 and data[11][t][self.infoseg[s]:self.infoseg[s+1]][-1]==1:
                    data[11][t][self.infoseg[s]:self.infoseg[s+1]] = \
                        data[11][t-1][self.infoseg[s]:self.infoseg[s+1]]
            #print np.nonzero(np.array(data[10][t]))
        #print np.array(data[0]).shape
        #raw_input()
        """
        for i in range(len(data)):
            try: data[i] = np.array(data[i],dtype='float32')
            except: pass
        """
        return data

    def iterate(self,mode='test',proc=True):
        # default implementation for iterate() function
        return self.data[mode]

    def structureDB(self):

        # all informable values
        print '\tformatting DB ...'

        # represent each db entry with informable values
        self.db2inf = []
        self.db2idx  = []
        self.idx2db = []
        self.idx2ent= {}
        for i in  range(len(self.db)):
            e = self.db[i]
            e2inf = []
            for s,v in e.iteritems():
                if s in self.s2v['informable']:
                    e2inf.append( self.infovs.index(s+'='+v) )
            e2inf = sorted(e2inf)

            # if not repeat, create new entry
            if e2inf not in self.db2inf:
                self.db2inf.append(e2inf)
                self.db2idx.append(len(self.db2inf)-1)
                self.idx2db.append([e2inf])
                self.idx2ent[self.db2inf.index(e2inf)] = [e]
            else: # if repeat, indexing back
                self.db2idx.append(self.db2inf.index(e2inf))
                self.idx2db[self.db2inf.index(e2inf)].append(e2inf)
                self.idx2ent[self.db2inf.index(e2inf)].append(e)

        # create hash for finding db index by name
        self.n2db = {}
        for i in range(len(self.db)):
            self.n2db[self.db[i]['name'].lower()] = self.db2idx[i]

    def loadVocab(self):

        # iterate through dialog and make vocab
        self.inputvocab = ['[VALUE_ANY]','[VALUE_COUNT]']  # changed DONTCARE to ANY
        self.outputvocab= ['[VALUE_ANY]','[VALUE_COUNT]']
        self.vocab = []

        # init inputvocab with informable values
        for s,vs in self.s2v['informable'].iteritems():
            for v in vs:
                if v=='none': continue
                self.inputvocab.extend(v.split())
            self.inputvocab.extend( ['[SLOT_'+s.upper()+']','[VALUE_'+s.upper()+']'])
            self.outputvocab.extend(['[SLOT_'+s.upper()+']','[VALUE_'+s.upper()+']'])

        # add every word in semidict into vocab
        for s in self.semidict.keys():
            for v in self.semidict[s]:
                self.inputvocab.extend(v.split())

        # print self.inputvocab
        # print len(self.inputvocab)
        # print
        # print self.outputvocab
        # print len(self.outputvocab)

        # for grouping sentences
        sentKeys = {}
        self.sentGroup= []

        # lemmatizer
        lmtzr = WordNetLemmatizer()

        # form lexican
        ivocab = []
        ovocab = []
        for i in range(len(self.dialog)):

            print '\tsetting up vocab, finishing ... %.2f%%\r' %\
                (100.0*float(i)/float(len(self.dialog))),
            sys.stdout.flush()

            # parsing dialog
            for j in range(len(self.dialog[i]['dial'])):
                # text normalisation
                #self.dialog[i]['dial'][j]['sys']['sent'] = normalize(
                #        self.dialog[i]['dial'][j]['sys']['sent'])
                #self.dialog[i]['dial'][j]['usr']['transcript'] = normalize(
                #        self.dialog[i]['dial'][j]['usr']['transcript'])
                # this turn
                turn = self.dialog[i]['dial'][j]

                # system side
                # Add start and end markers and returns a list

                # print turn['sys']['sent']

                words,_,_,_,_ = self.extractSeq(turn['sys']['tokens'], slotpos=turn['sys']['slotpos'],\
                    type='target',index=False, split=True)
                # add </s> to the front and back
                # delexicalise all words, values into [SLOT_<name]::supervalue
                # replace numbers with [VALUE_COUNT]
                # just looking at the delicalise words
                ovocab.extend(words)

                # sentence group key
                key = tuple(set(sorted(
                    [lmtzr.lemmatize(w) for w in words if w not in self.stopwords])))
                # get the root word for every non-stop word in the words

                if key in sentKeys:
                    sentKeys[key][1] += 1
                    self.sentGroup.append( sentKeys[key][0] )
                else:
                    sentKeys[key] = [len(sentKeys),1]
                    self.sentGroup.append( sentKeys[key][0] )

                # sentKeys is a dict of root-sentences, with each value storing
                # a tuple of (order_added, count)
                # self.sentGroup is a list containing the order_added of
                # this dialogue's root-sentence for every response

                # user side
                # words = self.delexicalise(turn['usr']['transcript'], turn['usr']['slotpos']).split()
                if 'usr' not in turn:
                    print self.dialog[i]['dialogue_id']
                    print turn
                mwords,words,_,_,_ = self.extractSeq(turn['usr']['tokens'], slotpos=turn['usr']['slotpos'],\
                    type='source',index=False, split=True)
                # CHANGED the turn from sys to usr, bug here
                # mworlds: delexicalised
                # words: lexicalised
                # now that, type is source, </s> is not added to the front and back

                # the weird thing here is, why extract the sys response here..
                # trying out to see if there is any diff if we weak
                ivocab.extend(mwords)
                #ivocab.extend(words)
                """
                for hyp in t['usr']['asr']:
                    words = self.delexicalise(normalize(hyp['asr-hyp'])).split()
                    ivocab.extend(words)
                """
        print

        # print ivocab[:20]
        # print len(ivocab)
        # print
        # print ovocab[:20]
        # print len(ovocab)
        # print
        # print self.sentGroup
        # print len(self.sentGroup)

        # re-assigning sentence group w.r.t their frequency
        mapping = {}
        idx = 0
        cnt = 0
        # process from most frequent sentence group first
        # creates a mapping dict, where the key is the order_added, and the value is
        # the index that is newly created
        # it then checks for the latent, by default during training, the value is 100
        # but for tracker, it is 0, and since idx will never < 0, the count is not incremented
        # i.e. tracker, cnt = 0
        # else, the cnt is incremented only for the first dl most frequent setence
        for key,val in sorted(sentKeys.iteritems(),key=lambda x:x[1][1],reverse=True):
            mapping[val[0]] = idx
            #print idx, val[1], key
            if idx < self.dl - 1:
                cnt+=val[1]
            idx += 1
        #raw_input()
        # because self.dl is 1, so idx will never be lesser than 0
        # thus cnt is always 0 (for policy config)
        # check NDM later...

        print '\tsemi-supervised action examples: %2.2f%%' % \
                (float(cnt)/float(len(self.sentGroup))*100)
        # here, for every sent-group, the count value is being replaced by
        # the minimum of the (index, and self.dl - 1), no more tuple of (order, count)
        # i.e. each sentGroup is an index denoting the highest frequency group,
        # but limited by self.dl - 1, being the last index with the lowest frequency
        # sentGroup
        for i in range(len(self.sentGroup)):
            self.sentGroup[i] = min(mapping[self.sentGroup[i]],self.dl-1)

        # set threshold for input vocab
        counts = dict()
        for w in ivocab:
            counts[w] = counts.get(w, 0) + 1
        self.inputvocab = ['<unk>','</s>'] + \
                sorted(list(set(self.inputvocab+\
                [w for w,c in sorted(counts.iteritems(),key=operator.itemgetter(1)) if c>1])))
        #print '\t\t\t"wonder in input?"', self.inputvocab.index('wonder')

        # set threshold for output vocab
        counts = dict()
        for w in ovocab:
            counts[w] = counts.get(w, 0) + 1
        self.outputvocab = ['<unk>','</s>'] + \
                sorted(list(set(self.outputvocab+['thank','you','goodbye']+\
                [w for w,c in sorted(counts.iteritems(),key=operator.itemgetter(1))])))
        #print '\t\t\t"wonder in output?"', self.inputvocab.index('wonder')

        print '\t\t\tinput vocab len:', len(self.inputvocab)
        print '\t\t\toutput vocab len:', len(self.outputvocab)

        # the whole vocab
        self.vocab = ['<unk>','</s>'] + \
                list(set(self.inputvocab[2:]).union(self.outputvocab[2:]))

        print '\t\t\toverall vocab len:', len(self.vocab)
        print '\t\t\tvocab:', self.vocab[:10], '...'

        # create snapshot dimension
        self.snapshots = ['OFFERED','CHANGED']
        for w in self.outputvocab:
            if w.startswith('[VALUE'):
                self.snapshots.append(w)
        self.snapshots = sorted(self.snapshots)

    def parseGoal(self):
        # parse goal into dict format
        self.goals = []
        # for computing corpus success
        requestables = self.s2v['requestable'].keys()
        #requestables = ['phone','address','postcode','food','area','pricerange']  # TODO: edit requestables
        vmc, success = 0., 0.
        # for each dialog
        for i in range(len(self.dialog)):
            d = self.dialog[i]
            dialog_goals = []
            for j in range(len(d['goal']['constraints'])):  # j number of tasks
                goal = [np.zeros(self.infoseg[-1]),
                        np.zeros(self.reqseg[-1])]
                # tuple (size(infovs), size(reqs))
                for s2v in d['goal']['constraints'][j]:
                    s,v = s2v
                    v = str(v).lower()
                    s2v = s+'='+v
                    if v!='any' and v!='none':
                        #goal['inf'].append( self.infovs.index(s2v) )
                        goal[0][self.infovs.index(s2v)] = 1

                # goal[0]:  [0, 0, 0, 1, 0, ... ]
                for s in d['goal']['request_slots'][j]:
                    #if s=='pricerange' or s=='area' or s=='food':
                    #    continue
                    #goal['req'].append(self.reqs.index(s+'=exist'))
                    s = s.split(':')[0]  # remove the value
                    goal[1][self.reqs.index(s+'=exist')] = 1
                dialog_goals.append(goal)

            # goal[1]: [0, 0, 1, 0, ...]
            # now dialog goal is a list of goal, with each goal to each task
            self.goals.append(dialog_goals)

            # compute corpus success
            # m_targetutt = self.masked_targetutts[i]
            # m_targetutt_len = self.masked_targetcutoffs[i]
            # m_sourceutt = self.masked_sourceutt[i]
            # new_task = False
            #
            # # for computing success
            # all_offered = []
            # all_requests = []
            #
            #
            # offered = False
            # requests= []  # contains index of requestable=exists
            # # iterate each turn
            # for t in range(len(m_targetutt)):
            #
            #     check = self.vocab[m_sourceutt[t][0]
            #     if check == 'oh':
            #         all_offered.append(offered)
            #         all_requests.append(requests)
            #         offered = False
            #         requests = []
            #
            #     sent_t = [self.vocab[w] for w in
            #               m_targetutt[t][:m_targetutt_len[t]]][1:-1]
            #             # remove the </s> front and back
            #     print sent_t
            #
            #     if '[VALUE_PLACE]' in sent_t: offered=True
            #     for requestable in requestables:
            #         if '[VALUE_'+requestable.upper()+']' in sent_t:
            #             requests.append(self.reqs.index(requestable+'=exist'))
            # all_offered.append(offered)
            # all_requests.append(requests)

            # compute success
        #     if all(all_offered):
        #         vmc += 1.
        #         # if set(index) >= set(goal indexes)
        #         for i, request in enumerate(all_requests):
        #             if set(requests).issuperset(set(dialog_goals[i][1].nonzero()[0].tolist())):
        #                 success += 1.
        #
        # # value match criteria??
        # # success: match and given requests
        # print '\tCorpus VMC       : %2.2f%%' % (vmc/float(len(self.dialog))*100)
        # print '\tCorpus Success   : %2.2f%%' % (success/float(len(self.dialog))*100)
        #print self.goals
        #print
        #print len(self.goals)
