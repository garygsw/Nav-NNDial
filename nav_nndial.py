######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################
import sys
import os

from utils.commandparser import NNSDSOptParser
from nn.NavNNDialogue import NNDial

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

from datetime import datetime

if __name__ == '__main__':

    start_time = datetime.now()
    print 'start time:', start_time.strftime('%H:%M:%S')

    args = NNSDSOptParser()
    config = args.config

    model = NNDial(config,args)
    if args.mode=='train' or args.mode=='adjust':
        model.trainNet()
    elif args.mode=='test' or args.mode=='valid':
        model.testNet()
    elif args.mode=='interact':
        while True: model.dialog()
    elif args.mode=='rl':
        model.trainNetRL()


    end_time = datetime.now()
    print 'end time:', end_time.strftime('%H:%M:%S')
    elasped_seconds = int((end_time - start_time).total_seconds())

    hours = elasped_seconds // 3600
    elasped_seconds %= 3600
    minutes = elasped_seconds // 60
    elasped_seconds %= 60
    seconds = elasped_seconds
    print 'Total training time: %d hrs %d mins %.6f s' % (hours, minutes,seconds)
