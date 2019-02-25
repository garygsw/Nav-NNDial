#!/bin/sh

#source activate python2.7
rm model/KvretNav.*

echo $(date) "kvret tracker start"  >> kvret.log
nohup python kvret_nndial.py -config config/Kvret_tracker.cfg -mode train
sleep 5
echo $(date) "kvret tracker end"  >> kvret.log

echo $(date) "kvret NDM start"  >> kvret.log
cp model/KvretNav.tracker.model model/KvretNav.NDM.model
nohup python kvret_nndial.py -config config/Kvret-NDM.cfg -mode adjust
sleep 5
python kvret_nndial.py -config config/Kvret-NDM.cfg -mode test >> kvret-NDM.log
sleep 5
echo $(date) "kvret NDM end"  >> kvret.log

echo $(date) "kvret Att-NDM start"  >> kvret.log
cp model/KvretNav.tracker.model model/KvretNav.Att-NDM.model
nohup python kvret_nndial.py -config config/Kvret-Att-NDM.cfg -mode adjust
sleep 5
python kvret_nndial.py -config config/Kvret-Att-NDM.cfg -mode test >> kvret-Att-NDM.log
sleep 5
echo $(date) "kvret Att-NDM end"  >> kvret.log

echo $(date) "kvret LIDM start"  >> kvret.log
cp model/KvretNav.tracker.model model/KvretNav.LIDM.model
nohup python kvret_nndial.py -config config/Kvret-LIDM.cfg -mode adjust
sleep 5
python kvret_nndial.py -config config/Kvret-LIDM.cfg -mode test >> kvret-LIDM.log
sleep 5
echo $(date) "kvret LIDM end"  >> kvret.log

echo $(date) "kvret LIDM-RL start"  >> kvret.log
cp model/KvretNav.tracker.model model/KvretNav.LIDM-RL.model
nohup python kvret_nndial.py -config config/Kvret-LIDM-RL.cfg -mode rl
sleep 5
python kvret_nndial.py -config config/Kvret-LIDM-RL.cfg -mode test >> kvret-LIDM-RL.log
sleep 5
echo $(date) "kvret LIDM-RL end"  >> kvret.log
