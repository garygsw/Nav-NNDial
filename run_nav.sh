#!/bin/sh

#source activate python2.7
rm model/MTNav.*

echo $(date) "MTNav tracker start"  >> mtnav.log
nohup python nav_nndial.py -config config/Nav-tracker.cfg -mode train
sleep 60
echo $(date) "MTNav tracker end"  >> mtnav.log
mail -s "MTNav-tracker run completed" garygsw@gmail.com < /dev/null

echo $(date) "MTNav NDM start"  >> mtnav.log
cp model/MTNav.tracker.model model/MTNav.NDM.model
nohup python nav_nndial.py -config config/Nav-NDM.cfg -mode adjust
sleep 60
python nav_nndial.py -config config/Nav-NDM.cfg -mode test > MTNav-NDM.log
sleep 60
echo $(date) "MTNav NDM end"  >> mtnav.log
mail -s "MTNav-NDM run completed" garygsw@gmail.com < /dev/null

echo $(date) "MTNav Att-NDM start"  >> mtnav.log
cp model/MTNav.tracker.model model/MTNav.Att-NDM.model
nohup python nav_nndial.py -config config/Nav-Att-NDM.cfg -mode adjust
sleep 60
python nav_nndial.py -config config/Nav-Att-NDM.cfg -mode test > MTNav-Att-NDM.log
sleep 60
echo $(date) "MTNav Att-NDM end"  >> mtnav.log
mail -s "Att-NDM run completed" garygsw@gmail.com < /dev/null

echo $(date) "MTNav LIDM start"  >> mtnav.log
cp model/MTNav.tracker.model model/MTNav.LDM.model
nohup python nav_nndial.py -config config/Nav-LIDM.cfg -mode adjust
sleep 60
python nav_nndial.py -config config/Nav-LIDM.cfg -mode test > MTNav-LIDM.log
sleep 60
echo $(date) "MTNav LIDM end"  >> mtnav.log
mail -s "MTNav-LIDM run completed" garygsw@gmail.com < /dev/null

echo $(date) "MTNav LIDM-RL start"  >> mtnav.log
cp model/MTNav.LDM.model model/MTNav.LDM-RL.model
nohup python nav_nndial.py -config config/Nav-LIDM-RL.cfg -mode rl
sleep 60
python nav_nndial.py -config config/Nav-LIDM-RL.cfg -mode test > MTNav-LIDM-RL.log
sleep 60
echo $(date) "MTNav LIDM-RL end"  >> mtnav.log
mail -s "MTNav-LIDM-RL run completed" garygsw@gmail.com < /dev/null
