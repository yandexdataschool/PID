#!/bin/bash

export RUN=${CHARGEDPROTOANNPIDTEACHERROOT}/training/Run.sh

export TMVAVARTRANSFORM="Norm"
#export TMVAVARTRANSFORM="Decorrelate"
#export TMVAVARTRANSFORM="Decorrelate,Norm"
#export TMVAVARTRANSFORM="None"

export TMVABDTBOOSTTYPE="AdaBoost"
#export TMVABDTBOOSTTYPE="Bagging"

export TMVABDTPRUNEMETHOD="NoPruning"
#export TMVABDTPRUNEMETHOD="CostComplexity"
#export TMVABDTPRUNEMETHOD="ExpectedError"

export TMVABDTMAXTREEDEPTH="3"

for TKPRESEL in "NoPreSels" "RejGhosts" ; do
 for TRAINCONFIGNAME in "TrainAllTks-EvalAllTks-NoReweight" "TrainAllTks-EvalAllTks-ReweightRICH2" ; do

  $RUN -TRAINOPTS ${TRAINCONFIGNAME} -MVATYPE TMVA -TMVAMETHOD BDT -TMVAVARTRANSFORM $TMVAVARTRANSFORM -TMVABDTBOOSTTYPE $TMVABDTBOOSTTYPE -TMVABDTNTREES 800 -TMVABDTMAXTREEDEPTH $TMVABDTMAXTREEDEPTH -TMVABDTPRUNEMETHOD $TMVABDTPRUNEMETHOD -CONFIGNAME ${TKPRESEL}-NoGECs

 done
done

exit 0
