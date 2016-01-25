#!/bin/bash

 # Startup
 cd /training/results/MC12/test/NB-NoPreSels-NumSPDR1R2/DIAG/ENTROPY/ScaleF1.2/LD1/BFGS/621/Ghost/Downstream
 rm -vf TRAINING-COMPLETE
 date               > TRAINING-RUNNING
 uname -a          >> TRAINING-RUNNING
 env | grep "LSB_" >> TRAINING-RUNNING
 cat TRAINING-RUNNING

 unset LD_LIBRARY_PATH
 echo CMTCONFIG=$CMTCONFIG 
 if [ 1 == "0" ]; then
  source /lhcb/scripts/lhcb-setup-afs.sh
 else
  source /afs/cern.ch/lhcb/software/releases/LBSCRIPTS/prod/InstallArea/scripts/LbLogin.sh -c x86_64-slc5-gcc46-opt
 fi
 source /afs/cern.ch/lhcb/software/releases/LBSCRIPTS/LBSCRIPTS_v8r4p2/InstallArea/scripts/SetupProject.sh Brunel v49r1
 cd /afs/cern.ch/user/d/derkach/cmtuser/Brunel_v49r1/Rec/ChargedProtoANNPIDTeacher/cmt
 source ./setup.sh
 cd /training/results/MC12/test/NB-NoPreSels-NumSPDR1R2/DIAG/ENTROPY/ScaleF1.2/LD1/BFGS/621/Ghost/Downstream

 # Copy in training parameters
 cp -v /training/configs/training/NB-test.txt train-config.txt
 for i in ; do
  cp -v /training/configs/TrackSelection/$i .
 done
 # MVA parameters file
 cp -v /training/configs/training//MVA-Configuration.txt MVA.txt
 echo ""  >> MVA.txt
 echo "# Custom MVA parameters"    >> MVA.txt
 echo "HiddenLayerScaleFactor = 1.2" >> MVA.txt
 if [ $MVATYPE == "NB" ]; then
  echo "NBLOSS      = ENTROPY"      >> MVA.txt
  echo "NBMETHOD    = BFGS"    >> MVA.txt
  echo "NBSHAPE     = DIAG"     >> MVA.txt
  echo "NBLEARNDIAG = 1" >> MVA.txt
  echo "NBPRE       = 621"       >> MVA.txt
 else
  echo "TMVAMETHOD           = MLP"           >> MVA.txt
  echo "TMVAMLPNCYCLES       = 500"       >> MVA.txt
  echo "TMVAVARTRANSFORM     = Norm"     >> MVA.txt
  echo "TMVAMLPNEURONTYPE    = sigmoid"    >> MVA.txt
  echo "TMVAMLPMETHOD        = BP"        >> MVA.txt
  echo "TMVAMLPESTIMATORTYPE = CE" >> MVA.txt
  echo "TMVABDTBOOSTTYPE     = AdaBoost"     >> MVA.txt
  echo "TMVABDTNTREES        = 800"        >> MVA.txt
  echo "TMVABDTPRUNEMETHOD   = CostComplexity"   >> MVA.txt
  echo "TMVABDTMAXTREEDEPTH  = 3"  >> MVA.txt
 fi
 export NNCONFIGNAME=NB-NoPreSels-NumSPDR1R2
 if [ $ONLXPLUS == "1" ]; then
  cp -v /training/configs/training//MC12-DataFiles-castor.txt datafiles.txt
 else
  cp -v /training/configs/training//MC12-DataFiles-Cambridge.txt datafiles.txt
 fi
 echo "Training in /training/results/MC12/test/NB-NoPreSels-NumSPDR1R2/DIAG/ENTROPY/ScaleF1.2/LD1/BFGS/621/Ghost/Downstream"
 sleep 2
 
 # Define local executable name
 export TRAINER=./ProtoPIDTeacher.exe

 #echo Running ldd on ProtoPIDTeacher.exe
 #ldd ProtoPIDTeacher.exe

 # Run valgrind profiling
 #source /afs/cern.ch/lhcb/group/rich/vol4/jonrob/scripts/new-valgrind.sh
 #export TRAINER="valgrind --tool=callgrind -v --dump-instr=yes --trace-jump=yes "$TRAINER

 # Run training
 rm -fv ahist.txt *.conf *-evaluation.root Downstream-Ghost-analysis-hist.txt *.xml *.C *-TMVA.root
 cp -v /training/configs/networks/NB-NoPreSels-NumSPDR1R2/GlobalPID_Ghost_Downstream_ANN.txt .
 sleep 2
 $TRAINER /training/results/MC12/test/NB-NoPreSels-NumSPDR1R2/DIAG/ENTROPY/ScaleF1.2/LD1/BFGS/621/Ghost/Downstream/GlobalPID_Ghost_Downstream_ANN.txt /training/results/MC12/test/NB-NoPreSels-NumSPDR1R2/DIAG/ENTROPY/ScaleF1.2/LD1/BFGS/621/Ghost/Downstream/train-config.txt /training/results/MC12/test/NB-NoPreSels-NumSPDR1R2/DIAG/ENTROPY/ScaleF1.2/LD1/BFGS/621/Ghost/Downstream/datafiles.txt /training/results/MC12/test/NB-NoPreSels-NumSPDR1R2/DIAG/ENTROPY/ScaleF1.2/LD1/BFGS/621/Ghost/Downstream/MVA.txt "Yes" "No"
 if [ $MVATYPE == "NB" ]; then
  mv -v ahist.txt Downstream-Ghost-analysis-hist.txt
 fi
 rm -fv *-TMVA.root
 sleep 2

 # (Re)Define local executable name (remove any valgrind settings)
 #export TRAINER=./ProtoPIDTeacher.exe

 # Run evaluation
 rm -rfv png eps pdf text Downstream-Ghost-analysis.* *-InputOutputData.* 
 $TRAINER /training/results/MC12/test/NB-NoPreSels-NumSPDR1R2/DIAG/ENTROPY/ScaleF1.2/LD1/BFGS/621/Ghost/Downstream/GlobalPID_Ghost_Downstream_ANN.txt /training/results/MC12/test/NB-NoPreSels-NumSPDR1R2/DIAG/ENTROPY/ScaleF1.2/LD1/BFGS/621/Ghost/Downstream/train-config.txt /training/results/MC12/test/NB-NoPreSels-NumSPDR1R2/DIAG/ENTROPY/ScaleF1.2/LD1/BFGS/621/Ghost/Downstream/datafiles.txt /training/results/MC12/test/NB-NoPreSels-NumSPDR1R2/DIAG/ENTROPY/ScaleF1.2/LD1/BFGS/621/Ghost/Downstream/MVA.txt "No"  "Yes"
 #bzip2 -v *-InputOutputData.txt
 sleep 2
 if [ $MVATYPE == "NB" ]; then
  ./ProtoPIDAsciiToRoot.exe Downstream-Ghost-analysis-hist.txt Downstream-Ghost-analysis.root
  sleep 2
  ./ProtoPIDAnalysis.exe Downstream-Ghost-analysis.root Downstream-Ghost-analysis.pdf sort Downstream-Ghost-analysis-correl.txt
  sleep 2
 fi

 # Clean up temporary files
 rm -fv train-config.txt MVA.txt datafiles.txt

 # Send an email
 echo "http://cern.ch/jonrob/LHCb/ANNPID/MC12/test/NB-NoPreSels-NumSPDR1R2/DIAG/ENTROPY/ScaleF1.2/LD1/BFGS/621/Ghost/Downstream" | mail -s "[ChargedProtoANNPIDTeacher] Training Complete /training/results/MC12/test/NB-NoPreSels-NumSPDR1R2/DIAG/ENTROPY/ScaleF1.2/LD1/BFGS/621/Ghost/Downstream" jonrob

 # Finish
 uname -a > TRAINING-COMPLETE
 rm -fv TRAINING-RUNNING
 date 
 rm -f run.sh 
 rm -f *.exe
