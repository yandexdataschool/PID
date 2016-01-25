#!/bin/bash

# Host name
export HOST=`hostname`
if [[ $HOST = *lxplus* ]]; then
 export ONLXPLUS="1"
else
 export ONLXPLUS="0"
fi

# General options
export TRAINCONFIGNAME="test"

# General Neural Network Options
export MLPHIDDENLAYERSCALEF=1.2

# NeuroBayes Defaults
export NBLOSS="ENTROPY"
export NBMETHOD="BFGS"
export NBPRE=621
export NBLEARNDIAG=1
export NBSHAPE="DIAG"
export CONFIGNAME="NoPreSels-NumSPDR1R2"
export MVATYPE="NB"

# TMVA Defaults 
# MLP
export TMVAMETHOD="MLP"
export TMVAMLPNCYCLES="500"
export TMVAMLPNEURONTYPE="sigmoid"
export TMVAMLPMETHOD="BP"
export TMVAMLPESTIMATORTYPE="CE"
# BDT
export TMVABDTBOOSTTYPE="AdaBoost"
export TMVABDTNTREES="800"
export TMVAVARTRANSFORM="Norm"
export TMVABDTPRUNEMETHOD="CostComplexity"
export TMVABDTMAXTREEDEPTH="3"

# Read options
eval set -- "$@"
while [ $# -gt 0 ]
do
 case "$1" in
  -MLPHIDDENLAYERSCALEF) MLPHIDDENLAYERSCALEF=$2;  shift 2 ;;
  -NBLOSS)         NBLOSS=$2;       shift 2 ;;
  -NBMETHOD)       NBMETHOD=$2 ;    shift 2 ;;
  -NBPRE)          NBPRE=$2 ;       shift 2 ;;
  -NBLEARNDIAG)    NBLEARNDIAG=$2 ; shift 2 ;;
  -NBSHAPE)        NBSHAPE=$2 ;     shift 2 ;;
  -TMVAMETHOD)           TMVAMETHOD=$2 ;           shift 2 ;;
  -TMVAMLPNCYCLES)       TMVAMLPNCYCLES=$2 ;       shift 2 ;;
  -TMVAMLPNEURONTYPE)    TMVAMLPNEURONTYPE=$2 ;    shift 2 ;;
  -TMVAMLPMETHOD)        TMVAMLPMETHOD=$2 ;        shift 2 ;;
  -TMVABDTBOOSTTYPE)     TMVABDTBOOSTTYPE=$2 ;     shift 2 ;;
  -TMVABDTNTREES)        TMVABDTNTREES=$2 ;        shift 2 ;;
  -TMVAVARTRANSFORM)     TMVAVARTRANSFORM=$2 ;     shift 2 ;;
  -TMVABDTPRUNEMETHOD)   TMVABDTPRUNEMETHOD=$2 ;   shift 2 ;;
  -TMVABDTMAXTREEDEPTH)  TMVABDTMAXTREEDEPTH=$2 ;  shift 2 ;;
  -TMVAMLPESTIMATORTYPE) TMVAMLPESTIMATORTYPE=$2 ; shift 2 ;;
  -CONFIGNAME)     CONFIGNAME=$2 ;       shift 2 ;;
  -TRAINOPTS)      TRAINCONFIGNAME=$2 ;  shift 2 ;;
  -MVATYPE)        MVATYPE=$2 ;          shift 2 ;;
   *) echo "Internal error!" ; exit 1 ;;
 esac
done

export NNCONFIGNAME=$MVATYPE"-"$CONFIGNAME

export CMTCONFIG=x86_64-slc5-gcc46-opt
#export CMTCONFIG=x86_64-slc5-gcc46-dbg
#export CMTCONFIG=x86_64-slc6-gcc46-opt

export BVER=v49r1
if [ $ONLXPLUS == "0" ]; then
 source /lhcb/scripts/lhcb-setup-afs.sh
else
 source /afs/cern.ch/lhcb/software/releases/LBSCRIPTS/prod/InstallArea/scripts/LbLogin.sh -c $CMTCONFIG
fi
source `which SetupProject.sh` Brunel $BVER
cd $User_release_area/Brunel_${BVER}/Rec/ChargedProtoANNPIDTeacher/cmt
source setup.sh

# Executables
export TRAINER=ProtoPIDTeacher.exe
export ASCIITOROOT=ProtoPIDAsciiToRoot.exe
export ANALYSIS=ProtoPIDAnalysis.exe

# Check NB license, if needed

# Overall root dir
export MAINROOT="/afs/cern.ch/user/d/derkach/cmtuser/Brunel_v49r1/Rec/ChargedProtoANNPIDTeacher/training"

# Config directory
export CONFIGDIR=${MAINROOT}"/configs"

# Traing Data
export DATAFILES="MC12"

# network tunings
export NNCONFIGDIR=${CONFIGDIR}"/networks/"${NNCONFIGNAME}

# Main training directory
export TRAINLOC=${DATAFILES}"/"${TRAINCONFIGNAME}"/"${NNCONFIGNAME}
if [ $MVATYPE == "NB" ]; then
 export TRAINLOC=${TRAINLOC}"/"${NBSHAPE}"/"${NBLOSS}"/ScaleF"${MLPHIDDENLAYERSCALEF}"/LD"${NBLEARNDIAG}"/"${NBMETHOD}"/"${NBPRE}
else
 if [ $TMVAMETHOD == "MLP" ]; then
  export TRAINLOC=${TRAINLOC}"/"$TMVAMETHOD"/"$TMVAVARTRANSFORM"/ScaleF"${MLPHIDDENLAYERSCALEF}"/"$TMVAMLPMETHOD"/NCycles"${TMVAMLPNCYCLES}"/"${TMVAMLPESTIMATORTYPE}"/"$TMVAMLPNEURONTYPE
 else
  export TRAINLOC=${TRAINLOC}"/"$TMVAMETHOD"/"$TMVAVARTRANSFORM"/"$TMVABDTBOOSTTYPE"/NTrees"$TMVABDTNTREES"/MaxDepth"$TMVABDTMAXTREEDEPTH"/"$TMVABDTPRUNEMETHOD
 fi
fi
export TRAINDIR=${MAINROOT}"/results/"${TRAINLOC}
if [ $ONLXPLUS == "0" ]; then
 export TRAINWWW="http://www.hep.phy.cam.ac.uk/~jonesc/lhcb/PID/ANNPIDTraining/"${TRAINLOC}
 export EMAILUSER=jonesc
else
 export TRAINWWW="http://cern.ch/jonrob/LHCb/ANNPID/"${TRAINLOC}
 export EMAILUSER=jonrob
fi
mkdir -p $TRAINDIR

# training Config file
export TRAINCONFIGDIR=${CONFIGDIR}"/training/"
export TRAINCONFIG=${TRAINCONFIGDIR}${MVATYPE}"-"${TRAINCONFIGNAME}".txt"

train()
{
 export PARTICLE=$1
 export TRACK=$2
 export WORKPATH=${TRAINDIR}"/"${PARTICLE}"/"${TRACK}
 export NETCONFIG="GlobalPID_"${PARTICLE}"_"${TRACK}"_ANN.txt"
 export LOGFILE=${TRACK}-${PARTICLE}.log
 mkdir -p $WORKPATH
 cd $WORKPATH
 rm -rf LSFJOB_*
 # Make local copies of the binaries
 cp -f `which $TRAINER` .
 if [ $ONLXPLUS == "1" ]; then
  cp -f `which $ASCIITOROOT` .
  cp -f `which $ANALYSIS` .
 fi
 rm -f $LOGFILE run.sh
cat > "run.sh" << EOF
#!/bin/bash

 # Startup
 cd $WORKPATH
 rm -vf TRAINING-COMPLETE
 date               > TRAINING-RUNNING
 uname -a          >> TRAINING-RUNNING
 env | grep "LSB_" >> TRAINING-RUNNING
 cat TRAINING-RUNNING

 unset LD_LIBRARY_PATH
 echo CMTCONFIG=\$CMTCONFIG 
 if [ $ONLXPLUS == "0" ]; then
  source /lhcb/scripts/lhcb-setup-afs.sh
 else
  source /afs/cern.ch/lhcb/software/releases/LBSCRIPTS/prod/InstallArea/scripts/LbLogin.sh -c $CMTCONFIG
 fi
 source `which SetupProject.sh` Brunel $BVER
 cd $User_release_area/Brunel_${BVER}/Rec/ChargedProtoANNPIDTeacher/cmt
 source ./setup.sh
 cd $WORKPATH

 # Copy in training parameters
 cp -v ${TRAINCONFIG} train-config.txt
 for i in `grep ".txt" ${NNCONFIGDIR}/${NETCONFIG}`; do
  cp -v ${CONFIGDIR}/TrackSelection/\$i .
 done
 # MVA parameters file
 cp -v ${TRAINCONFIGDIR}/MVA-Configuration.txt MVA.txt
 echo ""  >> MVA.txt
 echo "# Custom MVA parameters"    >> MVA.txt
 echo "HiddenLayerScaleFactor = $MLPHIDDENLAYERSCALEF" >> MVA.txt
 if [ \$MVATYPE == "NB" ]; then
  echo "NBLOSS      = $NBLOSS"      >> MVA.txt
  echo "NBMETHOD    = $NBMETHOD"    >> MVA.txt
  echo "NBSHAPE     = $NBSHAPE"     >> MVA.txt
  echo "NBLEARNDIAG = $NBLEARNDIAG" >> MVA.txt
  echo "NBPRE       = $NBPRE"       >> MVA.txt
 else
  echo "TMVAMETHOD           = $TMVAMETHOD"           >> MVA.txt
  echo "TMVAMLPNCYCLES       = $TMVAMLPNCYCLES"       >> MVA.txt
  echo "TMVAVARTRANSFORM     = $TMVAVARTRANSFORM"     >> MVA.txt
  echo "TMVAMLPNEURONTYPE    = $TMVAMLPNEURONTYPE"    >> MVA.txt
  echo "TMVAMLPMETHOD        = $TMVAMLPMETHOD"        >> MVA.txt
  echo "TMVAMLPESTIMATORTYPE = $TMVAMLPESTIMATORTYPE" >> MVA.txt
  echo "TMVABDTBOOSTTYPE     = $TMVABDTBOOSTTYPE"     >> MVA.txt
  echo "TMVABDTNTREES        = $TMVABDTNTREES"        >> MVA.txt
  echo "TMVABDTPRUNEMETHOD   = $TMVABDTPRUNEMETHOD"   >> MVA.txt
  echo "TMVABDTMAXTREEDEPTH  = $TMVABDTMAXTREEDEPTH"  >> MVA.txt
 fi
 export NNCONFIGNAME=$NNCONFIGNAME
 if [ \$ONLXPLUS == "1" ]; then
  cp -v ${TRAINCONFIGDIR}/${DATAFILES}-DataFiles-castor.txt datafiles.txt
 else
  cp -v ${TRAINCONFIGDIR}/${DATAFILES}-DataFiles-Cambridge.txt datafiles.txt
 fi
 echo "Training in $WORKPATH"
 sleep 2
 
 # Define local executable name
 export TRAINER=./$TRAINER

 #echo Running ldd on $TRAINER
 #ldd $TRAINER

 # Run valgrind profiling
 #source /afs/cern.ch/lhcb/group/rich/vol4/jonrob/scripts/new-valgrind.sh
 #export TRAINER="valgrind --tool=callgrind -v --dump-instr=yes --trace-jump=yes "\$TRAINER

 # Run training
 rm -fv ahist.txt *.conf *-evaluation.root ${TRACK}-${PARTICLE}-analysis-hist.txt *.xml *.C *-TMVA.root
 cp -v ${NNCONFIGDIR}/${NETCONFIG} .
 sleep 2
 \$TRAINER ${WORKPATH}/${NETCONFIG} ${WORKPATH}/train-config.txt ${WORKPATH}/datafiles.txt ${WORKPATH}/MVA.txt "Yes" "No"
 if [ \$MVATYPE == "NB" ]; then
  mv -v ahist.txt ${TRACK}-${PARTICLE}-analysis-hist.txt
 fi
 rm -fv *-TMVA.root
 sleep 2

 # (Re)Define local executable name (remove any valgrind settings)
 #export TRAINER=./$TRAINER

 # Run evaluation
 rm -rfv png eps pdf text ${TRACK}-${PARTICLE}-analysis.* *-InputOutputData.* 
 \$TRAINER ${WORKPATH}/${NETCONFIG} ${WORKPATH}/train-config.txt ${WORKPATH}/datafiles.txt ${WORKPATH}/MVA.txt "No"  "Yes"
 #bzip2 -v *-InputOutputData.txt
 sleep 2
 if [ \$MVATYPE == "NB" ]; then
  ./$ASCIITOROOT ${TRACK}-${PARTICLE}-analysis-hist.txt ${TRACK}-${PARTICLE}-analysis.root
  sleep 2
  ./$ANALYSIS ${TRACK}-${PARTICLE}-analysis.root ${TRACK}-${PARTICLE}-analysis.pdf sort ${TRACK}-${PARTICLE}-analysis-correl.txt
  sleep 2
 fi

 # Clean up temporary files
 rm -fv train-config.txt MVA.txt datafiles.txt

 # Send an email
 echo "${TRAINWWW}/${PARTICLE}/${TRACK}" | mail -s "[ChargedProtoANNPIDTeacher] Training Complete $WORKPATH" $EMAILUSER

 # Finish
 uname -a > TRAINING-COMPLETE
 rm -fv TRAINING-RUNNING
 date 
 rm -f run.sh 
 rm -f *.exe
EOF
 echo "Starting job for" $WORKPATH
 if [ $MVATYPE == "NB" ]; then
  sh ./run.sh 2>&1 | cat > $WORKPATH/$LOGFILE &
 else
  if [ $ONLXPLUS == "1" ]; then
cat > $WORKPATH"/lsfrun.sh" << EOFF
#!/bin/bash
echo ==================================================================================================================
echo ${TRAINWWW}/${PARTICLE}/${TRACK}
echo ==================================================================================================================
sh $WORKPATH/run.sh 2>&1 | cat > $WORKPATH/$LOGFILE
rm -f $WORKPATH"/lsfrun.sh"
exit 0
EOFF
   if [ $TMVAMETHOD == "MLP" ]; then
     if [ $TMVAMLPMETHOD == "BFGS" ]; then
      export LSFQUEUE=1nw
     else
      export LSFQUEUE=1nw
     fi
   else
     export LSFQUEUE=2nd
   fi
   if [ $TRAINCONFIGNAME == "test" ]; then
    export LSFQUEUE=8nh
   fi
   bsub -q $LSFQUEUE -J "${TRAINWWW}/${PARTICLE}/${TRACK}" < $WORKPATH"/lsfrun.sh"
   #sh ./run.sh 2>&1 | cat > $WORKPATH/$LOGFILE &
  else
cat > $WORKPATH"/condorrun.sh" << EOFFF
#!/bin/bash
echo ==================================================================================================================
echo ${TRAINWWW}/${PARTICLE}/${TRACK}
echo ==================================================================================================================
sh $WORKPATH/run.sh 2>&1 | cat > $WORKPATH/$LOGFILE
rm -f $WORKPATH"/condorrun.sh"
exit 0
EOFFF
cat > $WORKPATH"/condorrun.job" << MOOF
# Condor environment
Universe                = vanilla
getenv                  = true
copy_to_spool           = true
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
Requirements = ( Memory > 1999 && Arch == "X86_64" && OSTYPE == "SLC5" && (POOL == "GENERAL" || POOL == "GEN_FARM") )
Rank         = kflops
output = \$ENV(HOME)/CondorLogs/out.\$(Process)
error  = \$ENV(HOME)/CondorLogs/err.\$(Process)
Log    = \$ENV(HOME)/CondorLogs/log.\$(Process)
MOOF
   echo "Executable = "$WORKPATH"/condorrun.sh" >> $WORKPATH"/condorrun.job"
   echo "Queue 1" >> $WORKPATH"/condorrun.job"
   condor_submit $WORKPATH"/condorrun.job"
  fi
 fi
}

# Kill any currently running training threads
if [ $MVATYPE == "NB" ]; then
 killall $TRAINER
 sleep 4
 killall $TRAINER
 sleep 4
 killall $ASCIITOROOT
 sleep 4
 killall $ANALYSIS
 sleep 4
fi

#exit 0

# Start new trainings

train "Electron" "Long"
train "Electron" "Upstream"
train "Electron" "Downstream"

train "Muon" "Long"
train "Muon" "Upstream"
train "Muon" "Downstream"

train "Pion" "Long"
train "Pion" "Upstream"
train "Pion" "Downstream"

train "Kaon" "Long"
train "Kaon" "Upstream"
train "Kaon" "Downstream"

train "Proton" "Long"
train "Proton" "Upstream"
train "Proton" "Downstream"

train "Ghost" "Long"
train "Ghost" "Upstream"
train "Ghost" "Downstream"

exit 0
