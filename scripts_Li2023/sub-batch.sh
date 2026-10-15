#!/bin/bash 

#PBS -N test_gnu
#PBS -A p93300014 
#PBS -l walltime=01:50:00
#PBS -q regular
#PBS -j oe
#PBS -m a 
#PBS -M huili7@ucar.edu
#PBS -l select=1:ncpus=36:mem=109GB


starttime=$(date -u +"%s")

DOPOSTPROCESS=true

module load parallel
module load ncl 


NUMCORES=5
TIMESTAMP=`date +%s%N`
COMMANDFILE=commands.${TIMESTAMP}.txt

THEFILE="~/tc_tracking/tempest/2trajectories.txt.TEST.NE120.0080_0089.005"
#THEFILE="./tempest/trajectories.txt.TEST.NE120.0131_0160.005ax"
STYR=0080
ENYR=0089

for DATA_YEAR in $(eval echo {$STYR..$ENYR})
do
  UFILE="/glade/scratch//strandwg/fromALCF/b.e13.B1850C5CN.ne120_g16.tuning.005ax.cam.h4.UBOT."$DATA_YEAR"010100Z-"$DATA_YEAR"123121Z.nc"
  VFILE="/glade/scratch//strandwg/fromALCF/b.e13.B1850C5CN.ne120_g16.tuning.005ax.cam.h4.VBOT."$DATA_YEAR"010100Z-"$DATA_YEAR"123121Z.nc"
  U10FILE="/glade/scratch/strandwg/fromALCF/b.e13.B1850C5CN.ne120_g16.tuning.005ax.cam.h4.U10."$DATA_YEAR"010100Z-"$DATA_YEAR"123121Z.nc"

  NCLCOMMAND="ncl 'thefile=\"${THEFILE}\"' 'ubot_file=\"${UFILE}\"' 'vbot_file=\"${VFILE}\"' 'u10_file=\"${U10FILE}\"' extract_tcbot_batch.ncl"
  echo ${NCLCOMMAND} >> ${COMMANDFILE}
done

  source `which env_parallel.bash`
  env_parallel --jobs ${NUMCORES} -u --sshloginfile ${PBS_NODEFILE} --workdir ${PWD} < ${COMMANDFILE}

 
# if [ "$DOPOSTPROCESS" = true ] ; then
#   ncl et_concat_trajs.ncl 'nlfile="'${NAMELISTFILE}'"'
#   ncl et_avg_text.ncl 'nlfile="'${NAMELISTFILE}'"'
#   ncl et_yearly_clim.ncl 'nlfile="'${NAMELISTFILE}'"'
# fi

endtime=$(date -u +"%s")
tottime=$(($endtime-$starttime))

rm ${COMMANDFILE}

printf "${tottime}\n" >> timing.txt


