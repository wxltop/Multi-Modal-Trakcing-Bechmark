#!/bin/bash

RESULTDIR="/cluster/work/cvl/visobt4/tracking_results"
DATADIR="/cluster/work/cvl/visobt4/tracking_datasets"

if [ "${PARAM3}" == "lasot" ]; then
  DATASETNAME="LaSOTTesting"
elif [ "${PARAM3}" == "nfs" ]; then
  DATASETNAME="nfs"
elif [ "${PARAM3}" == "otb" ]; then
  DATASETNAME="otb_sequences"
elif [ "${PARAM3}" == "oxuva_dev" ]; then
  DATASETNAME="oxuva"
elif [ "${PARAM3}" == "oxuva_test" ]; then
  DATASETNAME="oxuva"
elif [ "${PARAM3}" == "uav" ]; then
  DATASETNAME="ECCV16_Dataset_UAV123"
elif [ "${PARAM3}" == "lasot_extension_subset" ]; then
  DATASETNAME="LaSOT_extension_subset"
else
  echo -e "ERROR: Dataset name: ${PARAM3} invalid."
  exit 1
fi

DATADIR="${DATADIR}/${DATASETNAME}"

SEED="$(($LSB_JOBINDEX-1))"

echo -e "JOBID: $LSB_JOBID"
cd ..
echo -e "Start copying dataset ${DATASETNAME} from ${DATADIR}"
get_dataset.sh -n 8 -d $DATADIR
echo -e "Finished copying data"

source $HOME/myenv-tracking/bin/activate

export TORCH_EXTENSIONS_DIR=$TMPDIR

# run experiments
python -u run_parameters.py $PARAM1 $PARAM2 $PARAM3 $SEED --threads 0

## build result dir path
#printf -v PAD_SEED "%03d" $SEED
#RESULTDIR="${RESULTDIR}/${PARAM1}"
#EXPDIR="${PARAM2}_${PAD_SEED}"
#
#echo -e "Random Seed ${PAD_SEED}"
#echo -e "change cwd to ${RESULTDIR}"
#
#cd RESULTDIR
#
## tar result path if all experiments are finished.
#tar -cf "${EXPDIR}.tar" $EXPDIR

echo -e "Completed successfully."