DATADIR=../data/motif_spikein_ATAGGC_50runs
RUN_CMD=''
cd script/

for RUN in {2..3}
do
	for LR in '5e-02'
	do
		
		# optimAdam
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/optimAdam_lr$LR \
		    --optimizer optimAdam  --lr $LR

		# optimAdam with 1:1 training ratio
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/optimAdam_ratio1_lr$LR \
		    --optimizer optimAdam  --g_interval 1 --lr $LR
		
		# SOMD (3 different versions)
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv1_lr$LR \
			--optimizer OMDA  -v 1 --lr $LR
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv2_lr$LR \
			--optimizer OMDA  -v 2 --lr $LR
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv3_lr$LR \
			--optimizer OMDA  -v 3 --lr $LR

		# SOMD with 1:1 training ratio (3 different versions)
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv1_ratio1_lr$LR \
			--optimizer OMDA  --g_interval 1 -v 1 --lr $LR
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv2_ratio1_lr$LR \
			--optimizer OMDA  --g_interval 1 -v 2 --lr $LR
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv3_ratio1_lr$LR \
			--optimizer OMDA  --g_interval 1 -v 3 --lr $LR

	done
done