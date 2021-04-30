DATADIR=../data/motif_spikein_ATAGGC_50runs
RUN_CMD=''
cd script/

for RUN in {0..24}
do
	for LR in '5e-02'
	do
		
		# SGD
        KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SGD_lr$LR \
        	--optimizer SGD --lr $LR
        
		# SGD with Adagrad
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/adagrad_lr$LR \
			--optimizer SGD  -s adagrad --lr $LR

		# SGD with Nesterov momentum
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/nesterov_lr$LR \
			--optimizer SGD  --momentum 0.9 --nesterov  --lr $LR

		# SGD with Adam
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/adam_lr$LR \
		    --optimizer SGD  -s adam --lr $LR

		# SOMD with 1:1 training ratio (3 different versions)
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv1_ratio1_lr$LR \
			--optimizer OMDA  --g_interval 1 -v 1 --lr $LR
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv2_ratio1_lr$LR \
			--optimizer OMDA  --g_interval 1 -v 2 --lr $LR
		KERAS_BACKEND=theano python wgan_train.py -d $DATADIR/run$RUN -o DNA/run$RUN/SOMDv3_ratio1_lr$LR \
			--optimizer OMDA  --g_interval 1 -v 3 --lr $LR
			
	done
done