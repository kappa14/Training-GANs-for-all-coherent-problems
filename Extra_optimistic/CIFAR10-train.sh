cd script/

## Adam
python cifar10.py -o cifar10/adam --optimizer SGD -v 1 --schedule adam

## Adam with ratio=1
python cifar10.py -o cifar10/adam_ratio1 --optimizer SGD -v 1 --schedule adam --training_ratio  1

## optimAdam
python cifar10.py -o cifar10/optimAdam --optimizer optimAdam

## optimAdam with ratio=1
python cifar10.py -o cifar10/optimAdam_ratio1 --optimizer optimAdam  --training_ratio 1