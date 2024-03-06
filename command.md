# experiment I
python train.py --net resnet18 --trial 1 --gpu 0
python train.py -target 0 --net resnet18 --split 50 50 --trial 1 --gpu 5
python train.py -target 1 --net resnet18 --split 50 50 --trial 1 --gpu 7

python distillation.py -target 0 --net resnet18 -netT_type resnet18 -netT_name target:0_acc:0.842800.pth target:1_acc:0.837600.pth -netT_classes 50 50 --split 50 50 --gpu 0 -T 2 -alpha 0.3 0.5

python train.py -target 0 --net resnet18 --trial 2 --gpu 0

python train.py -target 0 --net resnet18 --gpu 0,1,2,3