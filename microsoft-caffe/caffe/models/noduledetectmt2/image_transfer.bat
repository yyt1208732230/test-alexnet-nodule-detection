Build\x64\Release\convert_imageset.exe --shuffle --resize_height=64 --resize_width=64 data\nodulesdetect\ data\nodulesdetect\val.txt data\nodulesdetect\val_lmdb

Build\x64\Release\convert_imageset.exe --shuffle --resize_height=64 --resize_width=64 data\nodulesdetect\ data\nodulesdetect\train.txt data\nodulesdetect\train_lmdb

Build\x64\Release\compute_image_mean.exe data\nodulesdetect\train_lmdb data\nodulesdetect\mean.binaryproto

Build\x64\Release\caffe.exe train --gpu=all --solver=models\noduledetectmt2\solver.prototxt >log\alexnet-nodule-detection-m2-200t.log 2>&1

python plot_training_log.py 4 loss-iters.png alexnet_noduledetection_round1testnet.log
