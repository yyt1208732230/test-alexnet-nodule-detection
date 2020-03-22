Build\x64\Release\convert_imageset.exe --shuffle --resize_height=64 --resize_width=64 data\nodulesdetect\ data\nodulesdetect\val.txt data\nodulesdetect\val_lmdb

Build\x64\Release\convert_imageset.exe --shuffle --resize_height=64 --resize_width=64 data\nodulesdetect\ data\nodulesdetect\train.txt data\nodulesdetect\train_lmdb

Build\x64\Release\compute_image_mean.exe data\nodulesdetect\train_lmdb data\nodulesdetect\mean.binaryproto

Build\x64\Release\caffe.exe train --solver=models\bvlc_alexnet\solver.prototxt >log\alexnet_noduledetection_round1.log 2>&1

python plot_training_log.py 4 loss-iters.png alexnet_noduledetection_round1testnet.log

Build\x64\Release\classification.exe models\noduledetectmt2\deploy.prototxt models\noduledetectmt2\caffe_alexnet_train_iter_70000.caffemodel data\nodulesdetect\mean.binaryproto data\nodulesdetect\labels.txt data\nodulesdetect\nodule\LIDC-IDRI-0175_51_375-180_0_0.jpg