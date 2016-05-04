#!/usr/bin/env sh
/space/export/data/zwie_th/DLR/NeuralNetworks/caffe/build/tools/caffe train -solver /space/export/data/zwie_th/DLR/NeuralNetworks/dilation/solver_own.prototxt -gpu 0,1,2,3 -weights /space/export/data/zwie_th/DLR/NeuralNetworks/dilation/pretrained/dilated_convolution_context_coco. >> log.txt
