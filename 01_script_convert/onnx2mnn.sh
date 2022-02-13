./MNNConvert -f ONNX --modelFile $1 --MNNModel  $(dirname $1)/$(basename $1 .onnx).mnn --bizCode biz
