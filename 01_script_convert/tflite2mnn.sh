./MNNConvert -f TFLITE --modelFile $1 --MNNModel  $(dirname $1)/$(basename $1 .tflite).mnn --bizCode biz
