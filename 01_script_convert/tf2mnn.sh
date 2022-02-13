./MNNConvert -f TF --modelFile $1 --MNNModel  $(dirname $1)/$(basename $1 .pb).mnn --bizCode biz
