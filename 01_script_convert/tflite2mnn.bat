cd %~dp0

MNNConvert.exe -f TFLITE --modelFile %1 --MNNModel  %~p1/%~n1.mnn --bizCode biz
