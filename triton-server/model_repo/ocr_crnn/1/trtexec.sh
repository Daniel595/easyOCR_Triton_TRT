/usr/src/tensorrt/bin/trtexec \
  --onnx=crnn.onnx \
  --saveEngine=crnn.trt \
  --minShapes=input1:1x1x64x32 \
  --optShapes=input1:1x1x64x128 \
  --maxShapes=input1:1x1x64x512 \
  --timingCacheFile=timing_cache.json \
  --fp16
  