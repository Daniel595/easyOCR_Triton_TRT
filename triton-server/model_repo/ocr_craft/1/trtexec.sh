/usr/src/tensorrt/bin/trtexec \
  --onnx=craft.onnx \
  --saveEngine=craft.trt \
  --minShapes=input:1x3x512x256 \
  --optShapes=input:1x3x512x512 \
  --maxShapes=input:1x3x512x768 \
  --timingCacheFile=timing_cache.json \
  --fp16
