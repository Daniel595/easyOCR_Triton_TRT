# Tritonserver Easyocr implementation using TensorRT
Were using torchfree_ocr (easyocr adaption to use onnx models). The trt prediction is achieved by simply monkey patching the easy_ocr module. By this we just can use the pipeline knwon from easyocr `Reader.readtext()` but backed will use the trt models deployed as triton trt model.

To get it running some dirty hacks had to be applied in order not to crash during initialization (Download models and so on). It might be a source of further problems, especially when adapting N Instances of the python model.

## TODO: 
- Dockerfile
- convenient auto-setup
- dynamic shapes for trt engines
    - configure pipeline to ensure model inputs are inside the dynamic range
- Example code, Client usage

