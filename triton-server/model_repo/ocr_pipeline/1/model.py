import numpy as np
import cv2
import triton_python_backend_utils as pb_utils

def predict_craft(craft_input):
    inputs = [pb_utils.Tensor("input", craft_input)]
    outputs = ["output", "onnx::Conv_281"]
    craft_output = pb_utils.InferenceRequest(model_name="ocr_craft",
                                             requested_output_names=outputs,
                                             inputs=inputs,
                                             preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU))
    craft_response = craft_output.exec()
    if craft_response.has_error():
        raise pb_utils.TritonModelException(
            craft_response.error().message()
        )
    output = pb_utils.get_output_tensor_by_name(craft_response, outputs[0]).as_numpy()
    features = pb_utils.get_output_tensor_by_name(craft_response, outputs[1]).as_numpy()
    return output, features

def predict_crnn(crnn_input):
    inputs = [pb_utils.Tensor("input1", crnn_input)]
    outputs = ["output"]
    crnn_output = pb_utils.InferenceRequest(model_name="ocr_crnn",
                                             requested_output_names=outputs,
                                             inputs=inputs,
                                             preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU))
    crnn_response = crnn_output.exec()
    if crnn_response.has_error():
        raise pb_utils.TritonModelException(
            crnn_response.error().message()
        )
    output = pb_utils.get_output_tensor_by_name(crnn_response, outputs[0]).as_numpy()
    return output

class CraftDummySession:
    def __init__(self, path):
        pass
    
    def get_inputs(self):
        class Input:
            name = 'input'
        return [Input()]
    
    def run(self, _, inputs):
        x = inputs[list(inputs.keys())[0]]
        # predict_craft erwartet einzelne Bilder oder Batch
        return [predict_craft(x)[0]]

class CRNNDummySession:
    def __init__(self, path):
        pass

    def get_inputs(self):
        class Input:
            name = 'input1'
        return [Input()]
    
    def run(self, _, inputs):
        x = inputs[list(inputs.keys())[0]]
        # predict_craft erwartet einzelne Bilder oder Batch
        return [predict_crnn(x)]
    

class TritonPythonModel:
    def initialize(self, args):
        split = args['model_instance_name'].split('_')
        dev_cnt = int(split[-2])
        inst_cnt = int(split[-1])
        import time
        # time.sleep(20)
        time.sleep(.5 * inst_cnt)
        import shutil
        # # /usr/local/lib/python3.12/dist-packages/torchfree_ocr/utils.py
        def dummy(*args, **kwargs):
            shutil.copytree('/models/ocr_pipeline/1/ocr_models', '/root/.TorchfreeOCR', dirs_exist_ok=True)
            return True
        from torchfree_ocr import detection, recognition
        detection.InferenceSession = CraftDummySession
        recognition.InferenceSession = CRNNDummySession
        from torchfree_ocr import Reader
        Reader.__init__.__globals__['download_and_unzip_all'] = dummy
        # shutil.copyfile('/models/ocr_pipeline/1/torchfree_ocr/character/en_char.txt','/root/.TorchfreeOCR/character/en_char.txt')
        self.reader = Reader(["en"])


    def execute(self, requests):
        responses = []
        for request in requests:
            input_image = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()
            image = input_image[0]  # Batch=1
            result = self.reader.readtext(image)

            codes, angles, confidences = [], [], []
            for (bbox, text, prob) in result:
                codes.append(text)
                confidences.append(prob)
                x1, y1 = bbox[0]
                x2, y2 = bbox[1]
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)
                angles.append(angle_deg)

            codes_np = np.array(codes, dtype=np.object_)
            confs_np = np.array(confidences, dtype=np.float32)
            out_tensors = [
                pb_utils.Tensor("CODES", codes_np),
                pb_utils.Tensor("CONFIDENCES", confs_np)
            ]
            responses.append(pb_utils.InferenceResponse(output_tensors=out_tensors))

        return responses