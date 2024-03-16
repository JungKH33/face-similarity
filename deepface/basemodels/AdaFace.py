from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger
from typing import List

import onnx
import onnxruntime as ort
import cv2
import numpy as np 

logger = Logger(module="basemodels.AdaFace")


class AdaFaceClient(FacialRecognition):
    def __init__(self):
        self.model_name = "Adaface"
        self.input_shape = (112, 112)
        self.output_shape = 512
        
        model_path = "/home/aipark/projects/kh/deepface/adaface.onnx"
        
        # Load the ONNX model
        self.model = onnx.load(model_path)

        # Create an ONNX Runtime session for inference
        self.session = ort.InferenceSession(model_path)

    def find_embeddings(self, img: np.ndarray) -> List[float]:
        # Resize the input image to match the input shape of the model
        # resized_img = self.resize_image(img, self.input_shape)

        img = np.transpose(img, (0, 3, 1, 2))
        
        # img is in scale of [0, 1] but expected [0, 255]
        ##if img.max() <= 1:
        #   img = img * 255
        
        # Convert the resized image to BGR format
        #bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Convert the input image to the format expected by ONNX Runtime
        #input_data = np.expand_dims(bgr_img, axis=0).astype(np.float32)
        
        # Perform inference with the ONNX model using ONNX Runtime
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        embeddings = self.session.run([output_name], {input_name: img})[0]

        return embeddings[0].tolist() 