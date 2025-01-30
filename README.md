# gRPC Cat Classifier API

## Overview
This project implements a **gRPC-based API** for classifying cat images using the **ResNet34 model from TorchVision**. The API receives an image as input and returns the predicted class index.

## Features
- gRPC-based communication for efficient model inference
- Uses a pre-trained ResNet34 model for classification
- Supports batch processing of images
- Implements a defined **gRPC service** for remote inference

## Technologies Used
- Python
- gRPC
- TorchVision (ResNet34 model)
- Pillow (PIL) for image processing
- Protobuf for defining the service

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed and install dependencies using:

```sh
pip install grpcio grpcio-tools torch torchvision pillow
```

### Clone the Repository
```sh
git clone https://github.com/blue-romeo/grpc-cat-classifier.git
cd grpc
```

## Running the gRPC Server
1. Compile the gRPC protobuf definition:
```sh
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto
```

2. Start the server:
```sh
python inference.py
```

## Using the gRPC Client
After starting the server, you can send an image for classification using a gRPC client:
```sh
python client.py --image path/to/cat/image.jpg
```

## API Specification
### gRPC Service Definition (in `inference.proto`)
```proto
service InferenceServer {
  rpc inference (InferenceRequest) returns (InferenceReply);
}

message InferenceRequest {
  bytes image_data = 1;
}

message InferenceReply {
  int32 class_index = 1;
}
```

## Example Response
```json
{
  "class_index": 281
}
```

## Future Enhancements
- Implement a RESTful API wrapper
- Optimize inference using ONNX or TensorRT

## License
This project is licensed under the MIT License.



