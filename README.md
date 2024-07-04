
# Palm-Mask-Generator

Palm-Mask-Generator  is a project that detects and annotates hands in images using a combination of YOLOv7 and hand landmark detection models.


## Installation

Follow these steps to set up the project

1. Clone the repository

```bash
git clone https://github.com/Arjun91221/Palm-Generator.git
cd Palm-Generator
```

2. Create a virtual environment

```bash
python3 -m venv venv
source venv\bin\activate
```

3. Install the required packages

```bash
pip install -r requirements.txt
```

4. Download YOLOv7 model

```bash
mkdir yolov7
wget -q https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt -O yolov7/yolov7.pt
```


## Running the Application

You can run the application using one of the following commands

1. Using Python

```bash
  python main.py
```

2. Using Uvicorn

```bash
  uvicorn main:app --reload --host 0.0.0.0 --port 8001
```


## Usage

After running the application, you can use the provided endpoints to process images and detect hands. Ensure you have the necessary input images and follow the API documentation for more details on how to use the endpoints effectively.


## Acknowledgements

Special thanks to the developers and contributors of YOLOv7 and other open-source libraries used in this project.

