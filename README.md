# Medical Image Processor and Classifier

This Python project offers a suite of tools for medical image loading, processing, classification and saving.

## Project Structure 

```
├── .gitignore
├── README.md
├── images
│   ├── ...
├── main.py
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── classifier.py
│   ├── image_loader.py
│   ├── image_processor.py
│   └── pipelines
│       └── pipeline.py
└── tests
    └── test_pipeline.py
```

## How to Run

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**

   ```bash
   python main.py --input images --output output
   ```

   Classifies chest x-ray images and saves them into `Normal` and `Pneumonia` subdirectories

## How to Run the Tests

To run the tests use the following command:

```bash
python -m unittest discover tests
```
