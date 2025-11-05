import unittest
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from src.pipelines.pipeline import run_pipeline

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.input_dir = Path("test_images")
        self.output_dir = Path("test_output")
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # Create dummy images
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 256, (100, 100), dtype=np.uint8))
            img.save(self.input_dir / f"test_image_{i}.jpeg")

    def tearDown(self):
        shutil.rmtree(self.input_dir)
        shutil.rmtree(self.output_dir)

    def test_run_pipeline(self):
        run_pipeline(self.input_dir, self.output_dir, n=3)
        
        # Check that output directories were created
        self.assertTrue((self.output_dir / "Normal").exists())
        self.assertTrue((self.output_dir / "Pneumonia").exists())
        
        # Check that some images were created
        normal_images = list((self.output_dir / "Normal").glob("*.jpeg"))
        pneumonia_images = list((self.output_dir / "Pneumonia").glob("*.jpeg"))
        self.assertTrue(len(normal_images) + len(pneumonia_images) > 0)

if __name__ == "__main__":
    unittest.main()