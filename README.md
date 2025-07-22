# Gender Classifier using PyTorch

This project uses PyTorch to train a deep learning model that classifies images as either men or women. It includes a web interface for uploading images and getting predictions.

## Project Structure

```
gender_classifier/
├── data/                  # Directory for training and validation data
├── models/                # Directory for saved models
├── static/                # Static files for the web app
├── templates/             # HTML templates for the web app
├── app.py                 # Flask web application
├── download_data.py       # Script to download and prepare training data
├── train_model.py         # Script to train the PyTorch model
└── requirements.txt       # Python dependencies
```

## Setup and Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Download and prepare the training data:

```bash
python download_data.py
```

3. Train the model:

```bash
python train_model.py
```

4. Run the web application:

```bash
python app.py
```

5. Open your browser and navigate to `http://127.0.0.1:5000/` to use the application.

## Usage

1. Upload an image using the web interface.
2. Click the "Predict" button.
3. The application will display the prediction (man or woman) along with the confidence score.

## Model Details

- The model uses a pre-trained ResNet-18 architecture with transfer learning.
- The final layer is replaced to classify between two classes (male and female).
- Training uses cross-entropy loss and Adam optimizer.
- Data augmentation is applied during training to improve generalization.

## Notes

- The training data is downloaded from a subset of the CelebA dataset.
- The model's accuracy depends on the quality and diversity of the training data.
- For best results, use clear face images for prediction.

## License

This project is for educational purposes only. Please ensure you comply with all relevant data protection and privacy regulations when using facial recognition technology.
