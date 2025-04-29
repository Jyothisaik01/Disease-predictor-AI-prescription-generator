# Disease-predictor-AI-prescription-generator

This project is a machine learning-based disease prediction system that allows users to input symptoms and receive a likely diagnosis. It uses a pre-trained model to analyze symptom patterns and predict potential diseases.

🧠 Features
Predict diseases from user-input symptoms.

Pre-trained machine learning model using symptom data.

Audio feedback support, including Telugu language output.

Visual support with image assets and audio prescriptions.

🧰 Technologies Used
Python

scikit-learn (used for model training and prediction)

joblib (for model serialization)

Flask or Streamlit (likely for UI; please confirm)

Pandas & NumPy (for data manipulation)

Audio and image processing libraries (for media output)

📁 Project Structure
home.py – Main application logic.

disease_predictor_model.joblib – Serialized ML model.

Training.csv and Testing.csv – Datasets used for training and evaluation.

symptom_features.txt – List of symptoms/features used by the model.

prescription_audio.mp3 and telugu_audio.mp3 – Audio files for output.

.env – Environment configuration (likely API keys or settings).

requirements.txt – Python dependencies.

▶️ How to Run

Create venv files 

python -m venv venv

activate the venv

Scripts\activate\venv

Install dependencies

pip install -r requirements.txt

Run the application

Streamlit run home.py

Interact with the interface – Input symptoms, and receive predictions along with audio output.

📌 Note
Ensure your Python environment has access to the required libraries listed in requirements.txt, and you may need to install additional audio/GUI dependencies based on the final deployment interface.
