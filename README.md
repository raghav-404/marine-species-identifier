# marine-species-identifier
🐠 A simple CNN + Streamlit app to classify marine species from underwater images
# 🐠 Marine Species Identifier

A simple Convolutional Neural Network (CNN) and Streamlit app to classify marine species (Dolphin, Octopus, Seals, Seahorse, Sea Turtles) from underwater images.

## 🚀 Features
- Train a CNN with PyTorch
- Upload images via a web app (Streamlit)
- Instant predictions with species name
- Demo-ready for hackathons

## 📂 Project Structure
- `marine_species_pipeline.py` → Training pipeline
- `app.py` → Streamlit frontend
- `best_model.pth` → Trained model
- `requirements.txt` → Dependencies

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
