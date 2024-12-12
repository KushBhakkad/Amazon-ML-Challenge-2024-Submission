# Amazon-ML-Challenge-2024-Submission

This repository contains my submission for the **Amazon ML Challenge 2024** hackathon. The goal of the challenge was to build a machine learning solution to accurately extract product attributes (e.g., weight, voltage, dimensions) from images.

## 🚀 Solution Overview

The solution leverages Optical Character Recognition (OCR) using EasyOCR, natural language processing, and custom logic for entity extraction. Key features include:
- Segmented image processing to enhance OCR accuracy.
- Standardized unit mapping for consistent attribute values.
- A robust sanity check mechanism for output validation.

## 🖼️ Flow of the Solution

1. **Image Download**: Images are fetched from URLs provided in the dataset.
2. **Image Segmentation**: Each image is divided into three horizontal segments to optimize text extraction.
3. **OCR Processing**: Text is extracted from each segment using EasyOCR.
4. **Entity Parsing**: Extracted text is processed with regex patterns to identify and standardize attributes.
5. **Value Selection**: The most relevant attribute value is selected based on predefined logic.
6. **Output Prediction**: Predictions are saved in a CSV file for evaluation.
7. **Sanity Check**: Outputs are validated to ensure correctness and completeness.

![Flowchart_enhanced](https://github.com/user-attachments/assets/7b8e14ad-a293-44c7-b958-1451661d7874)

## 📊 Evaluation Metrics

The project calculates:
- **Precision**
- **Recall**
- **F1 Score**

For a sample test dataset, the solution achieved:
- **F1 Score**: 0.27

![Output](https://github.com/user-attachments/assets/e2b3312c-8d9a-4a0b-9191-30e281aa1909)

## 🛠️ Tech Stack

- **Python**: Core programming language.
- **EasyOCR**: For text extraction.
- **Pandas**: For data manipulation.
- **Scikit-learn**: For evaluation metrics.
- **Pillow**: For image processing.

## 📁 Project Structure

```plaintext
.
├── dataset/
│   ├── sample_test.csv                # Input dataset
│   ├── sample_test_out_predicted.csv  # Output predictions
├── src/
│   ├── utils.py                       # Helper functions
│   ├── constants.py                   # Constants for unit mapping
│   ├── sanity.py                      # Sanity check script
├── mycode.py                          # Main script
├── README.md                          # Project documentation

```

## 🌟 Highlights
- **Accuracy:** The solution demonstrated comapratively fair accuracy on the provided dataset.
- **Efficiency:** By segmenting images and using optimized OCR techniques, the processing time was minimized.
- **Scalability:** The modular structure allows easy addition of new entities and patterns.

## 🏆 Hackathon Experience

This project was developed as part of the Amazon ML Challenge 2024, where I applied my skills in:
- OCR and text extraction.
- Regex-based parsing and NLP.
- Building scalable and modular solutions for real-world problems.
- Learned a lesson of making sure that the technical capabilities of the local machines should be taken in consideration beforehand.
- As I spent a lot of time on improving the accuracy on the training and sample test dataset, and overestimating my device's capabilities, I was unable to generate the final output csv file, due to which the submission failed and I lost an opportunity. 
