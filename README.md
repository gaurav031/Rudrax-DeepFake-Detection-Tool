# Rudrax Deepfake Detection Tool

![Screenshot 2024-10-04 195714](https://github.com/user-attachments/assets/391ca6b9-c8a2-4b15-8cb2-f8526f04da01)  <!-- Replace with the actual link to your image -->

## Introduction
Rudrax is a powerful deepfake detection tool designed to identify fake videos and images with high accuracy. Leveraging advanced machine learning techniques, Rudrax enables users to upload any type of video or image for analysis. This tool is essential for anyone looking to verify the authenticity of multimedia content in an era where deepfakes are becoming increasingly prevalent. Whether you're a researcher, content creator, or simply curious, Rudrax provides the tools you need to ensure the integrity of visual media.

## Key Features
- **High Accuracy**: Detects deepfakes with impressive accuracy rates.
- **User-Friendly Interface**: Simple upload process for videos and images.
- **Versatile Uploads**: Supports a wide range of video and image formats.

Explore the capabilities of Rudrax and take a stand against misinformation in digital media.

## Instructions

1. **Add Your Data**
   - Place your data files in the `data` folder.

2. **Set Up the Virtual Environment**
   - Create a virtual environment by running:
     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment:
     - For Windows:
       ```bash
       venv\Scripts\activate
       ```
     - For macOS/Linux:
       ```bash
       source venv/bin/activate
       ```

3. **Install Dependencies**
   - Download and install all required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

4. **Prepare the Model**
   - Add your data to the `data` folder.
   - Train the model by running:
     ```bash
     python .\train_model.py
     ```

5. **Run the Application**
   - Start the application by executing:
     ```bash
     python .\app.py
     ```

## Download Presentation
Below is the link to the data of real and fake  from which you can download additional resources:
[Download Presentation](https://docs.google.com/presentation/d/1PbCjKB_Flfl2dNwJH_Mtivr7KyWRrPAs/edit?usp=sharing&ouid=104145226227150138618&rtpof=true&sd=true)  <!-- Replace with the actual link to your PPT -->
