# SwinUNETR Brain Tumor Segmentation on Amazon SageMaker

## Overview
This project replicates and extends the SwinUNETR model used for brain tumor segmentation as demonstrated in the [MONAI research contributions](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21). The goal of the project was to understand the model and procedure described in the original GitHub repository, replicate the approach, and adapt it to run on Amazon SageMaker for enhanced flexibility and scalability across different GPU architectures.

## Project Setup

### Initial Exploration
The project started by replicating the existing SwinUNETR model and procedure detailed in the [MONAI repository](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21). This involved understanding the underlying architecture of SwinUNETR, a model designed for efficient medical image segmentation with a focus on brain tumors.

### Amazon SageMaker Configuration
To adapt the model for Amazon SageMaker, the following steps were taken:

1. **Environment Setup:** Configured an Amazon SageMaker notebook instance equipped with the necessary libraries, including PyTorch, MONAI, and the SageMaker Python SDK.
2. **Data Preparation:** Utilized MONAI for image transformations, ensuring data was appropriately formatted and augmented for effective training.
3. **Model Training:** Set up training jobs in SageMaker, leveraging its managed spot training feature to optimize costs. Training was conducted using multiple GPU instances to evaluate performance across different hardware.
4. **Model Deployment:** Deployed the trained model using SageMaker endpoints, allowing for scalable inference across various GPU architectures. This setup enables easy integration with existing applications for real-time brain tumor segmentation.

## Key Learnings

### Amazon SageMaker
- **Training and Deployment:** Gained hands-on experience in configuring and managing training jobs on SageMaker, including setting up scalable environments for model training and deployment.
- **SageMaker Libraries:** Learned to effectively use SageMaker's Python SDK to automate the workflow of model training and deployment, reducing manual overhead and speeding up iterations.

### MONAI
- **Image Transformations:** Acquired skills in using the MONAI library for performing complex image transformations required for medical imaging tasks, which are crucial for preparing the data for model training.

### SwinUNETR Model
- **Model Architecture:** Deepened understanding of the SwinUNETR model, specifically its application in brain tumor segmentation using transformers and U-Net architectures. This included learning about the efficiency of its sliding-window approach and self-attention mechanisms for processing 3D medical images.

## Conclusion
This project not only replicated a state-of-the-art model for medical image segmentation but also adapted it for cloud-based environments using Amazon SageMaker, showcasing the model's adaptability and efficiency in a scalable cloud infrastructure. The experience has significantly enhanced my skills in machine learning workflows and cloud-based model deployment.