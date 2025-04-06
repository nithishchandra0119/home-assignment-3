Question 1 : Implementing a Basic Autoencoder

    Load MNIST Data: The dataset is loaded and normalized to have pixel values between 0 and 1.
    Define Autoencoder Model:
        Encoder: Compresses the image into a lower-dimensional representation (latent space).
        Decoder: Reconstructs the image from the latent space.
    Train the Autoencoder: The autoencoder is trained using binary cross-entropy loss with the original MNIST images as both input and target.
    Original vs. Reconstructed Images: The original and reconstructed images are displayed to visualize how well the model performs.
    Modify Latent Dimension: The latent space is modified to 16 dimensions, and the model is retrained. The effect of a smaller latent space on image reconstruction is analyzed.

Question 2 : Implementing a Denoising Autoencoder

    Load MNIST Dataset: The MNIST dataset is loaded and normalized for training.
    Add Gaussian Noise: Noise is added to the images to simulate real-world noise, and the model learns to denoise them.
    Define the Denoising Autoencoder: The autoencoder is a neural network model with an encoder and a decoder. It is trained to remove noise and reconstruct the original image.
    Train the Model: The autoencoder is trained with noisy inputs but clean outputs. The model learns to predict the original (clean) image from the noisy one.
    Visualize Results: The noisy images are compared to the denoised, reconstructed images to evaluate the model's performance.
    Compare with Basic Autoencoder: A basic autoencoder (without noise) is trained to compare its performance with the denoising autoencoder.
    Real-World Use Case: Denoising autoencoders are useful in medical imaging to clean noisy scans (like MRIs or CT scans), helping doctors get clearer images for diagnosis.

Question 3 : Implementing an RNN for Text Generation

    Load and Preprocess Data: We load the Shakespeare dataset and preprocess the text by converting each character to an integer and normalizing the data.
    Prepare Dataset for Training: The text is split into sequences of 50 characters, which are used as inputs (X) and the next character as the label (y).
    Define and Train LSTM Model: We build an LSTM-based model to predict the next character in a sequence. The model is trained using the preprocessed text for 10 epochs.
    Generate Text: Using the trained model, we generate text starting from a given string (e.g., "ROMEO: "), with the ability to adjust the randomness of the generated text using temperature scaling.
    Temperature Scaling Explanation: The temperature controls how creative the model is in generating text. Higher temperature means more randomness, and lower temperature makes the model's predictions more predictable.

Question 4 : Sentiment Classification Using RNN

    Load and Preprocess Data: The IMDB dataset is loaded, and text data is preprocessed by padding the sequences to ensure all reviews are of uniform length.
    Define and Train LSTM Model: An LSTM-based model is built for sentiment classification. The model is trained to predict whether a review is positive or negative based on its text.
    Evaluate Model Performance: After training, the model's performance is evaluated on the test set. We generate a confusion matrix and classification report to assess the model's accuracy, precision, recall, and F1-score.
    Precision-Recall Tradeoff: Precision and recall are important metrics in sentiment classification. The tradeoff between them helps decide which metric to prioritize based on the specific needs of the application.
