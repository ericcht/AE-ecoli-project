import numpy as np
import tarfile
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to read data from file
def read_data(file_path, header=0, index_col=None):
    try:
        df = pd.read_csv(file_path, sep='\t', header=header, index_col=index_col)
    except pd.errors.ParserError:
        df = pd.read_csv(file_path, sep='\t', header=header, index_col=index_col)
    return df

# Function to preprocess the data
def preprocess_data(data):
    features = data.iloc[:, 1:].astype(float)
    labels = data.iloc[:, 0]
    return features, labels

# Function to train the autoencoder model and plot losses
def train_autoencoder(X_train):
    input_dim = X_train.shape[1]
    encoding_dim = 32  # Adjust according to your needs

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation='relu')(input_layer)
    decoder = Dense(input_dim, activation='sigmoid')(encoder)

    autoencoder = Model(input_layer, decoder)
    autoencoder.compile(optimizer=Adam(lr=0.0000001, clipnorm=100), loss='binary_crossentropy')

    # Train autoencoder and get history
    history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

    # Plot training and validation losses
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.legend(['loss', 'val_loss'])
    plt.savefig('loss_plot_BCE.png')  # Save the plot as an image file
    plt.close()  # Close the plot to release memory

    return autoencoder

def main():
    # Extract data from tar file
    tar_path = 'data/matrix_0.0001.tar.gz'
    extract_path = 'data/'

    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

    # List extracted files
    extracted_files = [f for f in os.listdir(extract_path) if os.path.isfile(os.path.join(extract_path, f))]

    # Load data
    dfs = []
    for file in extracted_files:
        file_path = os.path.join(extract_path, file)
        df = read_data(file_path)
        dfs.append(df)

    # Concatenate all dataframes
    data = pd.concat(dfs, ignore_index=True)
    data = data.fillna(value=0.0)

    # Preprocess data
    features, labels = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train autoencoder and plot losses
    autoencoder = train_autoencoder(X_train)

if __name__ == "__main__":
    main()

