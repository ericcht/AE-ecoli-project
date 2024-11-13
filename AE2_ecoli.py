import os
import tarfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

def read_data(file_path, header=0, index_col=None):
    try:
        # Attempt to read the data without limiting the number of rows
        df = pd.read_csv(file_path, sep='\t', header=header, index_col=index_col)
    except pd.errors.ParserError:
        # If there's an error, try skipping the line causing the error
        df = pd.read_csv(file_path, sep='\t', header=header, index_col=index_col)
    return df

# Function to preprocess the data
def preprocess_data(data):
    # Split data into features and labels
    features = data.iloc[:, 1:].astype(float)
    labels = data.iloc[:, 0]

    return features, labels

# Function to define and train the autoencoder model
def train_autoencoder(X_train):
    # Define autoencoder architecture
    input_dim = X_train.shape[1]
    encoding_dim = 32  # Adjust according to your needs

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation='relu')(input_layer)
    decoder = Dense(input_dim, activation='sigmoid')(encoder)

    autoencoder = Model(input_layer, decoder)

    # Compile autoencoder model with binary cross-entropy loss
    autoencoder.compile(optimizer=Adam(lr=0.0000001, clipnorm=100), loss=BinaryCrossentropy())

    # Train autoencoder
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

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
    print(data)
    print(data.isnull().any())

    # Preprocess data
    features, labels = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, _, _ = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Define and train the autoencoder model
    autoencoder = train_autoencoder(X_train)

    # Encode and decode data
    encoded_data_train = autoencoder.predict(X_train)
    encoded_data_test = autoencoder.predict(X_test)

    # Save encoded representations if needed
    np.savetxt('encoded_data_train.csv', encoded_data_train, delimiter=',')
    np.savetxt('encoded_data_test.csv', encoded_data_test, delimiter=',')

if __name__ == "__main__":
    main()

