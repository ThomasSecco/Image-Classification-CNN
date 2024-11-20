import tensorflow as tf
from data_loader import load_and_preprocess_data
from model import create_cnn_model

def train_model():
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create model
    model = create_cnn_model()
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
    
    # Save the trained model
    model.save('fashion_mnist_cnn.h5')
    
    return model, history, (x_test, y_test)
