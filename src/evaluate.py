import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_data

def evaluate_model():
    # Load test data
    (_, _), (x_test, y_test) = load_and_preprocess_data()
    
    # Load the saved model
    model = tf.keras.models.load_model('fashion_mnist_cnn.h5')
    
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.2f}")
    
    # Generate predictions
    y_pred = model.predict(x_test)
    y_pred_classes = y_pred.argmax(axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
