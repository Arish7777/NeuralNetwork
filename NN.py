import streamlit as st
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Helper functions
def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of the ReLU function."""
    return (x > 0).astype(float)

def softmax(x):
    """Softmax activation for output layer."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    """Cross-entropy loss for classification."""
    n = targets.shape[0]
    loss = -np.sum(targets * np.log(predictions + 1e-8)) / n
    return loss

def one_hot(x, k, dtype=np.float32):
    """Convert labels to one-hot encoding."""
    return np.array(x[:, None] == np.arange(k), dtype)

class DeepNeuralNetwork:
    """Deep Neural Network class implementing forward and backward propagation."""
    def __init__(self, sizes, activation='relu'):
        self.sizes = sizes
        self.activation = activation
        self.weights = [np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
                       for input_size, output_size in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros((1, output_size)) for output_size in sizes[1:]]

    def forward(self, x):
        self.a = [x]
        self.z = []
        
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(self.a[-1], weight) + bias
            self.z.append(z)
            if self.activation == 'relu':
                self.a.append(relu(z))
            elif self.activation == 'sigmoid':
                self.a.append(1 / (1 + np.exp(-z)))

        z = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        self.z.append(z)
        self.a.append(softmax(z))
        return self.a[-1]

    def backward(self, x, y, learning_rate):
        m = x.shape[0]
        delta = self.a[-1] - y

        for i in reversed(range(len(self.weights))):
            grad_w = np.dot(self.a[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * (
                    relu_derivative(self.z[i - 1]) if self.activation == 'relu' 
                    else self.a[i] * (1 - self.a[i]))

            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b

    def predict(self, x):
        predictions = self.forward(x)
        return np.argmax(predictions, axis=1)

def create_streamlit_app():
    st.set_page_config(page_title="MNIST Neural Network", layout="wide")

    st.markdown(
        """
        <style>
            .stApp {
                background-color: #f0f0f0;
            }
            .st-emotion-cache-13k62yr {
                position: absolute;
                background: rgb(14, 17, 23);
                color: rgb(247 137 131);
                inset: 0px;
                color-scheme: dark;
                overflow: hidden;
            }
            .st-emotion-cache-1jicfl2 {
                width: 100%;
                padding: 2rem 2rem 1rem;
                min-width: auto;
                max-width: initial;
            }
            element.style {
                position: relative;
                width: 600.344px;
                height: 400px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("MNIST Digit Recognition Neural Network")
    st.write("This application demonstrates a deep neural network trained on the MNIST dataset.")

    # Sidebar for parameters
    st.sidebar.header("Training Parameters")
    activation = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid"])
    batch_size = st.sidebar.slider("Batch Size", 32, 256, 128, 32)
    learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
    epochs = st.sidebar.slider("Number of Epochs", 1, 400, 10)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Training Process")
        plot_placeholder = st.empty()  # Placeholder for the training metrics graph
        metrics_placeholder = st.empty()  # Placeholder for text metrics

        if st.button("Train Model"):
            # Load and preprocess data
            with st.spinner("Loading MNIST dataset..."):
                mnist_data = fetch_openml("mnist_784", version=1)
                x = mnist_data["data"].to_numpy() / 255.0
                y = mnist_data["target"].to_numpy()
                num_labels = 10
                y_new = one_hot(y.astype('int32'), num_labels)

                train_size = 60000
                x_train, x_test = x[:train_size], x[train_size:]
                y_train, y_test = y_new[:train_size], y_new[train_size:]

                # Store test data in session state
                st.session_state['x_test'] = x_test
                st.session_state['y_test'] = y_test

            # Initialize model and training metrics
            dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10], activation=activation)
            progress_bar = st.progress(0)
            
            losses = []
            accuracies = []

            # Training loop
            for epoch in range(epochs):
                indices = np.random.permutation(x_train.shape[0])
                x_train = x_train[indices]
                y_train = y_train[indices]

                for i in range(0, x_train.shape[0], batch_size):
                    x_batch = x_train[i:i+batch_size]
                    y_batch = y_train[i:i+batch_size]
                    dnn.forward(x_batch)
                    dnn.backward(x_batch, y_batch, learning_rate)

                predictions = np.argmax(dnn.forward(x_test), axis=1)
                accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
                loss = cross_entropy_loss(dnn.a[-1], y_test)
                
                losses.append(loss)
                accuracies.append(accuracy)

                progress_bar.progress((epoch + 1) / epochs)

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=losses, name="Loss"))
                fig.add_trace(go.Scatter(y=accuracies, name="Accuracy", line=dict(dash='dot')))
                fig.update_layout(title="Training Metrics",
                                xaxis_title="Epoch",
                                yaxis_title="Value")
                plot_placeholder.plotly_chart(fig)

                metrics_placeholder.markdown(f"### Epoch {epoch + 1}/{epochs}\n**Loss:** {loss:.4f} | **Accuracy:** {accuracy * 100:.2f}%")

            st.success("Training completed!")
            st.session_state['model'] = dnn

    with col2:
        st.subheader("Test Predictions")
        if st.button("Make Predictions") and 'model' in st.session_state and 'x_test' in st.session_state:
            num_samples = 5
            test_indices = np.random.choice(len(st.session_state['x_test']), num_samples)
            test_images = st.session_state['x_test'][test_indices]
            predictions = st.session_state['model'].predict(test_images)
            
            for i in range(num_samples):
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(test_images[i].reshape(28, 28), cmap='gray')
                ax.set_title(f"Predicted: {predictions[i]}")
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
        elif 'model' not in st.session_state:
            st.warning("Please train the model first!")

if __name__ == "__main__":
    create_streamlit_app()
