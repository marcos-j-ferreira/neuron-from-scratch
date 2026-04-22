import random
import gradio as gr

# ===== Model =====
class Neuron:
    def __init__(self):
        self.w = [0.0] * 2
        self.b = 0.0
        self.learning_rate = 0.01

    def forward(self, x):
        return sum(wi * xi for wi, xi in zip(self.w, x)) + self.b

    def backward(self, x, y):
        prediction = self.forward(x)
        error = prediction - y

        for i in range(len(self.w)):
            self.w[i] -= self.learning_rate * error * x[i]

        self.b -= self.learning_rate * error

        return error**2


# ===== Training =====
def train_model():
    n = Neuron()
    dataset = []

    for _ in range(200):
        x1 = random.random()
        x2 = random.random()
        y = x1 + x2
        dataset.append(([x1, x2], y))

    for _ in range(500):
        for x, y in dataset:
            n.backward(x, y)

    return n


model = train_model()


# ===== Utils =====
def normalize(x, max_value=10):
    return x / max_value


# ===== Interface function =====
def predict(x1, x2):
    x1_norm = normalize(x1)
    x2_norm = normalize(x2)

    prediction = model.forward([x1_norm, x2_norm])
    prediction_denorm = prediction * 10

    return f"Predicted: {prediction_denorm:.3f} | Expected: {x1 + x2}"


# ===== Gradio UI =====
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="First number"),
        gr.Number(label="Second number"),
    ],
    outputs="text",
    title="Simple Perceptron - Sum Predictor",
    description="Enter two numbers and see how the perceptron approximates their sum."
)

interface.launch()
