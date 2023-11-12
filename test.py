from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
model = load_model("model.keras")
print("Model loaded")
print(model.summary())

def predict(input_text):
    # input_encoding = tokenizer([input_text], truncation=True, padding=True, max_length=max_length, return_tensors='tf')
    input_embedding = embedding_model.encode([input_text])
    print("Input embedding: ", input_embedding)

    predictions = model.predict(input_embedding)
    print("Predictions: ", predictions)

    predicted_probabilities = predictions[0]
    print("Predicted probabilities: ", predicted_probabilities)

    class_labels = ['Hate', 'Misleading', 'Disinformation', 'Rumour', 'Sensationalism']

    confidence_levels = {}
    for label, prob in zip(class_labels, predicted_probabilities):
        confidence_levels[label] = f"{prob}"
        print(f"{label}: {prob}")
    
    return confidence_levels

if __name__ == "__main__":
    print(predict(input("Waiting for input: ")))
