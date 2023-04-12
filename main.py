import tensorflow_hub as hub
import tensorflow as tf

print(tf.__version__)

model = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
embeddings = model(["The rain in Spain.", "falls",
                      "mainly", "In the plain!"])

print(embeddings.shape)