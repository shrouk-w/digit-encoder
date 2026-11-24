import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "7"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from sklearn.neighbors import KNeighborsClassifier


tf.keras.utils.set_random_seed(7)


def create_autoencoder(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), padding="same", activation="elu"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(8, (3, 3), padding="same", activation="elu"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(4, (3, 3), padding="same", activation="elu"),
        layers.UpSampling2D((2, 2)),
        layers.Conv2DTranspose(8, (3, 3), padding="same", activation="elu"),
        layers.UpSampling2D((2, 2)),
        layers.Conv2DTranspose(1, (3, 3), padding="same", activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def prepare_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255
    return x_train, y_train, x_test, y_test


def flatten_code(code):
    return code.reshape(code.shape[0], -1)


def best_k(codes):
    sample = codes[:3000]
    results = {}
    for k in range(2, 16):
        labels = KMeans(n_clusters=k, random_state=7, n_init=10).fit_predict(sample)
        results[k] = silhouette_score(sample, labels)
    return max(results, key=results.get), {int(a): float(b) for a, b in results.items()}


def cluster_names(cluster_labels, true_labels):
    names = {}
    for cluster in np.unique(cluster_labels):
        digits = true_labels[cluster_labels == cluster]
        names[int(cluster)] = int(np.bincount(digits, minlength=10).argmax())
    return names


def rename_clusters(cluster_labels, names):
    return np.array([names[int(label)] for label in cluster_labels])


x_train, y_train, x_test, y_test = prepare_data()

autoencoder = create_autoencoder((28, 28, 1))
autoencoder.fit(x_train, x_train, epochs=2, batch_size=128, validation_split=0.1, verbose=2)

encoder = models.clone_model(autoencoder)
encoder.set_weights(autoencoder.get_weights())
for _ in range(4):
    encoder.pop()

train_code = flatten_code(encoder.predict(x_train, batch_size=512, verbose=0))
test_code = flatten_code(encoder.predict(x_test, batch_size=512, verbose=0))

print("kod train:", train_code.shape)
print("kod test:", test_code.shape)
print(train_code[:10])

k, scores = best_k(train_code)
print("silhouette:", scores)
print("najlepsze k:", k)

kmeans = KMeans(n_clusters=10, random_state=7, n_init=10)
train_clusters = kmeans.fit_predict(train_code)

names = cluster_names(train_clusters, y_train)
train_pred = rename_clusters(train_clusters, names)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_code[:20000], train_clusters[:20000])
test_clusters = knn.predict(test_code)
test_pred = rename_clusters(test_clusters, names)

print("klaster -> cyfra:", names)
print("accuracy train:", accuracy_score(y_train, train_pred))
print("accuracy test KNN:", accuracy_score(y_test, test_pred))
print("confusion matrix test:")
print(confusion_matrix(y_test, test_pred))
