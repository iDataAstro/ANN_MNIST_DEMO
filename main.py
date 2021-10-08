import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import logging
import os

from utils.get_data import get_prepared_data
from utils.get_model import get_prepared_model

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "main.py.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format=logging_str, filemode="a")

def main(no_classes, input_shape, loss, optimizer, metrics):
    """[summary]

    Args:
        no_classes ([INT]): No of classes for classificaiton
        input_shape ([int, int]): Input shape for model's input layer
        loss ([str]): Loss function for model
        optimizer ([str]): Optimizer for model
        metrics ([str]): Metrics to watch while training
    """
    logging.info("Getting data..")
    (X_train, y_train), (X_valid, y_valid), _ = get_prepared_data()

    logging.info("Getting compiled ann model..")
    model_ann = get_prepared_model(no_classes, input_shape, loss, optimizer, metrics)

    logging.info("Model training start..")
    EPOCHS = 30
    history = model_ann.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_valid, y_valid))
    logging.info("Model training ends..")

    logging.info("Plot Loss/Accuracy curves..")
    pd.DataFrame(history.history).plot(figsize=(10,8))
    plt.grid(True)
    plt.savefig("plots/loss_accuracy.png")

    model_ann.save("models/model_ann.h5")
    logging.info("Model saved successfully..")

if __name__ == '__main__':
    # Define variable to create model
    no_classes = 10
    IMG_SIZE = (28,28)
    # Define loss, optimizer and metrics
    LOSS_FUNC = "sparse_categorical_crossentropy"
    OPT = "SGD"
    METRICS = ["accuracy"]
    try:
        logging.info("\n>>>>>>>>>>> START OF MAIN <<<<<<<<<<")
        main(no_classes=no_classes, input_shape=IMG_SIZE, loss=LOSS_FUNC, optimizer=OPT, metrics=METRICS)
        logging.info("\n>>>>>>>>>>> END OF MAIN <<<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise(e)
