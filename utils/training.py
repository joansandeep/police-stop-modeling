import os
import joblib
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import io
import sys
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import numpy as np

class DualLogger:
    def __init__(self, buffer):
        self.terminal = sys.__stdout__
        self.buffer = buffer
    def write(self, message):
        self.terminal.write(message)
        self.buffer.write(message)
    def flush(self):
        self.terminal.flush()
        self.buffer.flush()

def train_and_save_xgb(X_train, y_train):
    model_path = 'models/xgb_model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        pos_weight = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=pos_weight)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
    return model

def train_and_save_tabnet_with_logs(X_train, y_train, cat_idxs, cat_dims, max_epochs=30):
    from imblearn.over_sampling import RandomOverSampler

    model_path = 'models/tabnet_model.zip'
    log_path = 'models/tabnet_training_logs.txt'

    if os.path.exists(model_path) and os.path.exists(log_path):
        model = TabNetClassifier(cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1, device_name='cuda')
        model.load_model(model_path)
        with open(log_path, 'r') as f:
            log_text = f.read()
        return model, log_text

    # Oversample minority class for TabNet
    ros = RandomOverSampler(random_state=42)
    X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

    buffer = io.StringIO()
    sys.stdout = DualLogger(buffer)

    model = TabNetClassifier(cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1, device_name='cuda')
    model.fit(
        X_train_bal, y_train_bal,
        max_epochs=max_epochs,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    model.save_model(model_path)

    sys.stdout = sys.__stdout__
    log_text = buffer.getvalue()
    buffer.close()

    with open(log_path, 'w', encoding="utf-8") as f:
        f.write(log_text)

    return model, log_text

def train_tfmlp_with_logs(X_train, y_train, X_val, y_val, input_dim):
    model_path = 'models/tfmlp_model.keras'
    log_path = 'models/tfmlp_training_logs.txt'

    if os.path.exists(model_path) and os.path.exists(log_path):
        model = load_model(model_path)
        with open(log_path, 'r', encoding="utf-8") as f:
            log_text = f.read()
        return model, log_text

    buffer = io.StringIO()
    sys.stdout = DualLogger(buffer)

    class_weights_array = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights_array))

    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='tanh')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=128,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        verbose=1,
        callbacks=[es]
    )
    model.save(model_path)
    sys.stdout = sys.__stdout__

    log_text = buffer.getvalue()
    buffer.close()

    with open(log_path, 'w', encoding="utf-8") as f:
        f.write(log_text)

    return model, log_text
