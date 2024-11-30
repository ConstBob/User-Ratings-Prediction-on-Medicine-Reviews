#!/usr/bin/env python
# coding: utf-8

# # Predicting Medicine Review Ratings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import warnings
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# from fastFM import als
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import FeatureHasher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", category=FutureWarning)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('Medicine_Details.csv')
print("Dataset Info: ")
data.info()
data.head()

data = data.drop(columns=['Image URL'])

data.columns = data.columns.str.strip()
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

data = data.dropna()

duplicates = data.duplicated(subset=['Medicine Name', 'Manufacturer']).sum()
print(f"Duplicated row number: {duplicates}")
data = data.drop_duplicates(subset=['Medicine Name', 'Manufacturer'])

data.head()

data['combined_text'] = data[['Composition', 'Uses', 'Side_effects', 'Manufacturer']].fillna("").agg(' '.join, axis=1)

X_text = data['combined_text'].tolist()
y = data[['Excellent Review %', 'Average Review %', 'Poor Review %']].values / 100 

embedding_method = 'bert'
model_type = 'mlp'

if embedding_method == 'bert':
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1').to(device)


    def get_biobert_embeddings(texts, tokenizer, model):
        embeddings = []
        for text in tqdm(texts, desc="Generating Bert embeddings"):
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)
        embeddings = np.vstack(embeddings)
        return embeddings


    if os.path.exists('biobert_embeddings.npy'):
        embeddings = np.load('biobert_embeddings.npy')
    else:
        embeddings = get_biobert_embeddings(X_text, tokenizer, model)
        np.save('biobert_embeddings.npy', embeddings)
    print(f"BioBERT shape: {embeddings.shape}")

elif embedding_method == 'word2vec':
    processed_texts = [simple_preprocess(text) for text in X_text]

    word2vec_model = Word2Vec(sentences=processed_texts, vector_size=768, window=5, min_count=1, workers=4, epochs=10)
    word2vec_model.save("word2vec.model")

    # Function to get Word2Vec embeddings by averaging word vectors
    def get_word2vec_embeddings(texts, model):
        embeddings = []
        for text in tqdm(texts, desc="Generating Word2Vec embeddings"):
            words = simple_preprocess(text)
            word_vectors = [model.wv[word] for word in words if word in model.wv]
            if word_vectors:
                avg_vector = np.mean(word_vectors, axis=0)
            else:
                avg_vector = np.zeros(model.vector_size)
            embeddings.append(avg_vector)
        embeddings = np.vstack(embeddings)
        return embeddings

    embeddings = get_word2vec_embeddings(X_text, word2vec_model)
    print(f"Word2Vec shape: {embeddings.shape}")

elif embedding_method == 'bagofwords':
    vectorizer = CountVectorizer(max_features=1000)
    embeddings = vectorizer.fit_transform(X_text).toarray()
    print(f"Bag of Words shape: {embeddings.shape}")

elif embedding_method == 'tfidf':
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    embeddings = vectorizer.fit_transform(X_text).toarray()
    print(f"TF-IDF shape: {embeddings.shape}")

else:
    raise ValueError("Unsupported embedding method.")

# Optional dimensionality reduction
reduce_dim = False
if reduce_dim:
    print("Conducting PCA...")
    pca = PCA(n_components=512, random_state=42)
    embeddings = pca.fit_transform(embeddings)
    print(f"Shape after dimensionality reduction: {embeddings.shape}")

np.save(embedding_method+'_embeddings.npy', embeddings)

X = embeddings
y = y

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class EarlyStopping:
        def __init__(self, patience=20, verbose=False, delta=0):
            self.patience = patience
            self.verbose = verbose
            self.delta = delta
            self.best_loss = None
            self.counter = 0
            self.early_stop = False

        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
                return False
            elif val_loss > self.best_loss - self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    return True
            else:
                self.best_loss = val_loss
                self.counter = 0
                return False
if model_type == 'transformer':
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
            self.register_buffer('pe', pe)

        def forward(self, x):
            """
            x: (seq_length, batch_size, d_model)
            """
            seq_length = x.size(0)
            pe = self.pe[:seq_length, :]  # (seq_length, 1, d_model)
            x = x + pe
            return self.dropout(x)

    class TransformerRegression(nn.Module):
        def __init__(self, input_dim=768, seq_length=1, num_layers=4, nhead=8, dim_feedforward=2048, dropout=0.1,
                     output_size=3):
            super(TransformerRegression, self).__init__()

            self.seq_length = seq_length
            self.input_dim = input_dim

            self.pos_encoder = PositionalEncoding(d_model=input_dim, dropout=dropout, max_len=seq_length)

            # Transformer Encoder Layer
            encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                        dropout=dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

            self.fc1 = nn.Linear(input_dim, 1024)
            self.bn1 = nn.BatchNorm1d(1024)
            self.dropout1 = nn.Dropout(p=dropout)

            self.fc2 = nn.Linear(1024, 512)
            self.bn2 = nn.BatchNorm1d(512)
            self.dropout2 = nn.Dropout(p=dropout)

            self.fc3 = nn.Linear(512, output_size)

            self.relu = nn.LeakyReLU()

        def forward(self, x):
            """
            x: (batch_size, seq_length, input_dim)
            """
            x = x.permute(1, 0, 2)  # (seq_length, batch_size, input_dim)

            x = self.pos_encoder(x)  # (seq_length, batch_size, input_dim)

            # Transformer Encoder
            x = self.transformer_encoder(x)  # (seq_length, batch_size, input_dim)

            x = x.permute(1, 0, 2)  # (batch_size, seq_length, input_dim)

            x = x.mean(dim=1)  # (batch_size, input_dim)

            x = self.fc1(x)  # (batch_size, 512)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout1(x)

            x = self.fc2(x)  # (batch_size, 256)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.dropout2(x)

            x = self.fc3(x)  # (batch_size, output_size)

            return x

    input_dim = X_train_scaled.shape[1]
    seq_length = 1
    num_layers = 4
    nhead = 8
    dim_feedforward = 512
    dropout = 0.5
    output_size = 3

    model = TransformerRegression(input_dim=input_dim, seq_length=seq_length, num_layers=num_layers, nhead=nhead,
                                  dim_feedforward=dim_feedforward, dropout=dropout, output_size=output_size).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    early_stopping = EarlyStopping(patience=10, verbose=True)

    batch_size = 64

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)  # (batch_size, seq_length=1, input_dim)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)  # (batch_size, seq_length=1, input_dim)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 1000
    print("Training Transformer Model...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        scheduler.step()

        epoch_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            all_preds = []
            all_targets = []
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

            y_pred = np.vstack(all_preds)
            y_true = np.vstack(all_targets)

            mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
            avg_mae = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')
            mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
            avg_mse = mean_squared_error(y_true, y_pred, multioutput='uniform_average')

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, MAE: {mae}, MSE: {mse}, Avg MAE: {avg_mae:.4f}, Avg MSE: {avg_mse:.4f}")

        if early_stopping(avg_mae):
            print("Early stopping triggered")
            break

    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor.to(device))
        y_pred = y_pred_tensor.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()

    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    avg_mae = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    avg_mse = mean_squared_error(y_true, y_pred, multioutput='uniform_average')

    print("Final Evaluation on Test Set:")
    for i in range(y_true.shape[1]):
        print(f"  Output {i + 1}: MAE = {mae[i]:.4f}, MSE = {mse[i]:.4f}")
    print(f"  Average MAE: {avg_mae:.4f}, Average MSE: {avg_mse:.4f}")


elif model_type == 'mlp':
    class MLPRegressor(nn.Module):
        def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.5, output_size=3):
            super(MLPRegressor, self).__init__()
            layers = []
            last_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(last_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                last_dim = hidden_dim
            layers.append(nn.Linear(last_dim, output_size))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    hidden_dims = [1024, 1024]
    dropout = 0.5
    output_size = 3

    model = MLPRegressor(input_dim=X_train_scaled.shape[1], hidden_dims=hidden_dims, dropout=dropout, output_size=output_size).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    early_stopping = EarlyStopping(patience=20, verbose=True)

    batch_size = 4096

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 1000
    print("Training MLP Model...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        scheduler.step()

        epoch_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            all_preds = []
            all_targets = []
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

            y_pred = np.vstack(all_preds)
            y_true = np.vstack(all_targets)

            mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
            avg_mae = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')
            mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
            avg_mse = mean_squared_error(y_true, y_pred, multioutput='uniform_average')

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, MAE: {mae}, MSE: {mse}, Avg MAE: {avg_mae:.4f}, Avg MSE: {avg_mse:.4f}")

        if early_stopping(avg_mae):
            print("Early stopping triggered")
            break

    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor.to(device))
        y_pred = y_pred_tensor.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()

    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    avg_mae = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    avg_mse = mean_squared_error(y_true, y_pred, multioutput='uniform_average')

    print("Final Evaluation on Test Set:")
    for i in range(y_true.shape[1]):
        print(f"  Output {i + 1}: MAE = {mae[i]:.4f}, MSE = {mse[i]:.4f}")
    print(f"  Average MAE: {avg_mae:.4f}, Average MSE: {avg_mse:.4f}")

elif model_type in ['randomforest', 'ridge']:
    print(f"Using Scikit-learn {model_type} model...")

    if model_type == 'randomforest':
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'ridge':
        regressor = Ridge(random_state=42)
    model = MultiOutputRegressor(regressor)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    avg_mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
    avg_mae = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')

    print("Final Evaluation on Test Set:")
    for i in range(y_test.shape[1]):
        print(f"  Output {i + 1}: MSE = {mse[i]:.4f}, MAE = {mae[i]:.4f}")
    print(f"  Average MSE: {avg_mse:.4f}, Average MAE: {avg_mae:.4f}")

elif model_type == 'onehot_linear':
    categorical_features = ['Composition', 'Uses', 'Side_effects', 'Manufacturer']
    numerical_features = []

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(data[categorical_features], y,
                                                        test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    avg_mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
    avg_mae = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')

    print("Final Evaluation on Test Set:")
    for i in range(y_test.shape[1]):
        print(f"  Output {i + 1}: MSE = {mse[i]:.4f}, MAE = {mae[i]:.4f}")
    print(f"  Average MSE: {avg_mse:.4f}, Average MAE: {avg_mae:.4f}")

elif model_type == 'factorization_machine':

    from fastFM import als
    from sklearn.preprocessing import OneHotEncoder

    n_features = 1000
    hasher = FeatureHasher(n_features=n_features, input_type='string')

    X_train_str = X_train_scaled.astype(str)
    X_test_str = X_test_scaled.astype(str)

    X_train_hashed = hasher.transform(X_train_str)
    X_test_hashed = hasher.transform(X_test_str)

    fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=10, l2_reg_w=0.1, l2_reg_V=0.5, random_state=42)

    fm_regressor = MultiOutputRegressor(fm)
    fm_regressor.fit(X_train_hashed, y_train)

    y_pred = fm_regressor.predict(X_test_hashed)

    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    avg_mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
    avg_r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

    print("Final Evaluation on Test Set:")
    for i in range(y_test.shape[1]):
        print(f"  Output {i + 1}: MSE = {mse[i]:.4f}, R2 = {r2[i]:.4f}")
    print(f"  Average MSE: {avg_mse:.4f}, Average R2: {avg_r2:.4f}")

elif model_type == 'svd_latent_factor':
    categorical_features = ['Composition', 'Uses', 'Side_effects', 'Manufacturer']
    numerical_features = []

    onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = onehot_encoder.fit_transform(data[categorical_features])

    from sklearn.decomposition import TruncatedSVD

    n_components = 500
    svd = TruncatedSVD(n_components=n_components, n_iter=3, random_state=42)
    X_reduced = svd.fit_transform(X_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    avg_mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
    avg_mae = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')

    print("Final Evaluation on Test Set:")
    for i in range(y_test.shape[1]):
        print(f"  Output {i + 1}: MSE = {mse[i]:.4f}, MAE = {mae[i]:.4f}")
    print(f"  Average MSE: {avg_mse:.4f}, Average MAE: {avg_mae:.4f}")

else:
    raise ValueError("Unsupported model type.")

outputs = ['Excellent Review %', 'Average Review %', 'Poor Review %']
for i, output in enumerate(outputs):
    residuals = y_test[:, i] - y_pred[:, i]
    plt.figure(figsize=(10, 4))
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='k')
    plt.axvline(0, color='r', linestyle='--', label='Zero Error')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'{output}: Residual Distribution')
    plt.legend()
    plt.grid()
    plt.show()