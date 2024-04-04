import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel, BertTokenizer
import torch
from tqdm import trange

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def process_directory(directory_path, tokenizer, model):
    data = []
    
    # Loop through each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                label = lines[0].strip() if lines else None
                text_lines = [line.strip() for line in lines[1:] if line.strip()] 
                
                encoded_input = tokenizer(text_lines, padding=True, return_tensors='pt')
                with torch.no_grad():
                    model_output = model(**encoded_input)
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

                for ind, sent in enumerate(text_lines):
                    data.append({'label': label, 'sentence': sent, 'embedding': sentence_embeddings[ind,:].tolist()})
    
    # Create pandas dataframe
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python clustering.py path_to_directory")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        sys.exit(1)
    
    style_tokenizer = AutoTokenizer.from_pretrained('AnnaWegmann/Style-Embedding')
    style_model = AutoModel.from_pretrained('AnnaWegmann/Style-Embedding')

    dataframe = process_directory(directory_path, style_tokenizer, style_model)

    reshaped_data = []

    for _, row in dataframe.iterrows():
        new_row = {'label': row['label'], 'sentence': row['sentence']}
        for ind, value in enumerate(row['embedding']):
            new_row[f'd_{ind}'] = value
        reshaped_data.append(new_row)

    df_reshaped = pd.DataFrame(reshaped_data)
    feature_cols = [f'd_{ind}' for ind in range(768)]

    reducer_pca = PCA(n_components=2)
    ppl = df_reshaped.groupby('label').size().mean() // 4 
    reducer_tsne = TSNE(n_components=2, perplexity=ppl, n_iter=30000, verbose=1, random_state=15324)
    pca_results = reducer_pca.fit_transform(df_reshaped[feature_cols].values)
    tsne_results = reducer_tsne.fit_transform(df_reshaped[feature_cols].values)

    df_reshaped['x_value_pca'] = pca_results[:,0]
    df_reshaped['y_value_pca'] = pca_results[:,1]
    df_reshaped['x_value_tsne'] = tsne_results[:,0]
    df_reshaped['y_value_tsne'] = tsne_results[:,1]

    alpha = 0.5 if len(df_reshaped) > 2000 else 1.0

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x='x_value_pca', y='y_value_pca',
        hue='label',
        palette=sns.color_palette("bright", df_reshaped['label'].nunique()),
        data=df_reshaped,
        alpha=alpha
    ).set(title=f'{directory_path}: PCA')
    plt.savefig(f'{directory_path}_pca.png')
    plt.clf()

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x='x_value_tsne', y='y_value_tsne',
        hue='label',
        palette=sns.color_palette("bright", df_reshaped['label'].nunique()),
        data=df_reshaped,
        alpha=alpha
    ).set(title=f'{directory_path}: TSNE')
    plt.savefig(f'{directory_path}_tsne.png')
    plt.clf()
