# Thanks to the contribution of Mateus Machado!!!
# https://github.com/mtarcinalli

import os
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings

tqdm.pandas()

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "2_sentences")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "3_sentences_selected")

if not os.path.exists(INPUT_FOLDER):
    print("Input folder does not exist.")
    exit(1)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    
WORD_TO_SEARCH = "inflação"
THRESHOLD_GLOBAL = 0.6 # %

df = pd.DataFrame()
filtered_df = pd.DataFrame()

def _meeting_key(meeting_id):
    return int(meeting_id.split('_', 1)[0])

def calcular_distancia(sentenca, vetor_referencia, embeddings):
    vetor_sentenca = embeddings.embed_query(sentenca)
    similaridade = cosine_similarity([vetor_sentenca], [vetor_referencia])[0][0]
    distancia = 1 - similaridade
    return distancia

def vectorize():
    dados = []
    
    for nome_arquivo in sorted(
        [f for f in os.listdir(INPUT_FOLDER) if os.path.isfile(os.path.join(INPUT_FOLDER, f))],
        key=lambda f: _meeting_key(os.path.splitext(f)[0])
    ):
        caminho = os.path.join(INPUT_FOLDER, nome_arquivo)
        data = os.path.splitext(nome_arquivo)[0]
        with open(caminho, "r", encoding="utf-8") as f:
            for linha in f:
                sentenca = linha.strip()
                if sentenca:
                    dados.append({"data": data, "sentenca": sentenca})
                        
    global df
    df = pd.DataFrame(dados)
    
    print('Quantidade inicial de sentenças:', len(df))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': device}
    )
    
    vetor_inflacao = embeddings.embed_query(WORD_TO_SEARCH)
    df["inflation"] = df["sentenca"].progress_apply(lambda x: calcular_distancia(x, vetor_inflacao, embeddings))
    
def select():
    global filtered_df
    filtered_df = df[(df['inflation'] < THRESHOLD_GLOBAL)]
      
    print('Quantidade final de sentenças:', len(filtered_df),'\n')
      
def save():
    grouped_df = filtered_df.groupby(filtered_df['data'])
    
    for date, group in grouped_df:
        output_file_path = os.path.join(OUTPUT_FOLDER, f"{date}.txt")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for sentence in group['sentenca']:
                output_file.write(sentence + "\n")

def main():
    vectorize()
    select()
    save()

if __name__ == "__main__":
    main()