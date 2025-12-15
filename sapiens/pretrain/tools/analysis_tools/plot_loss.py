import json
import matplotlib.pyplot as plt
import argparse
import os

# --- PASSO 1: Configurar a leitura de argumentos da linha de comando ---
parser = argparse.ArgumentParser(description='Gera gráfico de Loss a partir de um log JSON.')

# Define o argumento obrigatório (o caminho do arquivo)
parser.add_argument('json_file', type=str, help='Caminho para o arquivo .json ou .jsonl de log')

# Define um argumento opcional para o nome da imagem de saída
parser.add_argument('--output', type=str, default=None, help='Nome do arquivo de saída da imagem (opcional)')

args = parser.parse_args()

filename = args.json_file

# --- PASSO 2: Definir nome de saída automaticamente se não for passado ---
if args.output:
    output_img = args.output
else:
    # Se a entrada for "meu_log.json", a saída será "meu_log_plot.png"
    base_name = os.path.splitext(filename)[0]
    output_img = f"{base_name}_plot.png"

# --- PASSO 3: Ler o arquivo ---
steps = []
losses = []

print(f">>> Lendo arquivo: {filename}")

try:
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Verifica se as chaves existem (algumas linhas de log podem ser apenas info de sistema)
                if 'step' in data and 'loss' in data:
                    steps.append(data['step'])
                    losses.append(data['loss'])
                elif 'iter' in data and 'loss' in data: # Fallback se não tiver 'step'
                    steps.append(data['iter'])
                    losses.append(data['loss'])
                    
            except json.JSONDecodeError:
                continue # Pula linhas que não são JSON válido

    print(f">>> Encontrados {len(steps)} pontos de dados.")

except FileNotFoundError:
    print(f"ERRO: O arquivo '{filename}' não foi encontrado.")
    exit()

if not steps:
    print("ERRO: Nenhuma linha com 'loss' e 'step' foi encontrada no arquivo.")
    exit()

# --- PASSO 4: Gerar o Gráfico ---
# Usa backend 'Agg' para funcionar em servidores sem monitor (headless)
plt.switch_backend('Agg') 

plt.figure(figsize=(10, 6))
plt.plot(steps, losses, marker='', linestyle='-', linewidth=2, color='#1f77b4', label='Training Loss')

plt.title(f'Loss Curve: {os.path.basename(filename)}', fontsize=14)
plt.xlabel('Steps / Iterations', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Salva o arquivo
plt.savefig(output_img, dpi=150)
print(f"[SUCESSO] Gráfico salvo como: {output_img}")