# Geração Musical Simbólica — Dataset Nottingham (DCC831)

## Instalação

1. (Opcional, para síntese de áudio `.wav` em Linux/Ubuntu) Instale o `FluidSynth` e o soundfont General MIDI:
   ```bash
   sudo apt-get update && sudo apt-get install -y fluidsynth fluid-soundfont-gm
   ```

2. Crie um ambiente virtual Python e instale as dependências:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Reprodução dos Resultados

### Via Google Colab (Recomendado)
1. Abra o notebook `DCC831_Nottingham_Preparacao_Dataset.ipynb` no Google Colab.
2. Em **Tempo de execução > Alterar tipo de tempo de execução**, selecione **GPU T4**.
3. Clique em **Executar tudo (`Ctrl+F9`)**. O notebook executará em sequência:
   - Download, inspeção e divisão do dataset Nottingham (`train`, `validation` e `test`) por composição;
   - Tokenização `MIDILike` e preparação das janelas deslizantes;
   - Treinamento e geração de amostras dos modelos `MusicGRU` (*baseline*) e `MusicTransformer`;
   - Avaliação objetiva no conjunto de teste (perplexidade e descritores musicais), geração dos gráficos e síntese dos arquivos `.wav` em `entregaveis.zip`.

### Execução Local
1. Com o ambiente virtual ativado e as dependências instaladas, abra `DCC831_Nottingham_Preparacao_Dataset.ipynb` no Jupyter ou VS Code.
2. Ajuste a variável `BASE_DIR` na seção inicial caso queira salvar os artefatos fora de `/content/nottingham_project` e execute todas as células na ordem.
