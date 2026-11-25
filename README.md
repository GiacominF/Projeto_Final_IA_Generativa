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
1. Abra o notebook `DCC831_Nottingham_Projeto_Final.ipynb` no Google Colab.
2. Em **Tempo de execução > Alterar tipo de tempo de execução**, selecione **GPU T4**.
3. Clique em **Executar tudo (`Ctrl+F9`)**. O notebook executará em sequência:
   - Download do dataset Nottingham, filtragem das composições completas e divisão (`train`, `validation` e `test`) por composição;
   - Tokenização `MIDILike` (`pitch_range=(36, 89)`), *data augmentation* por transposição no treino e preparação das janelas deslizantes;
   - Treinamento, geração das curvas de aprendizado e amostragem dos modelos `MusicRNN` (*baseline*) e `MusicTransformer`;
   - Avaliação objetiva no conjunto de teste (perplexidade e cinco descritores musicais), geração dos gráficos comparativos e síntese dos arquivos `.wav` em `entregaveis.zip`.

### Execução Local
1. Com o ambiente virtual ativado e as dependências instaladas, abra `DCC831_Nottingham_Projeto_Final.ipynb` no Jupyter ou VS Code.
2. Ajuste a variável `BASE_DIR` caso queira salvar os artefatos fora de `/content/nottingham_project` e execute todas as células na ordem.
