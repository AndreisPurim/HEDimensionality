# How does reducing the dimension of feature vectors impact Biometric Systems that use Homomorphic Encryption?

Este repositório contém o código referente ao short paper "How does reducing the dimension of feature vectors impact Biometric Systems that use Homomorphic Encryption?", publicado na trilha principal do SBSeg 2025.

O artigo explora uma pergunta simples: se reduzirmos o tamanho dos vetores biométricos antes de criptografá-los, quanto o sistema acelera e quanta acurácia é perdida? Para testar isso, aplicamos uma Projeção Aleatória Gaussiana para comprimir vetores de 128 dimensões para tamanhos menores, e então medimos tanto a acurácia de comparação quanto o tempo de execução utilizando um sistema com criptografia homomórfica chamado [PEBA1](https://github.com/lab-incert/peba1/), que utiliza a biblioteca [TFHE](https://github.com/tfhe/tfhe/tree/master).

As principais hipóteses e conclusões do artigo, para o PEBA1, são:

* A perda de acurácia ao reduzir a dimensão é a mesma em sistemas criptografados e não criptografados, e
* O tempo de execução cai linearmente com a redução do vetor, pois a complexidade do algoritmo do PEBA escala com o comprimento do vetor (distância euclidiana), e não devido a fatores específicos da criptografia homomórfica.

O resumo (em inglês) é:

> Homomorphic Encryption (HE) enables biometric systems to perform matching directly on encrypted feature vectors, preserving user privacy throughout the process. However, the high computational cost of encrypted-domain operations, especially on high-dimensional inputs, remain a major barrier to real-world use. This study discusses presents a concrete example of how reducing the dimensionality of biometric feature vectors affects both matching accuracy and runtime, and discusses how system designers could estimate the trade-off between time savings and accuracy loss when choosing a target dimension.

# Estrutura do README.md

A estrutura desse arquivo é a seguinte:

* [Título e Contexto](#how-does-reducing-the-dimension-of-feature-vectors-impact-biometric-systems-that-use-homomorphic-encryption)
* [Estrutura do README.md](#estrutura-do-readmemd)
* [Selos Considerados](#selos-considerados)
* [Informações Básicas](#informações-básicas)
* [Dependências](#dependências)
* [Preocupações com Segurança](#preocupações-com-segurança)
* [Instalação](#instalação)
* [Teste Mínimo](#teste-mínimo)
* [Experimentos](#experimentos)
  * [Reivindicação 1](#reivindicação-1)
* [LICENSE](#license)

# Selos Considerados

Os selos considerados são: Disponíveis e Funcionais.

# Informações Básicas

Este código foi testado em uma máquina com Linux Mint 21.2 (64-bit), 8 GB de RAM e um processador Intel Core i5-4670 @ 3.40 GHz. O tempo de execução vai variar conforme o hardware, mas a tendência linear no tempo deve se manter. Os resultados de acurácia, por sua vez, são determinísticos e não dependem do sistema.

Para rodar este projeto, você precisará de:

* Git;
* Python (preferencialmente via Conda, Miniforge ou similar);
* Um compilador C++ (ex: g++ ≥ 5.2 ou clang ≥ 3.8);
* CMake (para compilar o TFHE e o PEBA);
* Paciência, pois a execução completa pode ser demorada (explicado mais abaixo)

# Dependências

As dependências estão listadas acima. As bibliotecas Python estão no arquivo `requirements.txt`.

# Preocupações com Segurança

A execução deste artefato **não apresenta riscos de segurança** aos avaliadores.

# Instalação

Aqui precisamos compilar duas dependências: TFHE e PEBA1. Primeiro, certifique-se de clonar o repositório **com os submódulos**:

```bash
git clone --recurse-submodules https://github.com/AndreisPurim/HEDimensionality.git hedim
cd hedim
```

Isso irá clonar o repositório e seus submódulos: `TFHE` no commit `bc71bfa` e `PEBA1` no commit `da28120`.

Para compilar o TFHE, siga as instruções em `tfhe/README.md`, ou simplesmente:

```bash
cd tfhe
make
make install   # use sudo se necessário
cd ..
```

Isso instalará o TFHE globalmente no sistema.

Agora, substitua o arquivo `main.cpp` original do PEBA1 por nossa versão customizada:

```bash
cp -f main.cpp peba1/src/main.cpp
```

E então compile o PEBA1:

```bash
cd peba1
mkdir build
cd build
cmake ..
make
cd ../../
```

O binário compilado estará em:

```bash
peba1/build/peba1
```

O código usa Python para preparar os dados, calcular resultados esperados e plotar. Crie um ambiente conda/miniforge (ou equivalente) e instale os requisitos:

```bash
conda create -n hedim python=3.12
conda activate hedim
pip install -r requirements.txt
```

Agora seu ambiente está pronto para execução.

# Teste Mínimo

Para executar um teste mínimo:

Rode:

```bash
python3 hedim_prepare_data.py
```

Esse script gera vetores aleatórios de N usuários com M amostras cada, com valores de 0 a 255 e ruído, salvando os vetores (e suas compressões) na pasta `output/normal/`. O tamanho do vetor reduzido (n) varia de 128 até 1, usando Projeção Aleatória Gaussiana (`sklearn`). Os vetores são salvos como:

```
{sample_id} {user_id} {n valores inteiros}
```

Além disso, o script calcula as distâncias euclidianas esperadas entre os vetores e salva em `comp_{n}_scores.csv` e as taxas FAR/FRR em `comp_{n}_rates.csv`.

Agora, para rodar a versão criptografada da distância euclidiana, execute:

```
python3 hedim_run_peba.py output/normal/comp_2_samples.txt 2 11 14
```

Esse exemplo compara os vetores de IDs 11 e 14 com dimensão 2. A saída será algo como:

```
[HEDim main.cpp] Sample A (ID 11):
[6 9 ]

[HEDim main.cpp] Sample B (ID 14):
[3 10 ]

[HEDim main.cpp] Executed in 566 seconds

[HEDim main.cpp] The result for HE Euclidean distance is 10 (expected 10)
```

Repita o processo para mais vetores se quiser calcular médias de tempo. Por exemplo, um vetor com n = 128 pode levar mais de 9 horas (33.536 segundos) para ser comparado.

# Experimentos

Você pode escolher entre duas abordagens:

* Executar todos os pares possíveis (N × M × tamanhos comprimidos), o que pode levar mais de 40 dias;
* Ou, escolher um par por dimensão, medir o tempo e usar esses valores no gráfico.

## Reivindicação 1

A Figura 2 do artigo pode ser reproduzida com este código. Altere o vetor `average_times` no script `hedim_plot.py` com os tempos medidos e execute:

```bash
python3 hedim_plot.py
```

# LICENSE

Este repositório está sob a licença MIT, conforme especificado no arquivo `LICENSE`.
