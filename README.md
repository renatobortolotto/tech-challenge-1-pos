# Tech Challenge Fase 1 - NPS Preditivo

Projeto desenvolvido para o Tech Challenge da Fase 1: analisar dados historicos de
pedidos, entregas e atendimento para entender fatores que impactam o NPS de um
e-commerce e propor um modelo preditivo para antecipar clientes com risco de se
tornarem detratores.

## Objetivo

Responder, com linguagem de negocio e apoio analitico:

- Qual problema de negocio esta sendo resolvido?
- Quais fatores operacionais mais impactam a satisfacao do cliente?
- O que mais gera detratores?
- Existe algum ponto de ruptura na experiencia?
- Como um modelo de IA poderia antecipar risco de NPS baixo?

## Base de dados

Arquivo principal: `data/desafio_nps_fase_1.csv`.

A base contem 2.500 registros, 19 colunas e nenhum valor ausente. As variaveis
incluem dados de cliente, pedido, pagamento, logistica, atendimento, recompra,
CSAT interno e `nps_score`.

A variavel de satisfacao e `nps_score`, coletada apos a experiencia de compra. Para
a analise, as notas foram agrupadas em:

- Detrator: `nps_score <= 6`
- Neutro: `7 <= nps_score <= 8`
- Promotor: `nps_score >= 9`

Para o modelo preditivo foi criada a target `is_detractor`, igual a 1 quando o
cliente e detrator. Essa formulacao foi escolhida porque permite acao preventiva:
priorizar clientes em risco antes da pesquisa de NPS.

## Metodologia

1. Validacao da base e criacao das classes de NPS.
2. Analise exploratoria com foco em negocio.
3. Identificacao de fatores criticos e pontos de ruptura.
4. Criacao de dois cenarios de modelagem:
   - operacional sem vazamento, excluindo `repeat_purchase_30d` e `csat_internal_score`;
   - comparativo com sinais pos-jornada, para demonstrar risco de vazamento.
5. Treino de modelos baseline e preditivos:
   - `DummyClassifier`;
   - `LogisticRegression`;
   - `RandomForestClassifier`.
6. Avaliacao com precision, recall, F1 para detratores, matriz de confusao e ROC-AUC.

## Principais resultados

- A base tem 74,04% de detratores, 17,92% de neutros e 8,04% de promotores.
- O NPS calculado da base e aproximadamente -66, indicando uma experiencia muito
  pressionada por problemas operacionais.
- As variaveis mais associadas a queda do NPS foram:
  - atraso na entrega: correlacao de -0,597;
  - numero de reclamacoes: correlacao de -0,497;
  - contatos com atendimento: correlacao de -0,351;
  - tempo de resolucao: correlacao de -0,191.
- Atraso e reclamacoes mostram pontos de ruptura claros:
  - clientes com qualquer atraso tiveram taxa de detratores de 78,72%, contra 36,46% sem atraso;
  - clientes com 3 ou mais reclamacoes tiveram taxa de detratores de 82,77%, contra 31,04% ate 2 reclamacoes.
- O melhor modelo operacional foi `LogisticRegression`, com:
  - ROC-AUC: 0,873;
  - precision para detratores: 0,9227;
  - recall para detratores: 0,7732;
  - F1 para detratores: 0,8414.

## Recomendações de negocio

- Logistica: criar gatilhos de comunicacao proativa para pedidos com atraso ou risco
  de atraso.
- Atendimento: priorizar clientes com multiplos contatos ou reclamacoes acumuladas.
- Operacoes: tratar reclamacoes recorrentes como causa-raiz, nao apenas como casos
  isolados.
- CRM: usar a probabilidade de risco para acionar clientes antes da pesquisa de NPS.
- Gestao: acompanhar NPS junto com SLA logistico, tempo de resolucao, reclamacoes e
  recompra.

## Como reproduzir

Crie um ambiente Python e instale as dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Execute o notebook:

```bash
jupyter nbconvert --to notebook --execute notebooks/eda_modelagem_nps.ipynb --inplace
```

Os dados numericos usados nos graficos foram exportados em CSV em
`reports/chart_data/`. O arquivo `00_manifesto_dados_graficos.csv` mapeia cada
grafico aos respectivos arquivos numericos.

## Estrutura do repositorio

```text
data/
  desafio_nps_fase_1.csv
docs/
  tech_challenge_fase_1.pdf
Predictive_NPS.pdf
models/
  modelo_risco_detrator.pkl
  modelo_risco_detrator_metadata.json
notebooks/
  eda_modelagem_nps.ipynb
reports/
  chart_data/
  figures/
  slides_storytelling.md
  roteiro_video.md
```

## Limitações

- A base e historica e pode nao representar mudancas futuras da operacao.
- Correlacao nao prova causalidade.
- `repeat_purchase_30d` e `csat_internal_score` podem representar informacoes
  posteriores ou proximas demais da satisfacao; por isso ficaram fora do modelo
  operacional escolhido.
- Como a classe detratora e majoritaria, o modelo deve ser monitorado tambem por
  custo operacional, falsos positivos e impacto real das intervencoes.
