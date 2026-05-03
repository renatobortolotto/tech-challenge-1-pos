# Case NPS Preditivo

## 1. Contexto

O e-commerce cresceu em volume de pedidos, entregas e interacoes com clientes.
Apesar de indicadores operacionais parecidos em alguns casos, a experiencia final
varia muito: parte dos clientes recomenda a marca, enquanto outra parte se torna
detratora.

Pergunta central: quais fatores operacionais explicam a satisfacao e como agir antes
da pesquisa de NPS?

---

## 2. O problema de negocio

Hoje o NPS e coletado apenas apos o encerramento da jornada. Isso limita a reacao da
empresa, porque o problema ja aconteceu quando a nota chega.

A oportunidade e transformar dados de entrega, atendimento e pedido em alertas
preventivos para reduzir detratores.

---

## 3. Retrato da base

- 2.500 pedidos/clientes analisados
- 19 variaveis de cliente, pedido, logistica, atendimento e satisfacao
- Sem valores ausentes
- Distribuicao de NPS:
  - 74,04% detratores
  - 17,92% neutros
  - 8,04% promotores
- NPS calculado: aproximadamente -66

---

## 4. Principais sinais encontrados

Os fatores mais associados a queda do NPS foram:

- atraso na entrega;
- reclamacoes registradas;
- contatos com atendimento;
- tempo de resolucao.

Esses fatores sao operacionais e acionaveis, o que torna a analise util para gestao.

---

## 5. O que mais gera detratores

Clientes sem atraso tiveram taxa de detratores de 36,46%.

Clientes com qualquer atraso tiveram taxa de detratores de 78,72%.

Atraso nao e apenas um indicador logistico: ele muda diretamente a percepcao da
experiencia.

---

## 6. Ponto de ruptura em reclamacoes

Clientes com ate 2 reclamacoes tiveram taxa de detratores de 31,04%.

Clientes com 3 ou mais reclamacoes tiveram taxa de detratores de 82,77%.

Esse e um gatilho claro para atendimento prioritario.

---

## 7. Segmentos de clientes

A diferenca por regiao, idade, tempo de relacionamento e valor do pedido foi menor
do que a diferenca provocada por atraso e reclamacoes.

Isso sugere que a prioridade deve estar menos em perfil demografico e mais em
eventos da jornada.

---

## 8. Modelo preditivo

Foi criado um modelo para prever risco de cliente detrator:

- Target: `is_detractor = 1` quando `nps_score <= 6`
- Modelo escolhido: Logistic Regression
- Cenario escolhido: operacional sem vazamento
- Variaveis removidas do modelo operacional:
  - `customer_id`
  - `order_id`
  - `repeat_purchase_30d`
  - `csat_internal_score`

---

## 9. Performance do modelo

No conjunto de teste, o modelo operacional atingiu:

- ROC-AUC: 0,873
- Precision para detratores: 0,9227
- Recall para detratores: 0,7732
- F1 para detratores: 0,8414

Na pratica, o modelo ajuda a ordenar clientes por risco e priorizar intervencoes.

---

## 10. Recomendacoes

- Logistica: acionar clientes de forma proativa quando houver risco de atraso.
- Atendimento: criar fila prioritaria para clientes com reclamacoes ou contatos
  repetidos.
- Operacoes: investigar causas-raiz de atraso e reclamacao.
- CRM: personalizar comunicacao conforme risco de detracao.
- Gestao: acompanhar NPS com SLA logistico, resolucao, reclamacoes e recompra.

---

## 11. Limitações e riscos

- A base historica pode nao representar a operacao futura.
- Correlacao nao prova causalidade.
- Variaveis pos-jornada podem inflar metricas.
- O modelo deve ser monitorado em producao por impacto real, custo de acao e falsos
  positivos.

---

## 12. Fechamento

O principal aprendizado e que o NPS pode ser antecipado por sinais operacionais.

A empresa nao precisa esperar a nota do cliente para agir: atraso, reclamacao e
recorrencia de atendimento ja indicam risco e permitem intervencao preventiva.
