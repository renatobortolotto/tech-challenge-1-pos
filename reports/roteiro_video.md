# Roteiro Do Video Executivo - Ate 5 Minutos

## 0:00 - 0:30 | Abertura

O projeto analisa a satisfacao de clientes de um e-commerce a partir de dados de
pedidos, entregas e atendimento. O desafio de negocio e entender por que alguns
clientes se tornam promotores enquanto outros viram detratores, mesmo em jornadas
aparentemente parecidas.

## 0:30 - 1:10 | Problema de negocio

Hoje o NPS e coletado apenas depois da experiencia de compra. Isso dificulta uma
atuacao preventiva: quando a nota chega, muitas vezes o atraso, a reclamacao ou o
problema de atendimento ja afetou a percepcao do cliente.

Nosso objetivo foi identificar fatores criticos da jornada e propor um modelo capaz de
antecipar risco de detracao.

## 1:10 - 2:10 | Principais achados da EDA

A base tem 2.500 registros e 74,04% dos clientes aparecem como detratores. Os
principais fatores associados a queda do NPS foram atraso na entrega, reclamacoes,
contatos com atendimento e tempo de resolucao.

O ponto mais forte esta em atraso: clientes sem atraso tiveram 36,46% de detratores,
enquanto clientes com qualquer atraso chegaram a 78,72%.

Outro ponto critico esta em reclamacoes: ate 2 reclamacoes, a taxa de detratores foi
31,04%; com 3 ou mais reclamacoes, subiu para 82,77%.

## 2:10 - 3:10 | Solucao preditiva

Para transformar essa analise em acao, criamos uma target binaria chamada
`is_detractor`, que vale 1 quando o NPS e menor ou igual a 6.

O modelo escolhido foi uma Logistic Regression no cenario operacional sem vazamento.
Removemos identificadores, recompra em 30 dias e CSAT interno, porque essas variaveis
podem nao estar disponiveis no momento correto ou podem carregar informacao muito
proxima da satisfacao final.

No teste, o modelo atingiu ROC-AUC de 0,873, precision de 0,9227 e recall de 0,7732
para detratores.

## 3:10 - 4:20 | Como usar na pratica

A empresa pode usar o modelo para priorizar clientes por risco. Por exemplo: se um
pedido apresenta atraso, multiplas reclamacoes ou contatos recorrentes, ele pode ser
encaminhado para uma fila de atendimento preventivo.

As principais acoes recomendadas sao:

- comunicacao proativa em casos de atraso;
- atendimento prioritario para clientes com reclamacoes acumuladas;
- analise de causa-raiz em problemas recorrentes;
- comunicacao personalizada via CRM;
- monitoramento conjunto de NPS, SLA logistico, tempo de resolucao e recompra.

## 4:20 - 5:00 | Limitações e fechamento

A analise mostra associacoes, nao causalidade definitiva. Alem disso, o modelo deve ser
monitorado com dados futuros para garantir que continue valido.

Mesmo assim, o resultado principal e claro: a empresa nao precisa esperar a pesquisa
de NPS para agir. Dados operacionais ja mostram sinais de risco e podem orientar
decisoes preventivas para melhorar a experiencia do cliente.
