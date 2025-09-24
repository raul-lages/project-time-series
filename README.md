# ANÁLISE PREDITIVA DO DESEMPREGO NO BRASIL: UMA COMPARAÇÃO ENTRE MODELOS ECONOMÉTRICOS E DE MACHINE LEARNING EM SÉRIES TEMPORAIS

> **Grupo 13**
> * **Andre Gustavo Monteiro Dos Santos F** – RA 10424359
> * **Raul Santos Lages** – RA 10424621

## 1. INTRODUÇÃO

O mercado de trabalho é um pilar para o desenvolvimento socioeconômico, e a taxa de desemprego no Brasil serve como um termômetro da estabilidade do país. Historicamente, suas variações são influenciadas por ciclos econômicos e políticos. Contudo, eventos globais recentes, como a pandemia de COVID-19, introduziram uma volatilidade sem precedentes, gerando choques abruptos que desafiam os modelos de previsão tradicionais. A motivação central deste projeto é a profunda relevância social e econômica da taxa de desemprego, cuja antecipação orienta desde políticas públicas de amparo ao trabalhador até decisões estratégicas de investimento no setor privado.

A crescente complexidade do cenário macroeconômico, marcada por alta volatilidade e quebras estruturais, levanta questionamentos sobre a confiabilidade de projeções baseadas apenas em modelos lineares tradicionais. Estudos comparativos observaram que, durante períodos de incerteza econômica, modelos não-lineares de machine learning podem apresentar erros de previsão menores que os modelos econométricos lineares, sugerindo que estes últimos podem ter sua robustez limitada em períodos de crise. Surge, assim, a motivação técnica de investigar o potencial de diferentes abordagens de modelagem. A hipótese é que a capacidade de algoritmos de machine learning em capturar padrões não-lineares pode gerar previsões mais acuradas em determinados contextos, justificando uma análise comparativa rigorosa.

A escolha deste tema se justifica por seu impacto direto na vida de milhões de brasileiros e pela necessidade de fornecer análises mais confiáveis e embasadas aos gestores públicos. Diante de um problema tão complexo, este projeto se propõe a realizar uma análise pluralista, avaliando um espectro de ferramentas que vai desde os modelos econométricos consagrados até os de machine learning que vêm ganhando popularidade. O alinhamento com o Objetivo de Desenvolvimento Sustentável (ODS) 8, que visa promover o trabalho decente e o crescimento econômico, reforça a relevância do estudo.

Nesse contexto, o objetivo geral do projeto é desenvolver e comparar modelos preditivos baseados em séries temporais capazes de analisar e projetar a taxa de desemprego no Brasil, utilizando dados históricos para expandir o conhecimento sobre o desempenho de diferentes abordagens e apoiar a formulação de políticas públicas.

Para alcançar tal objetivo, foram definidos os seguintes objetivos específicos:

* **Estruturar uma base de dados multivariada** a partir de fontes oficiais como o Instituto Brasileiro de Geografia e Estatística (IBGE) e o Banco Central do Brasil (BCB). A base contempla a série histórica da taxa de desemprego (PNAD Contínua) e 13 outras variáveis macroeconômicas relevantes, como a Taxa Selic, o IPCA, a Taxa de Câmbio, o IBC-Br e o Índice de Commodities (IC-Br). Os dados possuem granularidade mensal e abrangem o período de abril de 2012 a junho de 2025, totalizando 159 observações.
* **Analisar as características da série temporal**, identificando componentes de tendência, sazonalidade e possíveis quebras estruturais.
* **Implementar e avaliar comparativamente um espectro de modelos**, incluindo os econométricos (ARIMA, VAR) e de machine learning (Prophet, LSTM, XGBoost).
* **Aferir o desempenho das diferentes abordagens** utilizando métricas de erro (RMSE, MAPE, MAE), analisando o trade-off entre a acurácia preditiva e a interpretabilidade de cada modelo.
* **Sintetizar os resultados para elaborar recomendações práticas** que possam apoiar o planejamento de políticas voltadas ao mercado de trabalho brasileiro.

## 2. REFERENCIAL TEÓRICO

Este projeto se posiciona na interseção da econometria e da ciência de dados, comparando modelos de diferentes naturezas para a previsão do desemprego. A seguir, são definidos os conceitos centrais e discutidos os trabalhos correlacionados que fundamentam esta análise.

### Modelos Econométricos Tradicionais

* **ARIMA (Autoregressive Integrated Moving Average):** É um dos modelos mais clássicos para previsão de séries temporais, formalizado na metodologia de Box e Jenkins (1970). Ele combina três componentes: um componente autorregressivo (AR), que modela a relação entre uma observação e um número de observações defasadas; um componente de médias móveis (MA), que modela o erro da previsão como uma combinação linear de erros passados; e o componente integrado (I), que utiliza a diferenciação dos dados para tornar a série temporal estacionária.

* **VAR (Vector Autoregressive):** O modelo VAR, introduzido na macroeconomia por Sims (1980), é uma generalização do modelo autorregressivo para múltiplas séries temporais. Em um sistema VAR, cada variável é modelada como uma função linear de seus próprios valores passados e dos valores passados de todas as outras variáveis do sistema. Isso o torna particularmente útil para descrever a dinâmica de interdependência entre variáveis macroeconômicas.

### Modelos de Machine Learning

* **Prophet:** Desenvolvido pelo Facebook, o Prophet é um procedimento de previsão baseado em um modelo aditivo decomponível (TAYLOR; LETHAM, 2017), onde tendências não-lineares são ajustadas com sazonalidades e efeitos de feriados. Foi projetado para ser robusto e seus parâmetros são facilmente interpretáveis, permitindo que analistas com conhecimento de domínio ajustem o modelo de forma intuitiva.

* **LSTM (Long Short-Term Memory):** As redes LSTM são um tipo avançado de rede neural recorrente (RNN) projetadas para aprender dependências de longo prazo em dados sequenciais (HOCHREITER; SCHMIDHUBER, 1997). Por meio de mecanismos de "portões" (gates) que controlam o fluxo de informação, as LSTMs podem reter informações relevantes por longos períodos, tornando-as adequadas para capturar padrões complexos e não-lineares.

* **XGBoost (Extreme Gradient Boosting):** É um algoritmo de ensemble baseado em árvores de decisão que se destaca pela sua alta performance e eficiência (CHEN; GUESTRIN, 2016). Ele constrói um modelo preditivo de forma sequencial, onde cada nova árvore corrige os erros do conjunto de árvores anterior. Para séries temporais, o XGBoost é utilizado transformando o problema de previsão em um de regressão supervisionada.

### Trabalhos Correlacionados e Alternativas de Solução

A literatura recente oferece um rico panorama de comparações entre essas abordagens. O trabalho de Athey e Imbens (2019) fornece a justificativa teórica para este projeto, defendendo a integração de métodos de machine learning na pesquisa econômica para alavancar seu poder preditivo. Adicionalmente, revisões de literatura validam a seleção de modelos deste projeto como representativa do "estado da arte" (traduzindo, temos algo como "o que há de mais moderno"), confirmando a relevância da comparação entre abordagens econométricas e de machine learning (SHOBANA; UMAMAHESWARI, 2021).

Estudos empíricos sobre a previsão do desemprego revelam um trade-off claro. Ao analisarem a taxa de desemprego na Turquia, considerando uma amostra de 176 casos (janeiro de 2008 a agosto de 2022), pesquisadores concluíram que, embora o modelo ARIMA fosse adequado para o período completo, uma rede neural artificial (ANN) apresentou erros de previsão menores durante a alta incerteza da pandemia de COVID-19 (YAMACLI; YAMACLI, 2023). Isso evidencia a vantagem potencial de modelos de ML em cenários de quebra estrutural. De forma similar, a pesquisa sobre o mercado de trabalho irlandês,
considerando uma amostra de 305 casos (janeiro de 1998 a maio de 2023), destacou o bom desempenho de modelos como XGBoost e Ridge Regression, que possuem mecanismos intrínsecos para evitar sobreajuste (overfitting), uma limitação comum em dados ruidosos (KRISHNAMURTHY, 2023).

Outros estudos apontam para a importância dos dados de entrada. Foi demonstrado que a acurácia de modelos como ARIMA e VAR para prever o desemprego em Gana foi significativamente melhorada com a inclusão de dados do Google Trends como variáveis exógenas (ADU et al., 2023). Isso sugere que uma alternativa ou complemento à sofisticação do modelo é a engenharia de atributos com dados não convencionais.

Em domínios análogos, a hierarquia de desempenho se repete. Na previsão de preços de alimentos, modelos de deep learning (como LSTM) superaram o ARIMA, que por sua vez superou o Prophet em acurácia (MENCULINI et al., 2021). No entanto, a vantagem do deep learning veio ao custo de maior complexidade e tempo computacional. Por fim, foi observado na previsão de casos de COVID-19 que a acurácia do Prophet diminuía em horizontes de previsão mais longos em comparação com o ARIMA, destacando que a escolha do melhor modelo pode depender do horizonte temporal da previsão (SATRIO et al., 2021).

## 3. PIPELINE DA SOLUÇÃO

Para conduzir a análise de forma estruturada e mitigar os riscos identificados, propõe-se o seguinte pipeline de trabalho, dividido em sete etapas principais:

1.  **Coleta e Estruturação dos Dados:** Consolidação da base de dados com as 14 variáveis macroeconômicas de fontes oficiais (IBGE, BCB), garantindo o alinhamento temporal e a consistência dos dados no período de abril de 2012 a junho de 2025.
2.  **Análise Exploratória de Dados (AED):** Investigação aprofundada da série temporal da taxa de desemprego e das demais variáveis. Esta etapa incluirá a visualização das séries, a análise de componentes de tendência e sazonalidade, a verificação de correlações entre as variáveis e a aplicação de testes estatísticos (como o teste de Dickey-Fuller Aumentado) para avaliar a estacionariedade dos dados.
3.  **Pré-processamento e Engenharia de Atributos:** Com base na AED, os dados serão preparados para a modelagem. Isso pode incluir a diferenciação das séries para torná-las estacionárias, a normalização dos dados e a criação de novas variáveis (features), como defasagens (lags) das próprias séries e variáveis baseadas na data.
4.  **Prova de Conceito (PoC) com Modelo Univariado:** Para endereçar a preocupação com o volume de dados, será realizada uma prova de conceito rápida. Um modelo ARIMA univariado será treinado para verificar a viabilidade de se obter previsões razoáveis com o volume de dados disponível antes de prosseguir para modelos mais complexos.
5.  **Divisão dos Dados e Treinamento dos Modelos:** O conjunto de dados será dividido em conjuntos de treinamento e teste. Os cinco modelos propostos (ARIMA, VAR, Prophet, LSTM, XGBoost) serão implementados e treinados.
6.  **Avaliação e Comparação de Desempenho:** Os modelos treinados serão utilizados para fazer previsões sobre o conjunto de teste. O desempenho será quantificado utilizando métricas de erro (RMSE, MAPE, MAE) e será realizada uma análise qualitativa do trade-off entre acurácia e interpretabilidade.
7.  **Análise de Resultados e Documentação Final:** Os resultados da comparação serão sintetizados e interpretados no contexto da economia brasileira. Serão elaboradas as conclusões do estudo e as recomendações práticas, culminando no relatório final do projeto.

## 4. CRONOGRAMA

O cronograma a seguir detalha as atividades planejadas e as respectivas datas de entrega, estruturadas em três fases principais.

| Entrega | Data Limite | Atividades Principais |
| :--- | :--- | :--- |
| **Entrega 2** | 26 de Setembro de 2025 | Definição do escopo, revisão da literatura, proposta do pipeline da solução e planejamento inicial. |
| **Entrega 3** | 31 de Outubro de 2025 | Análise Exploratória de Dados (EDA), pré-processamento, implementação e avaliação de um modelo base (primeira versão). |
| **Entrega Final** | 28 de Novembro de 2025 | Refinamento de todos os modelos, análise comparativa final, discussão dos resultados, conclusão e entrega do projeto completo. |

## 5. REFERÊNCIAS

ADU, Williams Kwasi; APPIAHENE, Peter; AFRIFA, Stephen. VAR, ARIMAX and ARIMA models for nowcasting unemployment rate in Ghana using Google trends. *Journal of Electrical Systems and Information Technology*, v. 10, n. 12, p. 1-16, 2023. Disponível em: <https://link.springer.com/article/10.1186/s43067-023-00078-1>. Acesso em: 24 de setembro de 2025.

ATHEY, Susan; IMBENS, Guido W. Machine Learning Methods That Economists Should Know About. *Annual Review of Economics*, v. 11, p. 685-725, 2019. Disponível em: <https://www.annualreviews.org/content/journals/10.1146/annurev-economics-080217-053433>. Acesso em: 24 de setembro de 2025.

BOX, G. E. P.; JENKINS, G. M. *Time series analysis: Forecasting and control*. San Francisco: Holden-Day, 1970. Disponível em: <https://archive.org/details/timeseriesanalys0000boxg>. Acesso em: 24 de setembro de 2025.

CHEN, T.; GUESTRIN, C. XGBoost: A Scalable Tree Boosting System. In: PROCEEDINGS OF THE 22ND ACM SIGKDD INTERNATIONAL CONFERENCE ON KNOWLEDGE DISCOVERY AND DATA MINING, 2016, San Francisco. *Anais [...]*. San Francisco: ACM, 2016. p. 785-794. Disponível em: <https://arxiv.org/abs/1603.02754>. Acesso em: 24 de setembro de 2025.

HOCHREITER, S.; SCHMIDHUBER, J. Long Short-Term Memory. *Neural Computation*, v. 9, n. 8, p. 1735-1780, 1997. Disponível em: <https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext>. Acesso em: 24 de setembro de 2025.

KRISHNAMURTHY, Shree Hari. *A Comprehensive study of applying machine learning algorithms for time series data prediction to the Irish Labour Market Unemployment Rate*. 2023. 33 f. Dissertação (Mestrado em Análise de Dados) - School of Computing, National College of Ireland, Dublin, 2023. Disponível em: <https://norma.ncirl.ie/id/eprint/7164>. Acesso em: 24 de setembro de 2025.

MENCULINI, Lorenzo et al. Comparing Prophet and Deep Learning to ARIMA in Forecasting Wholesale Food Prices. *Forecasting*, v. 3, n. 3, p. 644-662, set. 2021. Disponível em: <https://www.mdpi.com/2571-9394/3/3/40>. Acesso em: 24 de setembro de 2025.

SATRIO, Christophorus Beneditto Aditya et al. Time series analysis and forecasting of coronavirus disease in Indonesia using ARIMA model and PROPHET. *Procedia Computer Science*, v. 179, p. 524-532, 2021. Disponível em: <https://www.sciencedirect.com/science/article/pii/S1877050921000417>. Acesso em: 24 de setembro de 2025.

SHOBANA, G.; UMAMAHESWARI, K. Forecasting by Machine Learning Techniques and Econometrics: A Review. In: INTERNATIONAL CONFERENCE ON INVENTIVE COMPUTATION TECHNOLOGIES (ICICT), 6., 2021, Coimbatore. *Anais [...]*. Coimbatore: IEEE, 2021. p. 1010-1016. Disponível em: <https://ieeexplore.ieee.org/abstract/document/9358514>. Acesso em: 24 de setembro de 2025.

SIMS, C. A. Macroeconomics and Reality. *Econometrica*, v. 48, n. 1, p. 1-48, 1980. Disponível em: <https://ideas.repec.org/a/ecm/emetrp/v48y1980i1p1-48.html>. Acesso em: 24 de setembro de 2025.

TAYLOR, Sean J.; LETHAM, Benjamin. *Forecasting at Scale*. 2017. Preprint. Disponível em: <https://peerj.com/preprints/3190/>. Acesso em: 24 de setembro de 2025.

YAMACLI, Dilek Surekci; YAMACLI, Serhan. Estimation of the unemployment rate in Turkey: A comparison of the ARIMA and machine learning models including Covid-19 pandemic periods. *Heliyon*, v. 9, n. 1, p. e12796, jan. 2023. Disponível em: <https://www.cell.com/heliyon/pdf/S2405-8440(23)00003-8.pdf>. Acesso em: 24 de setembro de 2025.
