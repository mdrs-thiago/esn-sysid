clc
clear all
close all

load resultadosOtimizacaoAEIQ.mat

%% Configurações AEIQ (etapa 2)

numPulses2 = 15;     % população quântica
numClassic2 = 45;    % população clássica
genToWidth2 = 5;    % intervalo de gerações para atualização da população quântica
generations2 = 700;   % número de gerações
crossoveRRSE = 0.6;  % taxa de crossover clássico real
qCrossoveRRSE = 0.4; % taxa de crossover quântico
gap2 = numClassic2;    

fun2 = 'calculaErroPesos'; % nome da função de aptidão

%% Etapa 2 - Otimização dos pesos do reservatório
tic
aux2=0;
for exp = 1:numExp

  reservatorio = ceil(sol{exp}(1)); % número de neurônios no reservatório

  raio =   sol{exp}(2);      % raio espectral
 
  ScalWin = sol{exp}(3);     % constante de multiplicação da matriz Win
 
  ScalWback = sol{exp}(4);   % constante de multiplicação da matriz Wback
 
  vazamento = sol{exp}(5);   % constante de vazamento do neurônio leaky integrator

  conectividade = sol{exp}(6); % percentual de conectividade do reservatório

  nInputUnits =  size(trainInputSequence,2); 
  nInternalUnits = reservatorio; 
  nOutputUnits = size(trainOutputSequence,2);

  esnP{exp} = generate_esn2(nInputUnits, nInternalUnits, nOutputUnits, ScalWin, ScalWback, conectividade, ...
      'spectralRadius',raio,'inputScaling',1* ones(nInputUnits,1),'inputShift',0*ones(nInputUnits,1), ...
      'teacherScaling',1*ones(nOutputUnits,1),'teacherShift',0*ones(nOutputUnits,1),'feedbackScaling', 1, ...
      'type', 'leaky_esn','leakage',vazamento,...
      'reservoirActivationFunction','tanh',...
      'outputActivationFunction', 'identity','inverseOutputActivationFunction','identity',...
      'methodWeightCompute','pseudoinverse', 'timeConstants',1 * ones(nInternalUnits,1));  
    
  esnP{exp}.internalWeights = esnP{exp}.spectralRadius * esnP{exp}.internalWeights_UnitSR;  
  
  posicoes = esnP{exp}.internalWeights ~= 0; % posições de W com valores diferentes de zero

  num_vaRRSE  = size(esnP{exp}.internalWeights(posicoes),1); % total de pesos não nulos de W

  limInf2 = (-0.5.*ones(1,num_vaRRSE))'; 
  limSup2 = (0.5.*ones(1,num_vaRRSE))';
 
  [solP{exp}, classic,  evobestFitness2{exp}, trace] = aeiq_br_real(num_vaRRSE, numPulses2, numClassic2, ...
     genToWidth2, generations2, crossoveRRSE, qCrossoveRRSE, showEv,  gap2, exp,seed,fun2,limInf2,limSup2,trainInputSequence,trainOutputSequence,valInputSequence,yValidacao,yMin,yMax,...
    nForgetPoints,esnP{exp},metricaErro,posicoes);

  aux2 = aux2+evobestFitness2{exp};

  bestFitness2(exp) = min(evobestFitness2{exp});

  esnP{exp}.internalWeights(posicoes) = solP{exp}';   
  opts3.tol = 1e-3;
  opts3.disp = 0;
  maxVal = max(abs(eigs(esnP{exp}.internalWeights,1, 'lm', opts3)));
  esnP{exp}.internalWeights = esnP{exp}.internalWeights/maxVal;
  esnP{exp}.internalWeights = esnP{exp}.internalWeights.*esnP{exp}.spectralRadius;

  % Treinamento da ESN
  trESNP{exp} = ...
  train_esn(trainInputSequence, trainOutputSequence, esnP{exp}, nForgetPoints) ; 
  
  % Previsão do conjunto de treinamento
  [predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, trESNP{exp}, nForgetPoints);
  yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);
  % Cálculo do RMSE no conjunto de treinamento
  trainErrorPesos(exp) = feval(metricaErro,yTreinoPrevisto, yTreino);
  RRSETrainErrorPesos(exp) = feval('RRSE',yTreinoPrevisto, yTreino);
  fitTrainErrorPesos(exp) = feval('fit',yTreinoPrevisto, yTreino);
 
  % Previsão do conjunto de validação 
  [predictedValOutput,finalState] = test_esn_modificada(valInputSequence, trESNP{exp}, 0,'startingState',finalState);
   yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
  % Cálculo do RMSE no conjunto de validação
  valErrorPesos(exp) = feval(metricaErro,yValPrevisto, yValidacao);
  RRSEValErrorPesos(exp) = feval('RRSE',yValPrevisto, yValidacao);
  fitValErrorPesos(exp) = feval('fit',yValPrevisto, yValidacao);
  
  % Previsão do conjunto de teste
  predictedTestOutput = test_esn_modificada(testInputSequence, trESNP{exp}, 0,'startingState',finalState);
  %predictedTestOutput = test_esn_modificada(testInputSequence, best_ESN, 0);
  yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);
  %Cálculo do RMSE no conjunto de teste
  testErrorPesos(exp) = feval(metricaErro,yTestePrevisto, yTeste);
  RRSETestErrorPesos(exp) = feval('RRSE',yTestePrevisto, yTeste);
  fitTestErrorPesos(exp) = feval('fit',yTestePrevisto, yTeste);
  
%   mse_test_error_y1_pesos (exp) = feval('MSE',yTestePrevisto(:,1), yTeste(:,1));
%   mse_test_error_y2_pesos (exp) = feval('MSE',yTestePrevisto(:,2), yTeste(:,2));
  
  RRSE_test_error_y1_pesos  (exp) = feval('RRSE',yTestePrevisto(:,1), yTeste(:,1));
  RRSE_test_error_y2_pesos  (exp)= feval('RRSE',yTestePrevisto(:,2), yTeste(:,2));
  
  clear posicoes limInf2 limSup2 maxVal num_vaRRSE 
end   
tempoTotalPesos = toc;
% Fim Otimização dos Pesos do Reservatório

MeanFitness2 = mean(bestFitness2);
StdFitness2 = std(bestFitness2);
meanEvoBestFitness2 = aux2./numExp;

figure('Name','Curva de Evolução - Etapa 2 - Otimização dos pesos AEIQ','NumberTitle','off');
plot(meanEvoBestFitness2)
title('Curva de Evolução');
xlabel('Gerações');
ylabel('Aptidão');
savefig('Curva_Evolucao_Etapa_2');

clear opts3 aux2 classic trace reservatorio raio ScalWin ScalWback vazamento conectividade...
      nInputUnits nInternalUnits nOutputUnits predictedTrainOutput predictedValOutput predictedTestOutput finalState...
      yTreinoPrevisto yValPrevisto yTestePrevisto

%% Cálculo de métricas, Impressão de resultados e gráficos

tempoTotalRandom = tRandomParcial + tgen;
tempoTotalSRG = tSRGParcial + tgen;
tempoMedioRandom = tempoTotalRandom./numExp;
tempoMedioSRG = tempoTotalSRG./numExp;
tempoMedioPesos = tempoTotalPesos./numExp;

meanTrainErrorSRG = mean(trainErrorSRG);
meanValErrorSRG = mean(valErrorSRG);
meanTestErrorSRG = mean(testErrorSRG);
stdTrainErrorSRG = std(trainErrorSRG);
stdValErrorSRG = std(valErrorSRG);
stdTestErrorSRG = std(testErrorSRG);
meanRRSETrainErrorSRG = mean(RRSETrainErrorSRG);
meanRRSEValErrorSRG = mean(RRSEValErrorSRG);
meanRRSETestErrorSRG = mean(RRSETestErrorSRG);
stdRRSETrainErrorSRG = std(RRSETrainErrorSRG);
stdRRSEValErrorSRG = std(RRSEValErrorSRG);
stdRRSETestErrorSRG = std(RRSETestErrorSRG);
meanfitTrainErrorSRG = mean(fitTrainErrorSRG);
meanfitValErrorSRG = mean(fitValErrorSRG);
meanfitTestErrorSRG = mean(fitTestErrorSRG);
stdfitTrainErrorSRG = std(fitTrainErrorSRG);
stdfitValErrorSRG = std(fitValErrorSRG);
stdfitTestErrorSRG = std(fitTestErrorSRG);

meanTrainErrorRandom = mean(trainErrorRandom);
meanValErrorRandom = mean(valErrorRandom);
meanTestErrorRandom = mean(testErrorRandom);
stdTrainErrorRandom = std(trainErrorRandom);
stdValErrorRandom = std(valErrorRandom);
stdTestErrorRandom = std(testErrorRandom);
meanRRSETrainErrorRandom = mean(RRSETrainErrorRandom);
meanRRSEValErrorRandom = mean(RRSEValErrorRandom);
meanRRSETestErrorRandom = mean(RRSETestErrorRandom);
stdRRSETrainErrorRandom = std(RRSETrainErrorRandom);
stdRRSEValErrorRandom = std(RRSEValErrorRandom);
stdRRSETestErrorRandom = std(RRSETestErrorRandom);
meanfitTrainErrorRandom = mean(fitTrainErrorRandom);
meanfitValErrorRandom = mean(fitValErrorRandom);
meanfitTestErrorRandom = mean(fitTestErrorRandom);
stdfitTrainErrorRandom = std(fitTrainErrorRandom);
stdfitValErrorRandom = std(fitValErrorRandom);
stdfitTestErrorRandom = std(fitTestErrorRandom);

meanTrainErrorPesos = mean(trainErrorPesos);
meanValErrorPesos = mean(valErrorPesos);
meanTestErrorPesos = mean(testErrorPesos);
stdTrainErrorPesos = std(trainErrorPesos);
stdValErrorPesos = std(valErrorPesos);
stdTestErrorPesos = std(testErrorPesos);
meanRRSETrainErrorPesos = mean(RRSETrainErrorPesos);
meanRRSEValErrorPesos = mean(RRSEValErrorPesos);
meanRRSETestErrorPesos = mean(RRSETestErrorPesos);
stdRRSETrainErrorPesos = std(RRSETrainErrorPesos);
stdRRSEValErrorPesos = std(RRSEValErrorPesos);
stdRRSETestErrorPesos = std(RRSETestErrorPesos);
meanfitTrainErrorPesos = mean(fitTrainErrorPesos);
meanfitValErrorPesos = mean(fitValErrorPesos);
meanfitTestErrorPesos = mean(fitTestErrorPesos);
stdfitTrainErrorPesos = std(fitTrainErrorPesos);
stdfitValErrorPesos = std(fitValErrorPesos);
stdfitTestErrorPesos = std(fitTestErrorPesos);

% mean_mse_test_error_y1_SRG = mean(mse_test_error_y1_SRG);
% mean_mse_test_error_y2_SRG = mean(mse_test_error_y2_SRG);

mean_RRSE_test_error_y1_SRG  = mean(RRSE_test_error_y1_SRG);
mean_RRSE_test_error_y2_SRG  = mean(RRSE_test_error_y2_SRG);

% std_mse_test_error_y1_SRG = std(mse_test_error_y1_SRG);
% std_mse_test_error_y2_SRG = std(mse_test_error_y2_SRG);

std_RRSE_test_error_y1_SRG  = std(RRSE_test_error_y1_SRG);
std_RRSE_test_error_y2_SRG  = std(RRSE_test_error_y2_SRG);

% mean_mse_test_error_y1_random = mean(mse_test_error_y1_random);
% mean_mse_test_error_y2_random = mean(mse_test_error_y2_random);

mean_RRSE_test_error_y1_random = mean(RRSE_test_error_y1_random);
mean_RRSE_test_error_y2_random = mean(RRSE_test_error_y2_random);
% 
% std_mse_test_error_y1_random = std(mse_test_error_y1_random);
% std_mse_test_error_y2_random = std(mse_test_error_y2_random);

std_RRSE_test_error_y1_random = std(RRSE_test_error_y1_random);
std_RRSE_test_error_y2_random = std(RRSE_test_error_y2_random);


% mean_mse_test_error_y1_pesos = mean(mse_test_error_y1_pesos);
% mean_mse_test_error_y2_pesos = mean(mse_test_error_y2_pesos);

mean_RRSE_test_error_y1_pesos = mean(RRSE_test_error_y1_pesos);
mean_RRSE_test_error_y2_pesos = mean(RRSE_test_error_y2_pesos);

% std_mse_test_error_y1_pesos = std(mse_test_error_y1_pesos);
% std_mse_test_error_y2_pesos = std(mse_test_error_y2_pesos);

std_RRSE_test_error_y1_pesos = std(RRSE_test_error_y1_pesos);
std_RRSE_test_error_y2_pesos = std(RRSE_test_error_y2_pesos);

% melhor ESN SRG em relação ao teste
[bestErrorSRG,posBestSRG]=min(testErrorSRG);
bestEsnSRG=tr_esn_SRG{1,posBestSRG};
% Previsão do conjunto de treinamento
[predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, bestEsnSRG, nForgetPoints);
yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);
% Previsão do conjunto de validação 
[predictedValOutput,finalState] = test_esn_modificada(valInputSequence, bestEsnSRG, 0,'startingState',finalState);
yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
% Previsão do conjunto de teste
predictedTestOutput = test_esn_modificada(testInputSequence, bestEsnSRG, 0,'startingState',finalState);
yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);


for j = 1:size(yTestePrevisto,2)
    figure('Name','Previsão das variáveis de saída com a melhor ESN (teste) da Etapa 2 - SRG','NumberTitle','on');
    amostras = [1:length([yTreino; yValidacao;yTeste])];
    subplot(3,1,1);
    plot(amostras((nForgetPoints+1):length(yTreino)),yTreino((nForgetPoints+1):end,j),'b');
    hold on
    plot(amostras((nForgetPoints+1):length(yTreino)),yTreinoPrevisto(:,j),'r');
    legend('Real','Previsto');
    xlabel('Amostras','FontSize',14);
    
    if(size(yTestePrevisto,2)==1)
        saida = strcat('y(k) ');
    else        
        auxr = num2str(j);
        saida = strcat('y_',auxr,'(k) ');
    end
    
    ylabel(saida,'FontSize',14);
    title('Treinamento','FontSize',14);
    subplot(3,1,2);
    plot(amostras(length(yTreino)+1:length([yTreino;yValidacao])),yValidacao(:,j),'b');
    hold on
    plot(amostras(length(yTreino)+1:length([yTreino;yValidacao])),yValPrevisto(:,j),'r');
    legend('Real','Previsto');
    xlabel('Amostras','FontSize',14);
    ylabel(saida,'FontSize',14);
    title('Validação','FontSize',14);
    subplot(3,1,3);
    plot(amostras(length([yTreino;yValidacao])+1:length([yTreino;yValidacao;yTeste])),yTeste(:,j),'b');
    hold on
    plot(amostras(length([yTreino;yValidacao])+1:length([yTreino;yValidacao;yTeste])),yTestePrevisto(:,j),'r');
    legend('Real','Previsto');
    xlabel('Amostras','FontSize',14);
    ylabel(saida,'FontSize',14);
    title('Teste','FontSize',14);
    filename = strcat('previsao_',saida,'_SRG');
    savefig(filename);
end

clear predictedTrainOutput predictedValOutput predictedTestOutput finalState...
      yTreinoPrevisto yValPrevisto yTestePrevisto filename saida auxr
  
plotregression(par_dist_SRG{posBestSRG}(1,:),par_dist_SRG{posBestSRG}(2,:));
subplot(1,1,1);
xlabel('Separação das entradas: ||u_i(t)-u_j(t)||');
ylabel('Separação dos estados:  ||x_i(t)-x_j(t)||');
legend('||x_i(t)-x_j(t)||=||u_i(t)-u_j(t)||','Regressão','Dados');
savefig('separation_ratio_graph_SRG');

% melhor ESN Random em relação ao teste
[bestErrorRandom,posBestRandom]=min(testErrorRandom);
bestEsnRandom=tr_esn_random{1,posBestRandom};
% Previsão do conjunto de treinamento
[predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, bestEsnRandom, nForgetPoints);
yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);
% Previsão do conjunto de validação 
[predictedValOutput,finalState] = test_esn_modificada(valInputSequence, bestEsnRandom, 0,'startingState',finalState);
yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
% Previsão do conjunto de teste
predictedTestOutput = test_esn_modificada(testInputSequence, bestEsnRandom, 0,'startingState',finalState);
yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);

for j = 1:size(yTestePrevisto,2)
    figure('Name','Previsão das variáveis de saída com a melhor ESN (teste) da Etapa 2 - Random','NumberTitle','on');
    amostras = [1:length([yTreino; yValidacao;yTeste])];
    subplot(3,1,1);
    plot(amostras((nForgetPoints+1):length(yTreino)),yTreino((nForgetPoints+1):end,j),'b');
    hold on
    plot(amostras((nForgetPoints+1):length(yTreino)),yTreinoPrevisto(:,j),'r');
    legend('Real','Previsto');
    xlabel('Amostras','FontSize',14);
    
    if(size(yTestePrevisto,2)==1)
        saida = strcat('y(k) ');
    else        
        auxr = num2str(j);
        saida = strcat('y_',auxr,'(k) ');
    end
    
    ylabel(saida,'FontSize',14);
    title('Treinamento','FontSize',14);
    subplot(3,1,2);
    plot(amostras(length(yTreino)+1:length([yTreino;yValidacao])),yValidacao(:,j),'b');
    hold on
    plot(amostras(length(yTreino)+1:length([yTreino;yValidacao])),yValPrevisto(:,j),'r');
    legend('Real','Previsto');
    xlabel('Amostras','FontSize',14);
    ylabel(saida,'FontSize',14);
    title('Validação','FontSize',14);
    subplot(3,1,3);
    plot(amostras(length([yTreino;yValidacao])+1:length([yTreino;yValidacao;yTeste])),yTeste(:,j),'b');
    hold on
    plot(amostras(length([yTreino;yValidacao])+1:length([yTreino;yValidacao;yTeste])),yTestePrevisto(:,j),'r');
    legend('Real','Previsto');
    xlabel('Amostras','FontSize',14);
    ylabel(saida,'FontSize',14);
    title('Teste','FontSize',14);
    filename = strcat('previsao_',saida,'_Random');
    savefig(filename);
end

clear predictedTrainOutput predictedValOutput predictedTestOutput finalState...
      yTreinoPrevisto yValPrevisto yTestePrevisto filename saida auxr

 for exp=1:numExp  
   
   % Período de washout
   stateCollectMat = compute_statematrix(trainInputSequence(1:nForgetPoints,:),trainOutputSequence(1:nForgetPoints,:),esn_random{exp},0);
   initialState =[stateCollectMat(end,:) trainOutputSequence(nForgetPoints,:)];     
   
   % Agrupamento usando os vizinhos mais próximos
   %[idx, dist] = knnsearch(trainInputSequence((nForgetPoints+1):end,:),trainInputSequence((nForgetPoints+1):end,:),'distance','euclidean','k',2); % Distancia
   pseudoPadrao = [trainInputSequence((nForgetPoints+1):end,:) trainOutputSequence(nForgetPoints:(end-1),:)];
   [idx, dist] = knnsearch(pseudoPadrao,pseudoPadrao,'distance','euclidean','k',2); % Distancia
   % Distância euclidiana dos padrões de entrada agrupados
   dist_entradas=dist(:,2);
   
   % Coleta dos estados gerados pelos padrões, desconsiderando o efeito de estados
   % anteriores e fazendo teach forcing na saída 
   estados = ...
    compute_statematrix_SRG_simplificada2(trainInputSequence((nForgetPoints+1):end,:), trainOutputSequence((nForgetPoints+1):end,:), esn_random{exp}, 0,initialState);
   
   % Distância dos estados agrupados pelo knn
   for j=1:size(idx,1) % Selecao de pares de estados para determinar a norma 
        primer_estado=estados(idx(j,1),:);  % Primeiro estado obtido
        segundo_estado=estados(idx(j,2),:); % Segundo estado obtido
        [idx_estados,dist_estados_parcial] = knnsearch(primer_estado,segundo_estado,'dist','euclidean','k',2); % obter a distancia euclidiana
        dist_estados(j,1)=dist_estados_parcial;
   end
   
   dist_entradas=dist_entradas';
   dist_estados=dist_estados';
   
   par_dist_bestRandom{exp} = [dist_entradas ; dist_estados]; % Armazena as distancias geradas para cada ESN

   clear primer_estado segundo_estado idx_estados dist_estados_parcial dist_estados estados ...
         idx dist esn dist_entradas

  for k=1:size(par_dist_bestRandom{exp},2) % Obtencao do erro da regressao linear
        erro_chvn(k)=abs(par_dist_bestRandom{exp}(1,k)-par_dist_bestRandom{exp}(2,k));
  end
   
  SRG_bestRandom(exp)=sum(erro_chvn);  
  
  clear erro_chvn SRG_rede distancias stateCollectMat initialState
 end
 
figure;
plotregression(par_dist_bestRandom{posBestRandom}(1,:),par_dist_bestRandom{posBestRandom}(2,:));
subplot(1,1,1);
xlabel('Separação das entradas: ||u_i(t)-u_j(t)||');
ylabel('Separação dos estados:  ||x_i(t)-x_j(t)||');
legend('||x_i(t)-x_j(t)||=||u_i(t)-u_j(t)||','Regressão','Dados');
savefig('separation_ratio_graph_Random');

% melhor ESN Pesos em relação ao teste
[bestErrorPesos,posBestPesos]=min(testErrorPesos);
bestEsnPesos=trESNP{posBestPesos};
% Previsão do conjunto de treinamento
[predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, bestEsnPesos, nForgetPoints);
yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);
% Previsão do conjunto de validação 
[predictedValOutput,finalState] = test_esn_modificada(valInputSequence, bestEsnPesos, 0,'startingState',finalState);
yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
% Previsão do conjunto de teste
predictedTestOutput = test_esn_modificada(testInputSequence, bestEsnPesos, 0,'startingState',finalState);
yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);

for j = 1:size(yTestePrevisto,2)
    figure('Name','Previsão das variáveis de saída com a melhor ESN (teste) da Etapa 2 - Otimização dos Pesos','NumberTitle','on');
    amostras = [1:length([yTreino; yValidacao;yTeste])];
    subplot(3,1,1);
    plot(amostras((nForgetPoints+1):length(yTreino)),yTreino((nForgetPoints+1):end,j),'b');
    hold on
    plot(amostras((nForgetPoints+1):length(yTreino)),yTreinoPrevisto(:,j),'r');
    legend('Real','Previsto');
    xlabel('Amostras','FontSize',14);
    
    if(size(yTestePrevisto,2)==1)
        saida = strcat('y(k) ');
    else        
        auxr = num2str(j);
        saida = strcat('y_',auxr,'(k) ');
    end
    
    ylabel(saida,'FontSize',14);
    title('Treinamento','FontSize',14);
    subplot(3,1,2);
    plot(amostras(length(yTreino)+1:length([yTreino;yValidacao])),yValidacao(:,j),'b');
    hold on
    plot(amostras(length(yTreino)+1:length([yTreino;yValidacao])),yValPrevisto(:,j),'r');
    legend('Real','Previsto');
    xlabel('Amostras','FontSize',14);
    ylabel(saida,'FontSize',14);
    title('Validação','FontSize',14);
    subplot(3,1,3);
    plot(amostras(length([yTreino;yValidacao])+1:length([yTreino;yValidacao;yTeste])),yTeste(:,j),'b');
    hold on
    plot(amostras(length([yTreino;yValidacao])+1:length([yTreino;yValidacao;yTeste])),yTestePrevisto(:,j),'r');
    legend('Real','Previsto');
    xlabel('Amostras','FontSize',14);
    ylabel(saida,'FontSize',14);
    title('Teste','FontSize',14);
    filename = strcat('previsao_',saida,'_Pesos');
    savefig(filename);
end
clear predictedTrainOutput predictedValOutput predictedTestOutput finalState...
      yTreinoPrevisto yValPrevisto yTestePrevisto filename saida auxr

 for exp=1:numExp  
     
   % Período de washout
   stateCollectMat = compute_statematrix(trainInputSequence(1:nForgetPoints,:),trainOutputSequence(1:nForgetPoints,:),esnP{exp},0);
   initialState =[stateCollectMat(end,:) trainOutputSequence(nForgetPoints,:)]; 
   
   % Agrupamento usando os vizinhos mais próximos
   %[idx, dist] = knnsearch(trainInputSequence((nForgetPoints+1):end,:),trainInputSequence((nForgetPoints+1):end,:),'distance','euclidean','k',2); % Distancia
   pseudoPadrao = [trainInputSequence((nForgetPoints+1):end,:) trainOutputSequence(nForgetPoints:(end-1),:)];
   [idx, dist] = knnsearch(pseudoPadrao,pseudoPadrao,'distance','euclidean','k',2); % Distancia
   % Distância euclidiana dos padrões de entrada agrupados
   dist_entradas=dist(:,2);
   
   % Coleta dos estados gerados pelos padrões, desconsiderando o efeito de estados
   % anteriores e fazendo teach forcing na saída 
   estados = ...
    compute_statematrix_SRG_simplificada2(trainInputSequence((nForgetPoints+1):end,:), trainOutputSequence((nForgetPoints+1):end,:), esnP{exp}, 0,initialState);
   
   % Distância dos estados agrupados pelo knn
   for j=1:size(idx,1) % Selecao de pares de estados para determinar a norma 
        primer_estado=estados(idx(j,1),:);  % Primeiro estado obtido
        segundo_estado=estados(idx(j,2),:); % Segundo estado obtido
        [idx_estados,dist_estados_parcial] = knnsearch(primer_estado,segundo_estado,'dist','euclidean','k',2); % obter a distancia euclidiana
        dist_estados(j,1)=dist_estados_parcial;
   end
   
   dist_entradas=dist_entradas';
   dist_estados=dist_estados';
   
   par_dist_bestPesos{exp} = [dist_entradas ; dist_estados]; % Armazena as distancias geradas para cada ESN

   clear primer_estado segundo_estado idx_estados dist_estados_parcial dist_estados estados ...
         idx dist esn dist_entradas

  for k=1:size(par_dist_bestPesos{exp},2) % Obtencao do erro da regressao linear
        erro_chvn(k)=abs(par_dist_bestPesos{exp}(1,k)-par_dist_bestPesos{exp}(2,k));
  end
   
  SRG_bestPesos(exp)=sum(erro_chvn);  
  
  clear erro_chvn SRG_rede distancias stateCollectMat initialState
 end
figure;
plotregression(par_dist_bestPesos{posBestPesos}(1,:),par_dist_bestPesos{posBestPesos}(2,:));
subplot(1,1,1);
xlabel('Separação das entradas: ||u_i(t)-u_j(t)||');
ylabel('Separação dos estados:  ||x_i(t)-x_j(t)||');
legend('||x_i(t)-x_j(t)||=||u_i(t)-u_j(t)||','Regressão','Dados');
savefig('separation_ratio_graph_Pesos');

% Impressão dos resultados
fprintf('\n\n             Etapa 1\n\n');
% Etapa 1
for exp =1:numExp
    
fprintf('Exp %d - Aptidao: %f\n',exp,bestFitness(exp));
    
end

fprintf('\nAptidão Média: %f\n',MeanFitness);
fprintf('\nAptidão Desvio: %f\n',StdFitness);
fprintf('\nTempo Total(%d experimentos): %f s\n',numExp,tempo1);

% Etapa 2 - Random
fprintf('\n\n             Etapa 2 - Random\n\n');
fprintf('                RMSE\n\n');
for exp =1:numExp
    
fprintf('Exp %d - Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',exp,trainErrorRandom(exp),...
    valErrorRandom(exp),testErrorRandom(exp),SRG_bestRandom(exp));
    
end

fprintf('\nMédia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanTrainErrorRandom,...
    meanValErrorRandom,meanTestErrorRandom);
fprintf('\nDesvio: Treinamento: %f  Validacao: %f  Teste: %f\n',stdTrainErrorRandom,...
    stdValErrorRandom,stdTestErrorRandom);
fprintf('\nMelhor: Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n\n',trainErrorRandom(posBestRandom),...
    valErrorRandom(posBestRandom),testErrorRandom(posBestRandom),SRG_bestRandom(posBestRandom));

fprintf('                fit\n\n');
for exp =1:numExp
    
fprintf('Exp %d - Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',exp,fitTrainErrorRandom(exp),...
    fitValErrorRandom(exp),fitTestErrorRandom(exp),SRG_bestRandom(exp));
    
end

fprintf('\nMédia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanfitTrainErrorRandom,...
    meanfitValErrorRandom,meanfitTestErrorRandom);
fprintf('\nDesvio: Treinamento: %f  Validacao: %f  Teste: %f\n',stdfitTrainErrorRandom,...
    stdfitValErrorRandom,stdfitTestErrorRandom);
fprintf('\nMelhor: Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',fitTrainErrorRandom(posBestRandom),...
    fitValErrorRandom(posBestRandom),fitTestErrorRandom(posBestRandom),SRG_bestRandom(posBestRandom));

fprintf('                RRSE\n\n');
for exp =1:numExp
    
fprintf('Exp %d - Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',exp,RRSETrainErrorRandom(exp),...
    RRSEValErrorRandom(exp),RRSETestErrorRandom(exp),SRG_bestRandom(exp));
    
end

fprintf('\nMédia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanRRSETrainErrorRandom,...
    meanRRSEValErrorRandom,meanRRSETestErrorRandom);
fprintf('\nDesvio: Treinamento: %f  Validacao: %f  Teste: %f\n',stdRRSETrainErrorRandom,...
    stdRRSEValErrorRandom,stdRRSETestErrorRandom);
fprintf('\nMelhor: Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',RRSETrainErrorRandom(posBestRandom),...
    RRSEValErrorRandom(posBestRandom),RRSETestErrorRandom(posBestRandom),SRG_bestRandom(posBestRandom));

% fprintf('                MSE\n\n');
% for exp =1:numExp
%     
% fprintf('Exp %d - Teste: y1 - %f y2 - %f  \n',exp,mse_test_error_y1_random(exp),mse_test_error_y2_random(exp));
%     
% end
% 
% fprintf('\nMédia: Teste: y1 - %f y2 - %f\n',mean_mse_test_error_y1_random, mean_mse_test_error_y2_random);
% fprintf('\nDesvio:Teste: y1 - %f y2 - %f\n',std_mse_test_error_y1_random,  std_mse_test_error_y2_random);

fprintf('                RRSE\n\n');
for exp =1:numExp
    
fprintf('Exp %d - Teste: y1 - %f y2 - %f  \n',exp,RRSE_test_error_y1_random(exp),RRSE_test_error_y2_random(exp));
    
end

fprintf('\nMédia: Teste: y1 - %f y2 - %f\n',mean_RRSE_test_error_y1_random, mean_RRSE_test_error_y2_random);
fprintf('\nDesvio:Teste: y1 - %f y2 - %f\n',std_RRSE_test_error_y1_random,  std_RRSE_test_error_y2_random);

fprintf('\nTempo Total(%d experimentos): %f s\n',numExp,tempoTotalRandom);

% Etapa 2 - SRG
fprintf('\n\n             Etapa 2 - SRG\n\n');
fprintf('                RMSE\n\n');
for exp =1:numExp
    
fprintf('Exp %d - Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',exp,trainErrorSRG(exp),...
    valErrorSRG(exp),testErrorSRG(exp),SRG{exp});
    
end

fprintf('\nMédia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanTrainErrorSRG,...
    meanValErrorSRG,meanTestErrorSRG);
fprintf('\nDesvio: Treinamento: %f  Validacao: %f  Teste: %f\n',stdTrainErrorSRG,...
    stdValErrorSRG,stdTestErrorSRG);
fprintf('\nMelhor: Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n\n',trainErrorSRG(posBestSRG),...
    valErrorSRG(posBestSRG),testErrorSRG(posBestSRG),SRG{posBestSRG});


fprintf('                fit\n\n');
for exp =1:numExp
    
fprintf('Exp %d - Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',exp,fitTrainErrorSRG(exp),...
    fitValErrorSRG(exp),fitTestErrorSRG(exp),SRG{exp});
    
end

fprintf('\nMédia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanfitTrainErrorSRG,...
    meanfitValErrorSRG,meanfitTestErrorSRG);
fprintf('\nDesvio: Treinamento: %f  Validacao: %f  Teste: %f\n',stdfitTrainErrorSRG,...
    stdfitValErrorSRG,stdfitTestErrorSRG);
fprintf('\nMelhor: Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',fitTrainErrorSRG(posBestSRG),...
    fitValErrorSRG(posBestSRG),fitTestErrorSRG(posBestSRG),SRG{posBestSRG});

fprintf('                RRSE\n\n');
for exp =1:numExp
    
fprintf('Exp %d - Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',exp,RRSETrainErrorSRG(exp),...
    RRSEValErrorSRG(exp),RRSETestErrorSRG(exp),SRG{exp});
    
end

fprintf('\nMédia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanRRSETrainErrorSRG,...
    meanRRSEValErrorSRG,meanRRSETestErrorSRG);
fprintf('\nDesvio: Treinamento: %f  Validacao: %f  Teste: %f\n',stdRRSETrainErrorSRG,...
    stdRRSEValErrorSRG,stdRRSETestErrorSRG);
fprintf('\nMelhor: Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',RRSETrainErrorSRG(posBestSRG),...
    RRSEValErrorSRG(posBestSRG),RRSETestErrorSRG(posBestSRG),SRG{posBestSRG});

% fprintf('                MSE\n\n');
% for exp =1:numExp
%     
% fprintf('Exp %d - Teste: y1 - %f y2 - %f  \n',exp,mse_test_error_y1_SRG(exp),mse_test_error_y2_SRG(exp));
%     
% end

% fprintf('\nMédia: Teste: y1 - %f y2 - %f\n',mean_mse_test_error_y1_SRG, mean_mse_test_error_y2_SRG);
% fprintf('\nDesvio:Teste: y1 - %f y2 - %f\n',std_mse_test_error_y1_SRG,  std_mse_test_error_y2_SRG);

fprintf('                RRSE\n\n');
for exp =1:numExp
    
fprintf('Exp %d - Teste: y1 - %f y2 - %f  \n',exp,RRSE_test_error_y1_SRG(exp),RRSE_test_error_y2_SRG(exp));
    
end

fprintf('\nMédia: Teste: y1 - %f y2 - %f\n',mean_RRSE_test_error_y1_SRG, mean_RRSE_test_error_y2_SRG);
fprintf('\nDesvio:Teste: y1 - %f y2 - %f\n',std_RRSE_test_error_y1_SRG, std_RRSE_test_error_y2_SRG);
fprintf('\nTempo Total(%d experimentos): %f s\n',numExp,tempoTotalSRG);

% Etapa 2 - Otimização dos pesos do reservatório
fprintf('\n\n             Etapa 2 - Otimização dos pesos do reservatório\n\n');
fprintf('                RMSE\n\n');
for exp =1:numExp
    
fprintf('Exp %d - Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',exp,trainErrorPesos(exp),...
    valErrorPesos(exp),testErrorPesos(exp),SRG_bestPesos(exp));
    
end

fprintf('\nMédia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanTrainErrorPesos,...
    meanValErrorPesos,meanTestErrorPesos);
fprintf('\nDesvio: Treinamento: %f  Validacao: %f  Teste: %f\n',stdTrainErrorPesos,...
    stdValErrorPesos,stdTestErrorPesos);
fprintf('\nMelhor: Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n\n',trainErrorPesos(posBestPesos),...
    valErrorPesos(posBestPesos),testErrorPesos(posBestPesos),SRG_bestPesos(posBestPesos));

fprintf('                fit\n\n');
for exp =1:numExp
    
fprintf('Exp %d - Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',exp,fitTrainErrorPesos(exp),...
    fitValErrorPesos(exp),fitTestErrorPesos(exp),SRG_bestPesos(exp));
    
end

fprintf('\nMédia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanfitTrainErrorPesos,...
    meanfitValErrorPesos,meanfitTestErrorPesos);
fprintf('\nDesvio: Treinamento: %f  Validacao: %f  Teste: %f\n',stdfitTrainErrorPesos,...
    stdfitValErrorPesos,stdfitTestErrorPesos);
fprintf('\nMelhor: Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',fitTrainErrorPesos(posBestPesos),...
    fitValErrorPesos(posBestPesos),fitTestErrorPesos(posBestPesos),SRG_bestPesos(posBestPesos));

fprintf('                RRSE\n\n');
for exp =1:numExp
    
fprintf('Exp %d - Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',exp,RRSETrainErrorPesos(exp),...
    RRSEValErrorPesos(exp),RRSETestErrorPesos(exp),SRG_bestPesos(exp));
    
end

fprintf('\nMédia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanRRSETrainErrorPesos,...
    meanRRSEValErrorPesos,meanRRSETestErrorPesos);
fprintf('\nDesvio: Treinamento: %f  Validacao: %f  Teste: %f\n',stdRRSETrainErrorPesos,...
    stdRRSEValErrorPesos,stdRRSETestErrorPesos);
fprintf('\nMelhor: Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',RRSETrainErrorPesos(posBestPesos),...
    RRSEValErrorPesos(posBestPesos),RRSETestErrorPesos(posBestPesos),SRG_bestPesos(posBestPesos));

fprintf('\nTempo Total(%d experimentos): %f s\n',numExp,tempoTotalPesos);

% fprintf('                MSE\n\n');
% for exp =1:numExp
%     
% fprintf('Exp %d - Teste: y1 - %f y2 - %f  \n',exp,mse_test_error_y1_pesos(exp),mse_test_error_y2_pesos(exp));
%     
% end

% fprintf('\nMédia: Teste: y1 - %f y2 - %f\n',mean_mse_test_error_y1_pesos, mean_mse_test_error_y2_pesos);
% fprintf('\nDesvio:Teste: y1 - %f y2 - %f\n',std_mse_test_error_y1_pesos,  std_mse_test_error_y2_pesos);

fprintf('                RRSE\n\n');
for exp =1:numExp
    
fprintf('Exp %d - Teste: y1 - %f y2 - %f  \n',exp,RRSE_test_error_y1_pesos(exp),RRSE_test_error_y2_pesos(exp));
    
end

fprintf('\nMédia: Teste: y1 - %f y2 - %f\n',mean_RRSE_test_error_y1_pesos, mean_RRSE_test_error_y2_pesos);
fprintf('\nDesvio:Teste: y1 - %f y2 - %f\n',std_RRSE_test_error_y1_pesos, std_RRSE_test_error_y2_pesos);

fprintf('\nTempo Total(%d experimentos): %f s\n',numExp,tempoTotalPesos);


save (nomeTeste)

diary off
 
