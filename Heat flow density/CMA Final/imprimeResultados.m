clc
clear all
close all

load resultadosOtimizacaoCMA.mat

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


