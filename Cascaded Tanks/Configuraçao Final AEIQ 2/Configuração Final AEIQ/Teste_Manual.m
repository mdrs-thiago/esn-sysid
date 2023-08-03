clear all
close all
clc

%load aeiq200000 % carrega resultados da otimização da etapa 1 por AEIQ
numExp=1;
numRedes=1;
metricaErro = 'RMSE';
load tankNorm.mat;
nForgetPoints= 30;
% %Geração das numRedes para os numExp realizados
tic
for exp =1:numExp
  for i=1:numRedes     
      
     reservatorio = 50;  % número de neurônios no reservatório

     raio = 0.7;          % raio espectral
 
     ScalWin = 0.5;       % constante de multiplicação da matriz Win
 
     ScalWback = 0.5;     % constante de multiplicação da matriz Wback
 
     vazamento = 0.9;     % leaky rate dos neurônios leaky integrator

     conectividade = 0.9; % percentual de conectividade dos neurônios do reservatório
     
          
     nInputUnits =  size(trainInputSequence,2); nInternalUnits = reservatorio; nOutputUnits = size(trainOutputSequence,2);
     % Gerando uma ESN
     esn_totais{exp,i} = generate_esn2(nInputUnits, nInternalUnits, nOutputUnits, ScalWin, ScalWback, conectividade, ...
      'spectralRadius',raio,'inputScaling',1* ones(nInputUnits,1),'inputShift',0*ones(nInputUnits,1), ...
      'teacherScaling',1*ones(nOutputUnits,1),'teacherShift',0*ones(nOutputUnits,1),'feedbackScaling', 1, ...
      'type', 'leaky_esn','leakage',vazamento,...
      'reservoirActivationFunction','tanh',...
      'outputActivationFunction', 'identity','inverseOutputActivationFunction','identity',...
      'methodWeightCompute','pseudoinverse','timeConstants',1 * ones(nInternalUnits,1));  
    
      esn_totais{exp,i}.internalWeights = esn_totais{exp,i}.spectralRadius * esn_totais{exp,i}.internalWeights_UnitSR;
   
  end
end
tgen = toc;

% save ('esn_totais.mat','esn_totais'); 
% save ('tgen.mat','tgen');
% load esn_totais.mat
% load tgen.mat
% Etapa 2 - SRG
tic
for exp = 1:numExp
  for i=1:numRedes  
   % Agrupamento usando os vizinhos mais próximos
   [idx, dist] = knnsearch(trainInputSequence((nForgetPoints+1):end,:),trainInputSequence((nForgetPoints+1):end,:),'distance','euclidean','k',2); % Distancia
   % Distância euclidiana dos padrões de entrada agrupados
   dist_entradas=dist(:,2);
   
   % Coleta dos estados gerados pelos padrões, desconsiderando o efeito de estados
   % anteriores e fazendo teach forcing na saída 
   estados = ...
    compute_statematrix_SRG_simplificada(trainInputSequence((nForgetPoints+1):end,:), trainOutputSequence((nForgetPoints+1):end,:), esn_totais{exp,i}, 0);
   
   % Distância dos estados agrupados pelo knn
   for j=1:size(idx,1) % Selecao de pares de estados para determinar a norma 
        primer_estado=estados(idx(j,1),:);  % Primeiro estado obtido
        segundo_estado=estados(idx(j,2),:); % Segundo estado obtido
        [idx_estados,dist_estados_parcial] = knnsearch(primer_estado,segundo_estado,'dist','euclidean','k',2); % obter a distancia euclidiana
        dist_estados(j,1)=dist_estados_parcial;
   end
   
   dist_entradas=dist_entradas';
   dist_estados=dist_estados';
   
   distancias {i} = [dist_entradas ; dist_estados]; % Armazena as distancias geradas para cada ESN

   clear primer_estado segundo_estado idx_estados dist_estados_parcial dist_estados estados ...
         idx dist esn dist_entradas

  end


  for j =1:size(distancias,2)
%     figure;
     par_dist = distancias{1,j}; % par de distancias para cada rede
    
    for k=1:size(par_dist,2) % Obtencao do erro da regressao linear
        erro_chvn(k)=abs(par_dist(1,k)-par_dist(2,k));
    end
   
    SRG_rede(j)=sum(erro_chvn);
    
%     for p=1:size(par_dist,2) % Indice de Separacao
%         sep(p)=par_dist(2,p)./par_dist(1,p);
%     end
%     coef(j,6)=mean(sep); % Armazenamento do indice de separacao, tem que ser o mais perto de 1
%     
%     for t=1:size(par_dist,2)
%         liap(t)=log(par_dist(2,t)./par_dist(1,t));
%     end
%     coef(j,7)=sum(liap)*raio; % Coeficiente do Liapunov, Liapunov positivo --> caotico, Lyapunov negativo --> atrator.
%     
%     coef(j,8)=norm(abs(par_dist(1,:)-par_dist(2,:))); % Norma da diferenca da separacao da entrada e a saida.
%     coef(j,9)=100*mean(abs(par_dist(1,:)-par_dist(2,:))./(((par_dist(1,:))+(par_dist(2,:)))/2)); %sMAPE da diferenca da separacao da entrada e a saida
    
    clear par_dist  
   
  end
  
  % Escolha da rede com menor SRG
  [minSRG,posMinSRG]=min(SRG_rede);
  esn_SRG{exp}=esn_totais{exp,posMinSRG};
  par_dist_SRG{exp} = distancias{1,posMinSRG}; % par de distancias para cada rede
  SRG{exp}=minSRG;
%   figure;
%   plotregression(par_dist_SRG{exp}(1,:),par_dist_SRG{exp}(2,:));
%   subplot(1,1,1);
%   xlabel('Separação das entradas: ||u_i(t)-u_j(t)||');
%   ylabel('Separação dos estados:  ||x_i(t)-x_j(t)||');
%   legend('||x_i(t)-x_j(t)||=||u_i(t)-u_j(t)||','Regressão','Dados');
  clear erro_chvn SRG_rede minSRG distancias
  % Treinamento da rede com o menor SRG 
  tr_esn_SRG{exp} = ...
  train_esn(trainInputSequence, trainOutputSequence, esn_SRG{exp}, nForgetPoints) ; 
  
  % Previsão do conjunto de treinamento
  [predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, tr_esn_SRG{exp}, nForgetPoints);
  yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);
  % Cálculo do RMSE no conjunto de treinamento
  trainErrorSRG(exp) = feval(metricaErro,yTreinoPrevisto, yTreino);
  fitTrainErrorSRG(exp) = feval('fit',yTreinoPrevisto, yTreino);
 
  % Previsão do conjunto de validação 
  [predictedValOutput,finalState] = test_esn_modificada(valInputSequence, tr_esn_SRG{exp}, 0,'startingState',finalState);
   yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
  % Cálculo do RMSE no conjunto de validação
  valErrorSRG(exp) = feval(metricaErro,yValPrevisto, yValidacao);
  fitValErrorSRG(exp) = feval('fit',yValPrevisto, yValidacao);
  
  % Previsão do conjunto de teste
  predictedTestOutput = test_esn_modificada(testInputSequence, tr_esn_SRG{exp}, 0,'startingState',finalState);
  %predictedTestOutput = test_esn_modificada(testInputSequence, best_ESN, 0);
  yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);
  %Cálculo do RMSE no conjunto de teste
  testErrorSRG(exp) = feval(metricaErro,yTestePrevisto, yTeste);
  fitTestErrorSRG(exp) = feval('fit',yTestePrevisto, yTeste);  
 
end
tSRGParcial = toc;
% Fim SRG
tic

% Etapa 2 - Random
for exp =1:numExp
  for i=1:numRedes     
           
     tr_esn_totais{exp,i}= train_esn(trainInputSequence, trainOutputSequence, esn_totais{exp,i}, nForgetPoints) ; 
  
     % Previsão do conjunto de treinamento
     [predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, tr_esn_totais{exp,i}, nForgetPoints);
     yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);

     % Previsão do conjunto de validação 
     [predictedValOutput,finalState] = test_esn_modificada(valInputSequence, tr_esn_totais{exp,i}, 0,'startingState',finalState);
     yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
     % Cálculo do RMSE no conjunto de validação
     valErrorR(i) = feval(metricaErro,yValPrevisto, yValidacao);
  
  end
  [minValErrorRandom,posMinVal]=min(valErrorR);
  tr_esn_random{exp}=tr_esn_totais{exp,posMinVal};
  esn_random{exp}=esn_totais{exp,posMinVal};
  % Previsão do conjunto de treinamento
  [predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, tr_esn_random{exp}, nForgetPoints);
  yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);
  % Cálculo do RMSE no conjunto de treinamento
  trainErrorRandom(exp) = feval(metricaErro,yTreinoPrevisto, yTreino);
  fitTrainErrorRandom(exp) = feval('fit',yTreinoPrevisto, yTreino);
  % Previsão do conjunto de validação 
  [predictedValOutput,finalState] = test_esn_modificada(valInputSequence, tr_esn_random{exp}, 0,'startingState',finalState);
  yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
  % Cálculo do RMSE no conjunto de validação
  valErrorRandom(exp) = feval(metricaErro,yValPrevisto, yValidacao);
  fitValErrorRandom(exp) = feval('fit',yValPrevisto, yValidacao);
  % Previsão do conjunto de teste
  predictedTestOutput = test_esn_modificada(testInputSequence, tr_esn_random{exp}, 0,'startingState',finalState);

  yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);
  % Cálculo do RMSE no conjunto de teste
  testErrorRandom(exp) = feval(metricaErro,yTestePrevisto, yTeste);
  fitTestErrorRandom(exp) = feval('fit',yTestePrevisto, yTeste);

  
end
tRandomParcial = toc;
% Fim Random
tempoTotalRandom = tRandomParcial + tgen;
tempoTotalSRG = tSRGParcial + tgen;
tempoMedioRandom = tempoTotalRandom./numExp;
tempoMedioSRG = tempoTotalSRG./numExp;

meanTrainErrorSRG = mean(trainErrorSRG);
meanValErrorSRG = mean(valErrorSRG);
meanTestErrorSRG = mean(testErrorSRG);
stdTrainErrorSRG = std(trainErrorSRG);
stdValErrorSRG = std(valErrorSRG);
stdTestErrorSRG = std(testErrorSRG);
meanfitTrainErrorSRG = mean(fitTrainErrorSRG);
meanfitValErrorSRG = mean(fitValErrorSRG);
meanfitTestErrorSRG = mean(fitTestErrorSRG);
stdfitTrainErrorSRG = std(fitTrainErrorSRG);
stdfitValErrorSRG = std(fitValErrorSRG);
stdfitTestErrorSRG = std(fitTestErrorSRG);

meanTrainErrorRandom = mean(trainErrorRandom)
meanValErrorRandom = mean(valErrorRandom)
meanTestErrorRandom = mean(testErrorRandom)
stdTrainErrorRandom = std(trainErrorRandom);
stdValErrorRandom = std(valErrorRandom);
stdTestErrorRandom = std(testErrorRandom);
meanfitTrainErrorRandom = mean(fitTrainErrorRandom);
meanfitValErrorRandom = mean(fitValErrorRandom);
meanfitTestErrorRandom = mean(fitTestErrorRandom);
stdfitTrainErrorRandom = std(fitTrainErrorRandom);
stdfitValErrorRandom = std(fitValErrorRandom);
stdfitTestErrorRandom = std(fitTestErrorRandom);

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
    figure;  
    amostras = [1:length([yTreino; yValidacao;yTeste])];
    subplot(3,1,1);
    plot(amostras((nForgetPoints+1):length(yTreino)),yTreino((nForgetPoints+1):end,j),'b');
    hold on
    plot(amostras((nForgetPoints+1):length(yTreino)),yTreinoPrevisto(:,j),'r');
    legend('Real','Previsto');
    xlabel('Amostras','FontSize',14);
    auxr = num2str(j);
    saida = strcat('y_',auxr,'(k) ');
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
end
    figure;
    plotregression(par_dist_SRG{posBestSRG}(1,:),par_dist_SRG{posBestSRG}(2,:));
    subplot(1,1,1);
    xlabel('Separação das entradas: ||u_i(t)-u_j(t)||');
    ylabel('Separação dos estados:  ||x_i(t)-x_j(t)||');
    legend('||x_i(t)-x_j(t)||=||u_i(t)-u_j(t)||','Regressão','Dados');

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
    figure;  
    amostras = [1:length([yTreino; yValidacao;yTeste])];
    subplot(3,1,1);
    plot(amostras((nForgetPoints+1):length(yTreino)),yTreino((nForgetPoints+1):end,j),'b');
    hold on
    plot(amostras((nForgetPoints+1):length(yTreino)),yTreinoPrevisto(:,j),'r');
    legend('Real','Previsto');
    xlabel('Amostras','FontSize',14);
    auxr = num2str(j);
    saida = strcat('y_',auxr,'(k) ');
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
end

 for exp=1:numExp  
   % Agrupamento usando os vizinhos mais próximos
   [idx, dist] = knnsearch(trainInputSequence((nForgetPoints+1):end,:),trainInputSequence((nForgetPoints+1):end,:),'distance','euclidean','k',2); % Distancia
   % Distância euclidiana dos padrões de entrada agrupados
   dist_entradas=dist(:,2);
   
   % Coleta dos estados gerados pelos padrões, desconsiderando o efeito de estados
   % anteriores e fazendo teach forcing na saída 
   estados = ...
    compute_statematrix_SRG_simplificada(trainInputSequence((nForgetPoints+1):end,:), trainOutputSequence((nForgetPoints+1):end,:), esn_random{exp}, 0);
   
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
  
  clear erro_chvn SRG_rede distancias
 end
figure;
plotregression(par_dist_bestRandom{posBestRandom}(1,:),par_dist_bestRandom{posBestRandom}(2,:));
subplot(1,1,1);
xlabel('Separação das entradas: ||u_i(t)-u_j(t)||');
ylabel('Separação dos estados:  ||x_i(t)-x_j(t)||');
legend('||x_i(t)-x_j(t)||=||u_i(t)-u_j(t)||','Regressão','Dados');
