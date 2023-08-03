%%TESTE de Otimiza��o com Matlab

clear all
close all
clc

%% Load de dados
load robotNorm.mat
nomeTeste ='resultadosOtimizacaoAEIQ';

%% Configura��es gerais

numExp = 10;           % n�mero de experimentos 

nForgetPoints = 30;  % T0 - n�mero de dados descartados durante o per�odo de washout da ESN no treinamento

numRedes = 10;         % n�mero de redes avaliadas por cada indiv�duo na etapa 1 de otimiza��o de par�metros da ESN

metricaErro = 'RMSE'; % metrica de erro a ser utilizada na fun��o de aptid�o

% 
% T = size(trainInputSequence,1);  % tamanho da base de treinamento


%% Configura��es AEIQ (etapa 1)

numPulses = 20;     % popula��o qu�ntica
numClassic = 60;   % popula��o cl�ssica
genToWidth = 5;     % intervalo de gera��es para atualiza��o da popula��o qu�ntica
generations = 100;  % n�mero de gera��es
crossover = 0.6;    % taxa de crossover cl�ssico real
qCrossover = 0.2;   % taxa de crossover qu�ntico
showEv = 1;
gap = numClassic;          
seed = rng;
xanterior = [];

fun = 'calculaErroFinal'; % nome da fun��o de aptid�o

% limInf = [floor((T-nForgetPoints)/10); 0.10; 0.0001; 0; 0.1;0.1]; % limites inferiores das vari�veis
% 
% limSup = [floor((T-nForgetPoints)/2); 0.99; 1; 1; 1; 1];   % limites superiores das vari�veis

limInf = [20; 0.10; 0.0001; 0; 0.1;0.1]; % limites inferiores das vari�veis
 
limSup = [150; 0.99; 1; 1; 1; 1];   % limites superiores das vari�veis

num_var = 6; % n�mero de vari�veis a serem otimizadas na etapa 1 de otimiza��o de par�metros da ESN
% tamanho do reservat�rio
% raio espectral
% fator de escala de Win
% fator de escala de Wback
% leaky rate
% percentual de conectividade do reservat�rio

%% Configura��es AEIQ (etapa 2)

numPulses2 = 10;     % popula��o qu�ntica
numClassic2 = 40;    % popula��o cl�ssica
genToWidth2 = 5;    % intervalo de gera��es para atualiza��o da popula��o qu�ntica
generations2 = 100;   % n�mero de gera��es
crossover2 = 0.3;  % taxa de crossover cl�ssico real
qCrossover2 = 0.8; % taxa de crossover qu�ntico
gap2 = numClassic2;    

fun2 = 'calculaErroPesos'; % nome da fun��o de aptid�o

%% Etapa 1 - Otimiza��o dos par�metros da ESN
tic
diary ('aeiq.txt');

poolobj = gcp('nocreate'); % If no pool, do not create new one.
if isempty(poolobj)
    parpool;
end

aux =0;
for exp = 1:numExp

[sol{exp}, classic,  evobestFitness{exp}, trace] = aeiq_br_real(num_var, numPulses, numClassic, ...
     genToWidth, generations, crossover, qCrossover, showEv,  gap, exp,seed,fun,limInf,limSup,trainInputSequence,trainOutputSequence,valInputSequence,yValidacao,yMin,yMax,...
    nForgetPoints,numRedes,metricaErro);

aux = aux+evobestFitness{exp};

bestFitness(exp) = min(evobestFitness{exp});
 
end
tempo1 = toc;


MeanFitness = mean(bestFitness);
StdFitness = std(bestFitness);
meanEvoBestFitness = aux./numExp;

figure('Name','Curva de Evolu��o - Etapa 1 AEIQ','NumberTitle','off');
plot(meanEvoBestFitness);
title('Curva de Evolu��o');
xlabel('Gera��es');
ylabel('Aptid�o');
savefig('Curva_Evolucao_Etapa_1');

clear aux classic trace

%% Gera��o das numRedes para os numExp experimentos realizados com as configura��es da etapa 1
tic
for exp =1:numExp
  for i=1:numRedes     
      
     reservatorio = ceil(sol{exp}(1));  % n�mero de neur�nios no reservat�rio

     raio = sol{exp}(2);          % raio espectral
 
     ScalWin = sol{exp}(3);       % constante de multiplica��o da matriz Win
 
     ScalWback = sol{exp}(4);     % constante de multiplica��o da matriz Wback
 
     vazamento = sol{exp}(5);     % leaky rate dos neur�nios leaky integrator

     conectividade = sol{exp}(6); % percentual de conectividade dos neur�nios do reservat�rio
               
     nInputUnits =  size(trainInputSequence,2); nInternalUnits = reservatorio; nOutputUnits = size(trainOutputSequence,2);
     % Gerando uma ESN
     esn_totais{exp,i} = generate_esn2(nInputUnits, nInternalUnits, nOutputUnits, ScalWin, ScalWback, conectividade, ...
      'spectralRadius',raio,'inputScaling',1* ones(nInputUnits,1),'inputShift',0*ones(nInputUnits,1), ...
      'teacherScaling',1*ones(nOutputUnits,1),'teacherShift',0*ones(nOutputUnits,1),'feedbackScaling', 1, ...
      'type', 'leaky_esn','leakage',vazamento,...
      'reservoirActivationFunction','tanh',...
      'outputActivationFunction', 'identity','inverseOutputActivationFunction','identity',...
      'methodWeightCompute','pseudoinverse', 'timeConstants',1 * ones(nInternalUnits,1));  
    
      esn_totais{exp,i}.internalWeights = esn_totais{exp,i}.spectralRadius * esn_totais{exp,i}.internalWeights_UnitSR;
   
  end
end
tgen = toc;

clear nInputUnits nInternalUnits nOutputUnits reservatorio raio ScalWin ScalWback vazamento conectividade

%% Etapa 2 - SRG
tic
for exp = 1:numExp
  for i=1:numRedes
   
    % Per�odo de washout
   stateCollectMat = compute_statematrix(trainInputSequence(1:nForgetPoints,:),trainOutputSequence(1:nForgetPoints,:),esn_totais{exp,i},0);
   initialState =[stateCollectMat(end,:) trainOutputSequence(nForgetPoints,:)];
      
   % Agrupamento usando os vizinhos mais pr�ximos
   %[idx, dist] = knnsearch(trainInputSequence((nForgetPoints+1):end,:),trainInputSequence((nForgetPoints+1):end,:),'distance','euclidean','k',2); % Distancia
   pseudoPadrao = [trainInputSequence((nForgetPoints+1):end,:) trainOutputSequence(nForgetPoints:(end-1),:)];
   [idx, dist] = knnsearch(pseudoPadrao,pseudoPadrao,'distance','euclidean','k',2); % Distancia
   % Dist�ncia euclidiana dos padr�es de entrada agrupados
   dist_entradas=dist(:,2);
   
   % Coleta dos estados gerados pelos padr�es, desconsiderando o efeito de estados
   % anteriores e fazendo teach forcing na sa�da 
%    estados = ...
%     compute_statematrix_SRG_simplificada(trainInputSequence((nForgetPoints+1):end,:), trainOutputSequence((nForgetPoints+1):end,:), esn_totais{exp,i}, 0);
   estados = ...
    compute_statematrix_SRG_simplificada2(trainInputSequence((nForgetPoints+1):end,:), trainOutputSequence((nForgetPoints+1):end,:), esn_totais{exp,i}, 0,initialState);  

   % Dist�ncia dos estados agrupados pelo knn
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
         idx dist esn dist_entradas initialState stateCollectMat

  end


  for j =1:size(distancias,2)

     par_dist = distancias{1,j}; % par de distancias para cada rede
    
    for k=1:size(par_dist,2) % Obtencao do erro da regressao linear
        erro_chvn(k)=abs(par_dist(1,k)-par_dist(2,k));
    end
   
    SRG_rede(j)=sum(erro_chvn);
    
    
    clear par_dist 
   
  end
  
  % Escolha da rede com menor SRG
  [minSRG,posMinSRG]=min(SRG_rede);
  esn_SRG{exp}=esn_totais{exp,posMinSRG};
  par_dist_SRG{exp} = distancias{1,posMinSRG}; % par de distancias para cada rede
  SRG{exp}=minSRG;

  clear erro_chvn SRG_rede minSRG distancias
  % Treinamento da rede com o menor SRG 
  tr_esn_SRG{exp} = ...
  train_esn(trainInputSequence, trainOutputSequence, esn_SRG{exp}, nForgetPoints) ; 
  
  % Previs�o do conjunto de treinamento
  [predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, tr_esn_SRG{exp}, nForgetPoints);
  yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);
  % C�lculo do RMSE no conjunto de treinamento
  trainErrorSRG(exp) = feval(metricaErro,yTreinoPrevisto, yTreino);
  RRSETrainErrorSRG(exp) = feval('RRSE',yTreinoPrevisto, yTreino);
  fitTrainErrorSRG(exp) = feval('fit',yTreinoPrevisto, yTreino);
 
  % Previs�o do conjunto de valida��o 
  [predictedValOutput,finalState] = test_esn_modificada(valInputSequence, tr_esn_SRG{exp}, 0,'startingState',finalState);
   yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
  % C�lculo do RMSE no conjunto de valida��o
  valErrorSRG(exp) = feval(metricaErro,yValPrevisto, yValidacao);
  RRSEValErrorSRG(exp) = feval('RRSE',yValPrevisto, yValidacao);
  fitValErrorSRG(exp) = feval('fit',yValPrevisto, yValidacao);
  
  % Previs�o do conjunto de teste
  predictedTestOutput = test_esn_modificada(testInputSequence, tr_esn_SRG{exp}, 0,'startingState',finalState);
  %predictedTestOutput = test_esn_modificada(testInputSequence, best_ESN, 0);
  yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);
  %C�lculo do RMSE no conjunto de teste
  testErrorSRG(exp) = feval(metricaErro,yTestePrevisto, yTeste);
  RRSETestErrorSRG(exp) = feval('RRSE',yTestePrevisto, yTeste);
  fitTestErrorSRG(exp) = feval('fit',yTestePrevisto, yTeste);
  
  
 
end
tSRGParcial = toc;
% Fim SRG

clear predictedTrainOutput predictedValOutput predictedTestOutput finalState...
      yTreinoPrevisto yValPrevisto yTestePrevisto minSRG posMinSRG

%% Etapa 2 - Random
tic
for exp =1:numExp
  for i=1:numRedes     
           
     tr_esn_totais{exp,i}= train_esn(trainInputSequence, trainOutputSequence, esn_totais{exp,i}, nForgetPoints) ; 
  
     % Previs�o do conjunto de treinamento
     [predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, tr_esn_totais{exp,i}, nForgetPoints);
     yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);

     % Previs�o do conjunto de valida��o 
     [predictedValOutput,finalState] = test_esn_modificada(valInputSequence, tr_esn_totais{exp,i}, 0,'startingState',finalState);
     yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
     % C�lculo do RMSE no conjunto de valida��o
     valErrorR(i) = feval(metricaErro,yValPrevisto, yValidacao);
  
  end
  [minValErrorRandom,posMinVal]=min(valErrorR);
  tr_esn_random{exp}=tr_esn_totais{exp,posMinVal};
  esn_random{exp}=esn_totais{exp,posMinVal};
  % Previs�o do conjunto de treinamento
  [predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, tr_esn_random{exp}, nForgetPoints);
  yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);
  % C�lculo do RMSE no conjunto de treinamento
  trainErrorRandom(exp) = feval(metricaErro,yTreinoPrevisto, yTreino);
  RRSETrainErrorRandom(exp) = feval('RRSE',yTreinoPrevisto, yTreino);
  fitTrainErrorRandom(exp) = feval('fit',yTreinoPrevisto, yTreino);
  % Previs�o do conjunto de valida��o 
  [predictedValOutput,finalState] = test_esn_modificada(valInputSequence, tr_esn_random{exp}, 0,'startingState',finalState);
  yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
  % C�lculo do RMSE no conjunto de valida��o
  valErrorRandom(exp) = feval(metricaErro,yValPrevisto, yValidacao);
  RRSEValErrorRandom(exp) = feval('RRSE',yValPrevisto, yValidacao);
  fitValErrorRandom(exp) = feval('fit',yValPrevisto, yValidacao);
  % Previs�o do conjunto de teste
  predictedTestOutput = test_esn_modificada(testInputSequence, tr_esn_random{exp}, 0,'startingState',finalState);
  %predictedTestOutput = test_esn_modificada(testInputSequence, best_ESN, 0);
  yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);
  % C�lculo do RMSE no conjunto de teste
  testErrorRandom(exp) = feval(metricaErro,yTestePrevisto, yTeste);
  RRSETestErrorRandom(exp) = feval('RRSE',yTestePrevisto, yTeste);
  fitTestErrorRandom(exp) = feval('fit',yTestePrevisto, yTeste);
  
    
end
tRandomParcial = toc;
% Fim Random
clear predictedTrainOutput predictedValOutput predictedTestOutput finalState...
      yTreinoPrevisto yValPrevisto yTestePrevisto minValErrorRandom posMinVal valErrorR

%% Etapa 2 - Otimiza��o dos pesos do reservat�rio
tic
aux2=0;
for exp = 1:numExp

  reservatorio = ceil(sol{exp}(1)); % n�mero de neur�nios no reservat�rio

  raio =   sol{exp}(2);      % raio espectral
 
  ScalWin = sol{exp}(3);     % constante de multiplica��o da matriz Win
 
  ScalWback = sol{exp}(4);   % constante de multiplica��o da matriz Wback
 
  vazamento = sol{exp}(5);   % constante de vazamento do neur�nio leaky integrator

  conectividade = sol{exp}(6); % percentual de conectividade do reservat�rio

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
  
  posicoes = esnP{exp}.internalWeights ~= 0; % posi��es de W com valores diferentes de zero

  num_var2  = size(esnP{exp}.internalWeights(posicoes),1); % total de pesos n�o nulos de W

  limInf2 = (-0.5.*ones(1,num_var2))'; 
  limSup2 = (0.5.*ones(1,num_var2))';
 
  [solP{exp}, classic,  evobestFitness2{exp}, trace] = aeiq_br_real(num_var2, numPulses2, numClassic2, ...
     genToWidth2, generations2, crossover2, qCrossover2, showEv,  gap2, exp,seed,fun2,limInf2,limSup2,trainInputSequence,trainOutputSequence,valInputSequence,yValidacao,yMin,yMax,...
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
  
  % Previs�o do conjunto de treinamento
  [predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, trESNP{exp}, nForgetPoints);
  yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);
  % C�lculo do RMSE no conjunto de treinamento
  trainErrorPesos(exp) = feval(metricaErro,yTreinoPrevisto, yTreino);
  RRSETrainErrorPesos(exp) = feval('RRSE',yTreinoPrevisto, yTreino);
  fitTrainErrorPesos(exp) = feval('fit',yTreinoPrevisto, yTreino);
 
  % Previs�o do conjunto de valida��o 
  [predictedValOutput,finalState] = test_esn_modificada(valInputSequence, trESNP{exp}, 0,'startingState',finalState);
   yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
  % C�lculo do RMSE no conjunto de valida��o
  valErrorPesos(exp) = feval(metricaErro,yValPrevisto, yValidacao);
  RRSEValErrorPesos(exp) = feval('RRSE',yValPrevisto, yValidacao);
  fitValErrorPesos(exp) = feval('fit',yValPrevisto, yValidacao);
  
  % Previs�o do conjunto de teste
  predictedTestOutput = test_esn_modificada(testInputSequence, trESNP{exp}, 0,'startingState',finalState);
  %predictedTestOutput = test_esn_modificada(testInputSequence, best_ESN, 0);
  yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);
  %C�lculo do RMSE no conjunto de teste
  testErrorPesos(exp) = feval(metricaErro,yTestePrevisto, yTeste);
  RRSETestErrorPesos(exp) = feval('RRSE',yTestePrevisto, yTeste);
  fitTestErrorPesos(exp) = feval('fit',yTestePrevisto, yTeste);
  
  
  clear posicoes limInf2 limSup2 maxVal num_var2 
end   
tempoTotalPesos = toc;
% Fim Otimiza��o dos Pesos do Reservat�rio

MeanFitness2 = mean(bestFitness2);
StdFitness2 = std(bestFitness2);
meanEvoBestFitness2 = aux2./numExp;

figure('Name','Curva de Evolu��o - Etapa 2 - Otimiza��o dos pesos AEIQ','NumberTitle','off');
plot(meanEvoBestFitness2)
title('Curva de Evolu��o');
xlabel('Gera��es');
ylabel('Aptid�o');
savefig('Curva_Evolucao_Etapa_2');

clear opts3 aux2 classic trace reservatorio raio ScalWin ScalWback vazamento conectividade...
      nInputUnits nInternalUnits nOutputUnits predictedTrainOutput predictedValOutput predictedTestOutput finalState...
      yTreinoPrevisto yValPrevisto yTestePrevisto

%% C�lculo de m�tricas, Impress�o de resultados e gr�ficos

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

% melhor ESN SRG em rela��o ao teste
[bestErrorSRG,posBestSRG]=min(testErrorSRG);
bestEsnSRG=tr_esn_SRG{1,posBestSRG};
% Previs�o do conjunto de treinamento
[predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, bestEsnSRG, nForgetPoints);
yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);
% Previs�o do conjunto de valida��o 
[predictedValOutput,finalState] = test_esn_modificada(valInputSequence, bestEsnSRG, 0,'startingState',finalState);
yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
% Previs�o do conjunto de teste
predictedTestOutput = test_esn_modificada(testInputSequence, bestEsnSRG, 0,'startingState',finalState);
yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);


for j = 1:size(yTestePrevisto,2)
    figure('Name','Previs�o das vari�veis de sa�da com a melhor ESN (teste) da Etapa 2 - SRG','NumberTitle','on');
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
    title('Valida��o','FontSize',14);
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
xlabel('Separa��o das entradas: ||u_i(t)-u_j(t)||');
ylabel('Separa��o dos estados:  ||x_i(t)-x_j(t)||');
legend('||x_i(t)-x_j(t)||=||u_i(t)-u_j(t)||','Regress�o','Dados');
savefig('separation_ratio_graph_SRG');

% melhor ESN Random em rela��o ao teste
[bestErrorRandom,posBestRandom]=min(testErrorRandom);
bestEsnRandom=tr_esn_random{1,posBestRandom};
% Previs�o do conjunto de treinamento
[predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, bestEsnRandom, nForgetPoints);
yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);
% Previs�o do conjunto de valida��o 
[predictedValOutput,finalState] = test_esn_modificada(valInputSequence, bestEsnRandom, 0,'startingState',finalState);
yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
% Previs�o do conjunto de teste
predictedTestOutput = test_esn_modificada(testInputSequence, bestEsnRandom, 0,'startingState',finalState);
yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);

for j = 1:size(yTestePrevisto,2)
    figure('Name','Previs�o das vari�veis de sa�da com a melhor ESN (teste) da Etapa 2 - Random','NumberTitle','on');
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
    title('Valida��o','FontSize',14);
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
   
   % Per�odo de washout
   stateCollectMat = compute_statematrix(trainInputSequence(1:nForgetPoints,:),trainOutputSequence(1:nForgetPoints,:),esn_random{exp},0);
   initialState =[stateCollectMat(end,:) trainOutputSequence(nForgetPoints,:)];     
   
   % Agrupamento usando os vizinhos mais pr�ximos
   %[idx, dist] = knnsearch(trainInputSequence((nForgetPoints+1):end,:),trainInputSequence((nForgetPoints+1):end,:),'distance','euclidean','k',2); % Distancia
   pseudoPadrao = [trainInputSequence((nForgetPoints+1):end,:) trainOutputSequence(nForgetPoints:(end-1),:)];
   [idx, dist] = knnsearch(pseudoPadrao,pseudoPadrao,'distance','euclidean','k',2); % Distancia
   % Dist�ncia euclidiana dos padr�es de entrada agrupados
   dist_entradas=dist(:,2);
   
   % Coleta dos estados gerados pelos padr�es, desconsiderando o efeito de estados
   % anteriores e fazendo teach forcing na sa�da 
   estados = ...
    compute_statematrix_SRG_simplificada2(trainInputSequence((nForgetPoints+1):end,:), trainOutputSequence((nForgetPoints+1):end,:), esn_random{exp}, 0,initialState);
   
   % Dist�ncia dos estados agrupados pelo knn
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
xlabel('Separa��o das entradas: ||u_i(t)-u_j(t)||');
ylabel('Separa��o dos estados:  ||x_i(t)-x_j(t)||');
legend('||x_i(t)-x_j(t)||=||u_i(t)-u_j(t)||','Regress�o','Dados');
savefig('separation_ratio_graph_Random');

% melhor ESN Pesos em rela��o ao teste
[bestErrorPesos,posBestPesos]=min(testErrorPesos);
bestEsnPesos=trESNP{posBestPesos};
% Previs�o do conjunto de treinamento
[predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, bestEsnPesos, nForgetPoints);
yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);
% Previs�o do conjunto de valida��o 
[predictedValOutput,finalState] = test_esn_modificada(valInputSequence, bestEsnPesos, 0,'startingState',finalState);
yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
% Previs�o do conjunto de teste
predictedTestOutput = test_esn_modificada(testInputSequence, bestEsnPesos, 0,'startingState',finalState);
yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);

for j = 1:size(yTestePrevisto,2)
    figure('Name','Previs�o das vari�veis de sa�da com a melhor ESN (teste) da Etapa 2 - Otimiza��o dos Pesos','NumberTitle','on');
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
    title('Valida��o','FontSize',14);
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
     
   % Per�odo de washout
   stateCollectMat = compute_statematrix(trainInputSequence(1:nForgetPoints,:),trainOutputSequence(1:nForgetPoints,:),esnP{exp},0);
   initialState =[stateCollectMat(end,:) trainOutputSequence(nForgetPoints,:)]; 
   
   % Agrupamento usando os vizinhos mais pr�ximos
   %[idx, dist] = knnsearch(trainInputSequence((nForgetPoints+1):end,:),trainInputSequence((nForgetPoints+1):end,:),'distance','euclidean','k',2); % Distancia
   pseudoPadrao = [trainInputSequence((nForgetPoints+1):end,:) trainOutputSequence(nForgetPoints:(end-1),:)];
   [idx, dist] = knnsearch(pseudoPadrao,pseudoPadrao,'distance','euclidean','k',2); % Distancia
   % Dist�ncia euclidiana dos padr�es de entrada agrupados
   dist_entradas=dist(:,2);
   
   % Coleta dos estados gerados pelos padr�es, desconsiderando o efeito de estados
   % anteriores e fazendo teach forcing na sa�da 
   estados = ...
    compute_statematrix_SRG_simplificada2(trainInputSequence((nForgetPoints+1):end,:), trainOutputSequence((nForgetPoints+1):end,:), esnP{exp}, 0,initialState);
   
   % Dist�ncia dos estados agrupados pelo knn
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
xlabel('Separa��o das entradas: ||u_i(t)-u_j(t)||');
ylabel('Separa��o dos estados:  ||x_i(t)-x_j(t)||');
legend('||x_i(t)-x_j(t)||=||u_i(t)-u_j(t)||','Regress�o','Dados');
savefig('separation_ratio_graph_Pesos');

% Impress�o dos resultados
fprintf('\n\n             Etapa 1\n\n');
% Etapa 1
for exp =1:numExp
    
fprintf('Exp %d - Aptidao: %f\n',exp,bestFitness(exp));
    
end

fprintf('\nAptid�o M�dia: %f\n',MeanFitness);
fprintf('\nAptid�o Desvio: %f\n',StdFitness);
fprintf('\nTempo Total(%d experimentos): %f s\n',numExp,tempo1);

% Etapa 2 - Random
fprintf('\n\n             Etapa 2 - Random\n\n');
fprintf('                RMSE\n\n');
for exp =1:numExp
    
fprintf('Exp %d - Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',exp,trainErrorRandom(exp),...
    valErrorRandom(exp),testErrorRandom(exp),SRG_bestRandom(exp));
    
end

fprintf('\nM�dia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanTrainErrorRandom,...
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

fprintf('\nM�dia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanfitTrainErrorRandom,...
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

fprintf('\nM�dia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanRRSETrainErrorRandom,...
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

fprintf('\nM�dia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanTrainErrorSRG,...
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

fprintf('\nM�dia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanfitTrainErrorSRG,...
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

fprintf('\nM�dia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanRRSETrainErrorSRG,...
    meanRRSEValErrorSRG,meanRRSETestErrorSRG);
fprintf('\nDesvio: Treinamento: %f  Validacao: %f  Teste: %f\n',stdRRSETrainErrorSRG,...
    stdRRSEValErrorSRG,stdRRSETestErrorSRG);
fprintf('\nMelhor: Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',RRSETrainErrorSRG(posBestSRG),...
    RRSEValErrorSRG(posBestSRG),RRSETestErrorSRG(posBestSRG),SRG{posBestSRG});


% Etapa 2 - Otimiza��o dos pesos do reservat�rio
fprintf('\n\n             Etapa 2 - Otimiza��o dos pesos do reservat�rio\n\n');
fprintf('                RMSE\n\n');
for exp =1:numExp
    
fprintf('Exp %d - Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',exp,trainErrorPesos(exp),...
    valErrorPesos(exp),testErrorPesos(exp),SRG_bestPesos(exp));
    
end

fprintf('\nM�dia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanTrainErrorPesos,...
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

fprintf('\nM�dia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanfitTrainErrorPesos,...
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

fprintf('\nM�dia: Treinamento: %f  Validacao: %f  Teste: %f\n',meanRRSETrainErrorPesos,...
    meanRRSEValErrorPesos,meanRRSETestErrorPesos);
fprintf('\nDesvio: Treinamento: %f  Validacao: %f  Teste: %f\n',stdRRSETrainErrorPesos,...
    stdRRSEValErrorPesos,stdRRSETestErrorPesos);
fprintf('\nMelhor: Treinamento: %f  Validacao: %f  Teste: %f  SRG: %f\n',RRSETrainErrorPesos(posBestPesos),...
    RRSEValErrorPesos(posBestPesos),RRSETestErrorPesos(posBestPesos),SRG_bestPesos(posBestPesos));

fprintf('\nTempo Total(%d experimentos): %f s\n',numExp,tempoTotalPesos);

save (nomeTeste)

diary off
 
