%%TESTE de Otimiza��o com Matlab

clear all
close all
clc

% Load de dados
load steamNorm.mat
nomeTeste ='resultadosOtimizacaoAEIQ';

numExp = 1;       % n�mero de experimentos 

nForgetPoints = 100; % T0 - n�mero de dados descartados durante o washout

numRedes = 5;

metricaErro ='RMSE';

num_var = 6; % n�mero de vari�veis a serem otimizadas
% tamanho do reservat�rio
% raio espectral
% fator de escala de Win
% fator de escala de Wback
% leaky rate
% percentual de conectividade do reservat�rio

T = size(trainInputSequence,1);
limInf = [floor((T-nForgetPoints)/10); 0.10; 0.0001; 0; 0.1;0.1]; % limites inferiores das vari�veis

limSup = [floor((T-nForgetPoints)/2); 0.99; 1; 1; 1; 1];   % limites superiores das vari�veis

fun = 'calculaErroFinal'; % nome da fun��o de aptid�o

insigma = 0.3*(limSup-limInf); % initial step size

c = eye(num_var);  % covariance matrix

xstart = limInf + rand(num_var,1).*(limSup-limInf); % media inicial da fun��o normal

pop = 50;     % population size (lambda)
opts.ParentNumber = 25; % (mi)
geracoes = '100';

opts.StopFitness  = '-Inf';
opts.MaxFunEvals  = 'inf';
opts.StopFunEvals = 'Inf';
opts.TolX         = '-inf';
opts.TolUpX       = 'inf';
opts.StopOnStagnation = 'off';
opts.StopIter     = geracoes;
opts.TolFun       = '-inf % stop if fun-changes smaller TolFun';
opts.TolHistFun   = '-inf % stop if back fun-changes smaller TolHistFun';
opts.StopOnEqualFunctionValues = 'inf  % number of iterations';
opts.EvalInitialX = 'no';
opts.IncPopSize = 0;
opts.PopSize      = pop;
opts.LBounds = limInf; 
opts.UBounds = limSup; 
opts.DispModulo = 1;
% Etapa 1 - Otimiza��o dos par�metros da ESN
tic
diary ('cma.txt');

aux =0;
for exp = 1:numExp


[sol{exp}, bestFitness(exp), counteval, stopflag,out,bestever,evobestFitness{exp}] = cmaes( ...
    fun, ...    % name of objective/fitness function
    xstart, ...    % objective variables initial point, determines N
    insigma, ...   % initial coordinate wise standard deviation(s)
    opts,...    % options struct, see defopts below
    trainInputSequence,trainOutputSequence,valInputSequence,yValidacao,yMin,yMax,...
    nForgetPoints,numRedes,metricaErro);

aux = aux+evobestFitness{exp};
 
end
tempo1 = toc;

MeanFitness = mean(bestFitness);
StdFitness = std(bestFitness);
meanEvoBestFitness = aux./numExp;

plot(meanEvoBestFitness)
title('Curva de Evolu��o')
xlabel('Gera��es')
ylabel('Aptid�o')


% %Gera��o das numRedes para os numExp realizados
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

% Etapa 2 - SRG
tic
for exp = 1:numExp
  for i=1:numRedes  
   % Agrupamento usando os vizinhos mais pr�ximos
   [idx, dist] = knnsearch(trainInputSequence((nForgetPoints+1):end,:),trainInputSequence((nForgetPoints+1):end,:),'distance','euclidean','k',2); % Distancia
   % Dist�ncia euclidiana dos padr�es de entrada agrupados
   dist_entradas=dist(:,2);
   
   % Coleta dos estados gerados pelos padr�es, desconsiderando o efeito de estados
   % anteriores e fazendo teach forcing na sa�da 
   estados = ...
    compute_statematrix_SRG_simplificada(trainInputSequence((nForgetPoints+1):end,:), trainOutputSequence((nForgetPoints+1):end,:), esn_totais{exp,i}, 0);
   
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
         idx dist esn dist_entradas

  end


  for j =1:size(distancias,2)
%     figure;
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
 
  % Previs�o do conjunto de valida��o 
  [predictedValOutput,finalState] = test_esn_modificada(valInputSequence, tr_esn_SRG{exp}, 0,'startingState',finalState);
   yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
  % C�lculo do RMSE no conjunto de valida��o
  valErrorSRG(exp) = feval(metricaErro,yValPrevisto, yValidacao);
  RRSEValErrorSRG(exp) = feval('RRSE',yValPrevisto, yValidacao);
  
  % Previs�o do conjunto de teste
  predictedTestOutput = test_esn_modificada(testInputSequence, tr_esn_SRG{exp}, 0,'startingState',finalState);
  %predictedTestOutput = test_esn_modificada(testInputSequence, best_ESN, 0);
  yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);
  %C�lculo do RMSE no conjunto de teste
  testErrorSRG(exp) = feval(metricaErro,yTestePrevisto, yTeste);
  RRSETestErrorSRG(exp) = feval('RRSE',yTestePrevisto, yTeste);
  
 
end
tSRGParcial = toc;
% Fim SRG
tic

% Etapa 2 - Random
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
  % Previs�o do conjunto de valida��o 
  [predictedValOutput,finalState] = test_esn_modificada(valInputSequence, tr_esn_random{exp}, 0,'startingState',finalState);
  yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
  % C�lculo do RMSE no conjunto de valida��o
  valErrorRandom(exp) = feval(metricaErro,yValPrevisto, yValidacao);
  RRSEValErrorRandom(exp) = feval('RRSE',yValPrevisto, yValidacao);
  % Previs�o do conjunto de teste
  predictedTestOutput = test_esn_modificada(testInputSequence, tr_esn_random{exp}, 0,'startingState',finalState);
  %predictedTestOutput = test_esn_modificada(testInputSequence, best_ESN, 0);
  yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);
  % C�lculo do RMSE no conjunto de teste
  testErrorRandom(exp) = feval(metricaErro,yTestePrevisto, yTeste);
  RRSETestErrorRandom(exp) = feval('RRSE',yTestePrevisto, yTeste);

  
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
meanRRSETrainErrorSRG = mean(RRSETrainErrorSRG);
meanRRSEValErrorSRG = mean(RRSEValErrorSRG);
meanRRSETestErrorSRG = mean(RRSETestErrorSRG);
stdRRSETrainErrorSRG = std(RRSETrainErrorSRG);
stdRRSEValErrorSRG = std(RRSEValErrorSRG);
stdRRSETestErrorSRG = std(RRSETestErrorSRG);

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
    title('Valida��o','FontSize',14);
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
xlabel('Separa��o das entradas: ||u_i(t)-u_j(t)||');
ylabel('Separa��o dos estados:  ||x_i(t)-x_j(t)||');
legend('||x_i(t)-x_j(t)||=||u_i(t)-u_j(t)||','Regress�o','Dados');

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
    title('Valida��o','FontSize',14);
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
   % Agrupamento usando os vizinhos mais pr�ximos
   [idx, dist] = knnsearch(trainInputSequence((nForgetPoints+1):end,:),trainInputSequence((nForgetPoints+1):end,:),'distance','euclidean','k',2); % Distancia
   % Dist�ncia euclidiana dos padr�es de entrada agrupados
   dist_entradas=dist(:,2);
   
   % Coleta dos estados gerados pelos padr�es, desconsiderando o efeito de estados
   % anteriores e fazendo teach forcing na sa�da 
   estados = ...
    compute_statematrix_SRG_simplificada(trainInputSequence((nForgetPoints+1):end,:), trainOutputSequence((nForgetPoints+1):end,:), esn_random{exp}, 0);
   
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
  
  clear erro_chvn SRG_rede distancias
 end
figure;
plotregression(par_dist_bestRandom{posBestRandom}(1,:),par_dist_bestRandom{posBestRandom}(2,:));
subplot(1,1,1);
xlabel('Separa��o das entradas: ||u_i(t)-u_j(t)||');
ylabel('Separa��o dos estados:  ||x_i(t)-x_j(t)||');
legend('||x_i(t)-x_j(t)||=||u_i(t)-u_j(t)||','Regress�o','Dados');

clc
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

fprintf('\nTempo Total(%d experimentos): %f s\n',numExp,tempoTotalSRG);

save (nomeTeste)

diary off
 