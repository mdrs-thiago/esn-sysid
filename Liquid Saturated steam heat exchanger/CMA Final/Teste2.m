clc
clear all
close all
tic

load steamNorm.mat

param_real = 6;

numPulses = 10;

numClassic = 100;

genToWidth = 2;

generations = 100;

crossover = 0.6;

qCrossover = 0.8;

showEv = 1;

gap = numClassic;

experimento = 1;

seed = rng;

fun = 'calculaErro4';

nForgetPoints = 100;

metricaErro = 'RMSE';

nInputUnits = 4;
nOutputUnits = 4;

limInf = [50; 0.10; 0.0001; 0; 0.1; 0.1]; % limites inferiores das variáveis

limSup = [500; 0.5; 1; 1; 1; 1];   % limites superiores das variáveis

[x, posicoes, classic,  evobestFitness, trace] = aeiq_br_real3(param_real, numPulses, numClassic, ...
     genToWidth, generations, crossover, qCrossover, showEv,  gap, experimento,seed,fun,limInf,limSup,trainInputSequence,trainOutputSequence,valInputSequence,yValidacao,yMin,yMax,...
    nForgetPoints,metricaErro);

reservatorio = ceil(x(1));
    raio = x(2);
    ScalWin = x(3);
    ScalWback = x(4);
    vazamento = x(5);
    conectividade = x(6);
    
    Wo_in = reshape(x((param_real+1):(param_real+nInputUnits*reservatorio)),reservatorio,[]);
    
    Wo_back = reshape(x((param_real+nInputUnits*reservatorio+1):(param_real+nInputUnits*reservatorio+nOutputUnits*reservatorio)),reservatorio,[]);
    
    Wo = double(posicoes{1,1});
    Wo (posicoes{1,1})= x((param_real+nInputUnits*reservatorio+nOutputUnits*reservatorio+1):end);
    
      nInternalUnits =reservatorio;
      esn = generate_esn5(nInputUnits, nInternalUnits, nOutputUnits, ScalWin, ScalWback, Wo_in, Wo_back, Wo, ...
      'spectralRadius',raio,'inputScaling',1* ones(nInputUnits,1),'inputShift',0*ones(nInputUnits,1), ...
      'teacherScaling',1*ones(nOutputUnits,1),'teacherShift',0*ones(nOutputUnits,1),'feedbackScaling', 1, ...
      'type', 'leaky_esn','leakage',vazamento,...
      'reservoirActivationFunction','tanh',...
      'outputActivationFunction', 'identity','inverseOutputActivationFunction','identity',...
      'methodWeightCompute','pseudoinverse', 'timeConstants',1 * ones(nInternalUnits,1));  
    
      esn.internalWeights = esn.internalWeights_UnitSR;

    % Treinamento da ESN
    trainedEsn = ...
   train_esn(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ; 
   
    % Previsão do conjunto de treinamento
   [predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, trainedEsn, nForgetPoints);
    
    % Previsão do conjunto de validação 
    predictedValOutput = test_esn(valInputSequence, trainedEsn, 0,'startingState',finalState);
    
     yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
    % Cálculo do RMSE no conjunto de validação
    valErrorFinal = feval(metricaErro,yValPrevisto,yValidacao)
    
    % Previsão do conjunto de teste
    predictedTestOutput = test_esn_modificada(testInputSequence, trainedEsn, 0,'startingState',finalState);
    yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);
    
    testError = feval(metricaErro,yTestePrevisto,yTeste)
    
    toc
 