
function valErrorFinal = calculaErroFinal(x,varargin)

reservatorio  = ceil(x(1)); % tamanho do reservatorio
raio          = x(2);       % raio espectral
ScalWin       = x(3);       % fator de escala da matriz Win
ScalWback     = x(4);       % fator de escala da matriz Wback
vazamento     = x(5);       % leaky rate do neurônio leaky integrator
conectividade = x(6);       % percentual de conectividade do reservatório
trainInputSequence  = varargin {1};
trainOutputSequence = varargin {2};
valInputSequence    = varargin {3};
valOutputSequence   = varargin {4};
yMin                = varargin {5};
yMax                = varargin {6};
nForgetPoints       = varargin {7};
numRedes            = varargin {8};
metricaErro         = varargin {9};

valErrorM = 0;

%%%% generate an esn 
nInputUnits = size(trainInputSequence,2); nInternalUnits = reservatorio; nOutputUnits = size(trainOutputSequence,2); 
 
for R = 1:numRedes

    esn = generate_esn2(nInputUnits, nInternalUnits, nOutputUnits, ScalWin, ScalWback, conectividade, ...
      'spectralRadius',raio,'inputScaling',1* ones(nInputUnits,1),'inputShift',0*ones(nInputUnits,1), ...
      'teacherScaling',1*ones(nOutputUnits,1),'teacherShift',0*ones(nOutputUnits,1),'feedbackScaling', 1, ...
      'type', 'leaky_esn','leakage',vazamento,...
      'reservoirActivationFunction','tanh',...
      'outputActivationFunction', 'identity','inverseOutputActivationFunction','identity',...
      'methodWeightCompute','pseudoinverse', 'timeConstants',1 * ones(nInternalUnits,1));  

    esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;

    % Treinamento da ESN
    [trainedEsn stateMatrix] = ...
   train_esn(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ; 
   
    % Previsão do conjunto de treinamento
   [predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, trainedEsn, nForgetPoints);
    
    % Previsão do conjunto de validação 
    predictedValOutput = test_esn(valInputSequence, trainedEsn, 0,'startingState',finalState);
    
     yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
    % Cálculo do RMSE no conjunto de validação
    valError = feval(metricaErro,yValPrevisto,valOutputSequence);
   
    % Acumulação do erro
    valErrorM = valErrorM + mean(valError);   
end
 
% Erro de Validação Final Médio
valErrorFinal = valErrorM./numRedes;

end