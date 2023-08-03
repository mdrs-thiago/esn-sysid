function valErrorFinal = calculaErroPesos(x,varargin)

pesos = x';  % pesos de W
trainInputSequence  = varargin {1};
trainOutputSequence = varargin {2};
valInputSequence    = varargin {3};
valOutputSequence   = varargin {4};
yMin                = varargin {5};
yMax                = varargin {6};
nForgetPoints       = varargin {7};
esn                 = varargin {8};
metricaErro         = varargin {9};
posicoes            = varargin {10};

    esn.internalWeights (posicoes)= pesos;
    opts.tol = 1e-2;
    opts.disp = 0;
    maxVal = max(abs(eigs(esn.internalWeights,1, 'lm', opts)));
    esn.internalWeights = esn.internalWeights/maxVal;
    esn.internalWeights = esn.internalWeights.*esn.spectralRadius;

    % Treinamento da ESN
    trainedEsn = ...
   train_esn(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ; 
   
    % Previsão do conjunto de treinamento
   [predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, trainedEsn, nForgetPoints);
    
    % Previsão do conjunto de validação 
    predictedValOutput = test_esn(valInputSequence, trainedEsn, 0,'startingState',finalState);
    
     yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
    % Cálculo do RMSE no conjunto de validação
    valErrorFinal = feval(metricaErro,yValPrevisto,valOutputSequence);
end