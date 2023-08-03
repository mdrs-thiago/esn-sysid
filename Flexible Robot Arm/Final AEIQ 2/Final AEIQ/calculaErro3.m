 function valErrorFinal = calculaErro3(x,varargin)

 pesos = x';  % pesos de W
 trainInputSequence  = varargin {1};
 trainOutputSequence = varargin {2};
 valInputSequence    = varargin {3};
 valOutputSequence   = varargin {4};
 yMin                = varargin {5};
 yMax                = varargin {6};
 nForgetPoints       = varargin {7};
 metricaErro         = varargin {8};
 nInputUnits         = varargin {9};
 nOutputUnits        = varargin {10};
 posicoes            = varargin {11};

       
     num_par = 6;
%     a_reservatorio = 5;
%     a_raio = 1;
%     a_ScalWin = 1;
%     a_ScalWback = 1;
%     a_vazamento = 1;
%     a_conectividade = 0.7;
%     
%     nInputUnits = 4;
%     nOutputUnits = 4;
%     a_Win = [0.0648 0.0316 0.6687 -0.9664 -0.9374; -0.9982 0.5483 -0.9330 -0.0545 0.6632];
%     a_Wback = [0.2937 0.6833 -0.7548; -0.9440 0.4879 0.0146; -0.3033 0.5730 -0.2336;...
%         -0.5478 0.8749 0.8204; -0.3003 0.9741 -0.5229];
% 
%     a_W = c;
    
%     posicoes = a_W~=0;
%     
%     v = full(a_W(posicoes));
%     
%     x = [a_reservatorio a_raio a_ScalWin a_ScalWback a_vazamento a_conectividade...
%         reshape(a_Win',1,nInputUnits.*a_reservatorio) reshape(a_Wback,1,nOutputUnits.*a_reservatorio)...
%         v'];
    
    reservatorio = ceil(x(1));
    raio = x(2);
    ScalWin = x(3);
    ScalWback = x(4);
    vazamento = x(5);
    conectividade = x(6);
    
    Wo_in = reshape(x((num_par+1):(num_par+nInputUnits*reservatorio)),reservatorio,[]);
    
    Wo_back = reshape(x((num_par+nInputUnits*reservatorio+1):(num_par+nInputUnits*reservatorio+nOutputUnits*reservatorio)),reservatorio,[]);
    
    Wo = double(posicoes);
    Wo (posicoes)= x((num_par+nInputUnits*reservatorio+nOutputUnits*reservatorio+1):end);
    
      nInternalUnits =reservatorio;
      esn = generate_esn4(nInputUnits, nInternalUnits, nOutputUnits, ScalWin, ScalWback, Wo_in, Wo_back, Wo, ...
      'spectralRadius',raio,'inputScaling',1* ones(nInputUnits,1),'inputShift',0*ones(nInputUnits,1), ...
      'teacherScaling',1*ones(nOutputUnits,1),'teacherShift',0*ones(nOutputUnits,1),'feedbackScaling', 1, ...
      'type', 'leaky_esn','leakage',vazamento,...
      'reservoirActivationFunction','tanh',...
      'outputActivationFunction', 'identity','inverseOutputActivationFunction','identity',...
      'methodWeightCompute','pseudoinverse', 'timeConstants',1 * ones(nInternalUnits,1));  
    
      esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;

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
% end