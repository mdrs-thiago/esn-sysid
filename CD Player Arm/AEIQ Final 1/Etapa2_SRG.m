%% Etapa 2 - SRG
clc
clear all
close all
load resultadosOtimizacaoAEIQ
tic
for exp = 1:numExp
  for i=1:numRedes
   
    % Período de washout
   stateCollectMat = compute_statematrix(trainInputSequence(1:nForgetPoints,:),trainOutputSequence(1:nForgetPoints,:),esn_totais{exp,i},0);
   initialState =[stateCollectMat(end,:) trainOutputSequence(nForgetPoints,:)];
      
   % Agrupamento usando os vizinhos mais próximos
   %[idx, dist] = knnsearch(trainInputSequence((nForgetPoints+1):end,:),trainInputSequence((nForgetPoints+1):end,:),'distance','euclidean','k',2); % Distancia
   pseudoPadrao = [trainInputSequence((nForgetPoints+1):end,:) trainOutputSequence(nForgetPoints:(end-1),:)];
   [idx, dist] = knnsearch(pseudoPadrao,pseudoPadrao,'distance','euclidean','k',2); % Distancia
   % Distância euclidiana dos padrões de entrada agrupados
   dist_entradas=dist(:,2);
   
   % Coleta dos estados gerados pelos padrões, desconsiderando o efeito de estados
   % anteriores e fazendo teach forcing na saída 
%    estados = ...
%     compute_statematrix_SRG_simplificada(trainInputSequence((nForgetPoints+1):end,:), trainOutputSequence((nForgetPoints+1):end,:), esn_totais{exp,i}, 0);
   estados = ...
    compute_statematrix_SRG_simplificada2(trainInputSequence((nForgetPoints+1):end,:), trainOutputSequence((nForgetPoints+1):end,:), esn_totais{exp,i}, 0,initialState);  

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
  exp
  % Escolha da rede com menor SRG
  [minSRG,posMinSRG]=min(SRG_rede);
  esn_SRG{exp}=esn_totais{exp,posMinSRG};
  par_dist_SRG{exp} = distancias{1,posMinSRG}; % par de distancias para cada rede
  SRG{exp}=minSRG;

  clear erro_chvn SRG_rede minSRG distancias
  % Treinamento da rede com o menor SRG 
  tr_esn_SRG{exp} = ...
  train_esn(trainInputSequence, trainOutputSequence, esn_SRG{exp}, nForgetPoints) ; 
  
  % Previsão do conjunto de treinamento
  [predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, tr_esn_SRG{exp}, nForgetPoints);
  yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);
  % Cálculo do RMSE_mV no conjunto de treinamento
  trainErrorSRG(exp) = feval(metricaErro,yTreinoPrevisto, yTreino)
  RRSETrainErrorSRG(exp) = feval('RRSE',yTreinoPrevisto, yTreino);
  fitTrainErrorSRG(exp) = feval('fit',yTreinoPrevisto, yTreino);
 
  % Previsão do conjunto de validação 
  [predictedValOutput,finalState] = test_esn_modificada(valInputSequence, tr_esn_SRG{exp}, 0,'startingState',finalState);
   yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
  % Cálculo do RMSE_mV no conjunto de validação
  valErrorSRG(exp) = feval(metricaErro,yValPrevisto, yValidacao)
  RRSEValErrorSRG(exp) = feval('RRSE',yValPrevisto, yValidacao);
  fitValErrorSRG(exp) = feval('fit',yValPrevisto, yValidacao);
  
  % Previsão do conjunto de teste
  predictedTestOutput = test_esn_modificada(testInputSequence, tr_esn_SRG{exp}, 0,'startingState',finalState);
  %predictedTestOutput = test_esn_modificada(testInputSequence, best_ESN, 0);
  yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);
  %Cálculo do RMSE_mV no conjunto de teste
  testErrorSRG(exp) = feval(metricaErro,yTestePrevisto, yTeste)
  RRSETestErrorSRG(exp) = feval('RRSE',yTestePrevisto, yTeste);
  fitTestErrorSRG(exp) = feval('fit',yTestePrevisto, yTeste);
  
 
end
tSRGParcial = toc;
% Fim SRG