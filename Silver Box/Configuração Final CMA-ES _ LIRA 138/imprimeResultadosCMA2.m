
clc
clear all
close all

load resultadosOtimizacaoCMA.mat;

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


% for j = 1:size(yTestePrevisto,2)
%     figure('Name','Previsão das variáveis de saída com a melhor ESN (teste) da Etapa 2 - SRG','NumberTitle','on');
% %     amostras = [1:length([yTreino; yValidacao;yTeste])];
%     amostras = [40001:91072 91073:131072 1:40000];
%     subplot(3,1,1);
%     plot(amostras((nForgetPoints+1):length(yTreino)),yTreino((nForgetPoints+1):end,j),'b');
%     hold on
%     plot(amostras((nForgetPoints+1):length(yTreino)),yTreinoPrevisto(:,j),'r');
%     lgd = legend('Real','CMA-ES/SRG/ESN');
%     lgd.FontSize = 16;
%     xlabel('Amostras','FontSize',14);
%     
%     if(size(yTestePrevisto,2)==1)
%         saida = strcat('y(k) ');
%     else        
%         auxr = num2str(j);
%         saida = strcat('y_',auxr,'(k) ');
%     end
%     
%     ylabel(saida,'FontSize',14);
%     title('Treinamento','FontSize',14);
%     subplot(3,1,2);
%     plot(amostras(length(yTreino)+1:length([yTreino;yValidacao])),yValidacao(:,j),'b');
%     hold on
%     plot(amostras(length(yTreino)+1:length([yTreino;yValidacao])),yValPrevisto(:,j),'r');
%     lgd = legend('Real','CMA-ES/SRG/ESN');
%     lgd.FontSize = 16;
%     xlabel('Amostras','FontSize',14);
%     ylabel(saida,'FontSize',14);
%     title('Validação','FontSize',14);
%     subplot(3,1,3);
%     plot(amostras(length([yTreino;yValidacao])+1:length([yTreino;yValidacao;yTeste])),yTeste(:,j),'b');
%     hold on
%     plot(amostras(length([yTreino;yValidacao])+1:length([yTreino;yValidacao;yTeste])),yTestePrevisto(:,j),'r');
%     lgd = legend('Real','CMA-ES/SRG/ESN');
%     lgd.FontSize = 16;
%     xlabel('Amostras','FontSize',14);
%     ylabel(saida,'FontSize',14);
%     title('Teste','FontSize',14);
%     filename = strcat('previsao_',saida,'_SRG');
%     savefig(filename);
% end
% 
% clear predictedTrainOutput predictedValOutput predictedTestOutput finalState...
%       yTreinoPrevisto yValPrevisto yTestePrevisto filename saida auxr
% 
% % melhor ESN Random em relação ao teste
% [bestErrorRandom,posBestRandom]=min(testErrorRandom);
% bestEsnRandom=tr_esn_random{1,posBestRandom};
% % Previsão do conjunto de treinamento
% [predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, bestEsnRandom, nForgetPoints);
% yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);
% % Previsão do conjunto de validação 
% [predictedValOutput,finalState] = test_esn_modificada(valInputSequence, bestEsnRandom, 0,'startingState',finalState);
% yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
% % Previsão do conjunto de teste
% predictedTestOutput = test_esn_modificada(testInputSequence, bestEsnRandom, 0,'startingState',finalState);
% yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);
% 
% for j = 1:size(yTestePrevisto,2)
%     figure('Name','Previsão das variáveis de saída com a melhor ESN (teste) da Etapa 2 - Random','NumberTitle','on');
%     %amostras = [1:length([yTreino; yValidacao;yTeste])];
%     amostras = [40001:91072 91073:131072 1:40000];
%     subplot(3,1,1);
%     plot(amostras((nForgetPoints+1):length(yTreino)),yTreino((nForgetPoints+1):end,j),'b');
%     hold on
%     plot(amostras((nForgetPoints+1):length(yTreino)),yTreinoPrevisto(:,j),'r');
%     lgd = legend('Real','CMA-ES/Random/ESN');
%     lgd.FontSize = 16;
%     xlabel('Amostras','FontSize',14);
%     
%     if(size(yTestePrevisto,2)==1)
%         saida = strcat('y(k) ');
%     else        
%         auxr = num2str(j);
%         saida = strcat('y_',auxr,'(k) ');
%     end
%     
%     ylabel(saida,'FontSize',14);
%     title('Treinamento','FontSize',14);
%     subplot(3,1,2);
%     plot(amostras(length(yTreino)+1:length([yTreino;yValidacao])),yValidacao(:,j),'b');
%     hold on
%     plot(amostras(length(yTreino)+1:length([yTreino;yValidacao])),yValPrevisto(:,j),'r');
%     lgd = legend('Real','CMA-ES/Random/ESN');
%     lgd.FontSize = 16;
%     xlabel('Amostras','FontSize',14);
%     ylabel(saida,'FontSize',14);
%     title('Validação','FontSize',14);
%     subplot(3,1,3);
%     plot(amostras(length([yTreino;yValidacao])+1:length([yTreino;yValidacao;yTeste])),yTeste(:,j),'b');
%     hold on
%     plot(amostras(length([yTreino;yValidacao])+1:length([yTreino;yValidacao;yTeste])),yTestePrevisto(:,j),'r');
%     lgd = legend('Real','CMA-ES/Random/ESN');
%     lgd.FontSize = 16;
%     xlabel('Amostras','FontSize',14);
%     ylabel(saida,'FontSize',14);
%     title('Teste','FontSize',14);
%     filename = strcat('previsao_',saida,'_Random');
%     savefig(filename);
% end
% 
% clear predictedTrainOutput predictedValOutput predictedTestOutput finalState...
%       yTreinoPrevisto yValPrevisto yTestePrevisto filename saida auxr
% 
% 
% % melhor ESN Pesos em relação ao teste
% [bestErrorPesos,posBestPesos]=min(testErrorPesos);
% bestEsnPesos=trESNP{posBestPesos};
% % Previsão do conjunto de treinamento
% [predictedTrainOutput,finalState] = test_esn_modificada(trainInputSequence, bestEsnPesos, nForgetPoints);
% yTreinoPrevisto = desnormaliza(predictedTrainOutput,yMin,yMax);
% % Previsão do conjunto de validação 
% [predictedValOutput,finalState] = test_esn_modificada(valInputSequence, bestEsnPesos, 0,'startingState',finalState);
% yValPrevisto = desnormaliza(predictedValOutput,yMin,yMax);
% % Previsão do conjunto de teste
% predictedTestOutput = test_esn_modificada(testInputSequence, bestEsnPesos, 0,'startingState',finalState);
% yTestePrevisto = desnormaliza(predictedTestOutput,yMin,yMax);
% 
% for j = 1:size(yTestePrevisto,2)
%     figure('Name','Previsão das variáveis de saída com a melhor ESN (teste) da Etapa 2 - Otimização dos Pesos','NumberTitle','on');
%     %amostras = [1:length([yTreino; yValidacao;yTeste])];
%     amostras = [40001:91072 91073:131072 1:40000];
%     subplot(3,1,1);
%     plot(amostras((nForgetPoints+1):length(yTreino)),yTreino((nForgetPoints+1):end,j),'b');
%     hold on
%     plot(amostras((nForgetPoints+1):length(yTreino)),yTreinoPrevisto(:,j),'r');
%     lgd = legend('Real','CMA-ES/CMA-ES/ESN');
%     lgd.FontSize = 16;
%     xlabel('Amostras','FontSize',14);
%     
%     if(size(yTestePrevisto,2)==1)
%         saida = strcat('y(k) ');
%     else        
%         auxr = num2str(j);
%         saida = strcat('y_',auxr,'(k) ');
%     end
%     
%     ylabel(saida,'FontSize',14);
%     title('Treinamento','FontSize',14);
%     subplot(3,1,2);
%     plot(amostras(length(yTreino)+1:length([yTreino;yValidacao])),yValidacao(:,j),'b');
%     hold on
%     plot(amostras(length(yTreino)+1:length([yTreino;yValidacao])),yValPrevisto(:,j),'r');
%     lgd = legend('Real','CMA-ES/CMA-ES/ESN');
%     lgd.FontSize = 16;
%     xlabel('Amostras','FontSize',14);
%     ylabel(saida,'FontSize',14);
%     title('Validação','FontSize',14);
%     subplot(3,1,3);
%     plot(amostras(length([yTreino;yValidacao])+1:length([yTreino;yValidacao;yTeste])),yTeste(:,j),'b');
%     hold on
%     plot(amostras(length([yTreino;yValidacao])+1:length([yTreino;yValidacao;yTeste])),yTestePrevisto(:,j),'r');
%     lgd = legend('Real','CMA-ES/CMA-ES/ESN');
%     lgd.FontSize = 16;
%     xlabel('Amostras','FontSize',14);
%     ylabel(saida,'FontSize',14);
%     title('Teste','FontSize',14);
%     filename = strcat('previsao_',saida,'_Pesos');
%     savefig(filename);
% end
% clear predictedTrainOutput predictedValOutput predictedTestOutput finalState...
%       yTreinoPrevisto yValPrevisto yTestePrevisto filename saida auxr

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
