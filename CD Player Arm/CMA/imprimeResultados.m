clc
clear all
close all

load resultadosOtimizacaoCMA.mat

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
