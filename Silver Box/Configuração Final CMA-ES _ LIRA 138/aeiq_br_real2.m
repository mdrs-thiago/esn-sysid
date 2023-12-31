 function [bestexperimento, posicoesBestExperimento,classic,  evobestFitness, trace] = aeiq_br_real2(param_real, numPulses, numClassic, ...
     genToWidth, generations, crossover, qCrossover, showEv,  gap, experimento,seed,fun,limInf,limSup,varargin)


nInput = size(varargin{1},2);

nOutput =  size(varargin{2},2);

qreal = [limInf limSup];

% n�mero de genes, obtido pela matriz de intervalos qreal
numGenesQreal = size(qreal,1);

% o n�mero de sorteios para os indiv�duos cl�ssicos deve ser um m�ltiplo do
% n�mero de ind�viduos qu�nticos
if (rem(numClassic, numPulses) ~= 0)
   error('O n�mero de indiv�duos cl�ssicos deve ser m�ltiplo do n�mero de individuos qu�nticos');
end

% REPRESENTA��O INDIV�DUO REAL

%amplitude (sigma)
for j=1:numGenesQreal;
initialWidth(1,j)=(qreal(j,2)-qreal(j,1));
end

%centro  (mi)
for j=1:numGenesQreal;
initialCenter(1,j)=((qreal(j,2)+qreal(j,1))/2);
end

%limiteinferior
lowerQreal=[qreal(1:end,1)]';

for j=1:gap
    for k=1:numGenesQreal;
    lowerverifica(j,k)=lowerQreal(1,k);
    end
end
         
for j=1:numPulses
    for k=1:numGenesQreal;
    lowerverificaP(j,k)=lowerQreal(1,k);
    end
end

%limitesuperior
upperQreal=[qreal(1:end,2)]';

for j=1:gap
    for k=1:numGenesQreal;
    upperverifica(j,k)=upperQreal(1,k);
    end
end
       
for j=1:numPulses
    for k=1:numGenesQreal;
    upperverificaP(j,k)=upperQreal(1,k);
    end
end


% LISTA DE INDIV�DUOS QUANTICOS REAIS

%pulsesWidth
for j=1:numPulses
    for k=1:numGenesQreal;
      pulsesWidth(j,k)=initialWidth(1,k);
    end
end

%pulsesCenter
for j=1:numPulses
    for k=1:numGenesQreal;
      pulsesCenter(j,k)=initialCenter(1,k);
    end
end

% PAR�METROS DE INTERESSE

bestInd = [];
bestFitness = [];
best = [];
posicoesBest = [];
bestQreal = [];
bestWo = [];
bestWo_in = [];
bestWo_back = [];

evobestFitness = [];

% INICIALIZA LOOPING

for g = 1:generations,
    
    % SE O GAP DE INDIV�DUOS PARA CROSSOVER FOR MAIOR QUE INDIVIDUOS
    % QU�NTICOS, ENT�O, ALGUNS INDIVIDUOS QUANTICOS SER�O SELECIONADOS MAIS
    % VEZES ALEATORIAMENTE PARA CROSSOVER. CASO CONTR�RIO, ALGUNS
    % INDIVIDUOS QUANTICOS DEIXAR�O DE SEREM SELECIONADOS PARA CROSSOVER.
    
    if (gap <= numPulses)
        thePulses = randperm(numPulses);
        thePulses = thePulses(1:gap);
    else
        thePulses = [repmat(1:numPulses,[1 floor(gap / numPulses)]) (1:mod(gap, numPulses))];
    end
                                                                                                                                                                                         
    if (g == 1) 
        
        % INICIALIZA POPULA��O CL�SSICA. INDIV�DUOS POSS�VEIS TEM A MESMA
        % PROBABILIDADE DE SEREM GERADOS.
        
        % gerando indiv�duos rand�micos
        classicQreal = rand(numClassic, numGenesQreal);
        % corrigindo no dom�nio
        classicQreal= classicQreal.* repmat(initialWidth,numClassic,1) + repmat(lowerQreal,numClassic,1);
                 
    else
        
        % GAP DE INDIV�DUOS CL�SSICOS S�O GERADOS A PARTIR DA OBSERVA��O DE
        % INDIV�DUOS QU�NTICOS PARA CROSSOVER COM GAP DE B(T)
        
        % REAIS
        for j=1:gap
            for k=1:numGenesQreal
              classicQreal(j,k) = rand .* pulsesWidth(thePulses(1,j),k) + (pulsesCenter(thePulses(1,j),k) - (pulsesWidth(thePulses(1,j),k) / 2));
              %%% sigma * (y-1/2 + mi) = sigma*y + (mi-sigma/2)
              % CERTIFICA-SE QUE SORTEIOS EST�O DENTRO DOS LIMITES
              classicQreal(j,k) = min(classicQreal(j,k),upperverifica(j,k));
              classicQreal(j,k) = max(classicQreal(j,k),lowerverifica(j,k));
            end
        end
       
           
    end
    
    %EMBARALHA OS INDIV�DUOS QU�NTICOS GERADOS PARA CROSSOVER VISANDO AUMENTAR A DIVERSIDADE EM B(T)
    qwert = randperm(gap);
    classicQreal = classicQreal(qwert,:); 
    
    % Gera��o dos pesos associados aos indiv�duos cl�ssicos observados
    for j=1:gap
        
          [wo_in{j,1},wo_back{j,1},wo{j,1},posicoes{j,1}] = geraPesos(classicQreal(j,1),classicQreal(j,6),nInput,nOutput);
         
    end
   
    %FAZ CROSSOVER DE INDIV�DUOS OBSERVADOS COM GAP DE MELHORES INDIV�DUOS
%     if g==1
%         contadorQbit=zeros(numClassic,numGenesQbit);
%     end
    
    if (g > 1)

        % o gap de indiv�duos cl�ssicos selecionados para crossover deve
        % ser menor ou igual � popula��o de indiv�duos cl�ssicos
        if (gap>numClassic)
           error('O gap de indiv�duos cl�ssicos selecionados para crossover � maior que a pr�pria popula��o cl�ssica');
        end
             
        
        % M�SCARA DE CROSSOVER     
        
        maskxQreal = rand(gap, 4) <= crossover;
     
        new_Qreal = zeros(size(maskxQreal,1),size(maskxQreal,2));
             
        idxCrossover = 1:numClassic;
        tmpBestQreal = bestQreal(idxCrossover(1:gap),2:5);
        
        tmpClassicQreal = classicQreal(:,2:5);
        

        for j=1:gap
            
            % par�metros para o cruzamento
            
            distance = rand; % alfa do cruzamento aritm�tico
            escolhido_1=randi(gap); %  1� indiv�duo escolhido (P(t))
            escolhido_2=randi(gap); %  2� indiv�duo escolhido (B(t))
            mskReal = maskxQreal(j,:); % quais par�metros reais ser�o alterados
                                       
            % hora de cruzar
         
            new_Qreal (j,(mskReal == 1)) = tmpBestQreal(escolhido_2,(mskReal == 1)) + (tmpClassicQreal(escolhido_1,(mskReal == 1)) - tmpBestQreal(escolhido_2,(mskReal == 1))) * distance;
            new_Qreal (j,(mskReal == 0)) = tmpBestQreal(escolhido_2,(mskReal == 0));
            
            
           
            [new_wo{j,1},new_w_in{j,1},new_w_back{j,1},new_res(j,1),new_conec(j,1),new_posicoes{j,1}] = ...
            cruzaReservoir(bestWo{escolhido_2,1}, wo{escolhido_1,1}, bestQreal(escolhido_2,1), classicQreal(escolhido_1,1),bestQreal(escolhido_2,6),classicQreal(escolhido_1,6),...
            bestWo_in{escolhido_2,1},wo_in{escolhido_1,1},bestWo_back{escolhido_2,1},wo_back{escolhido_1,1},distance);
        end
            classicQreal(:,2:5) = new_Qreal; 
            classicQreal(:,1) = new_res;
            classicQreal(:,6) = new_conec;
            wo = new_wo;
            wo_in = new_w_in;
            wo_back = new_w_back;
            posicoes = new_posicoes;
            
            clear newQreal new_res new_conec new_wo new_w_in new_w_back new_posicoes 
             
    end
            clear mskReal idxCrossover maskxQreal tmpBestQreal tmpClassicQreal
    
    % AVALIA INDIV�DUOS AP�S CRUZAMENTO
    
    fitness = [];
   
    
       
    
    if g==1 
        for j=1:numClassic
         wo_v{j,1} = full(wo{j,1}(posicoes{j,1}));
         nInternal = ceil(classicQreal(j,1));
         pesos{j,1} = [reshape(wo_in{j,1},1,nInput.*nInternal)... 
         reshape(wo_back{j,1},1,nOutput.*nInternal) wo_v{j,1}'];
         classic{j,:} = [classicQreal(j,:) pesos{j,1}];
        end
        parfor j=1:numClassic
 
              fitness (j,1) = feval(fun,classic{j,1} ,varargin{:},nInput,nOutput,posicoes{j,1});
              
              if isnan(fitness(j,1))
                  fitness(j,1)=inf;
              end
        end
    else
        for j=1:gap
            wo_v{j,1} = full(wo{j,1}(posicoes{j,1}));         
            nInternal = ceil(classicQreal(j,1));
            pesos{j,1} = [reshape(wo_in{j,1},1,nInput.*nInternal)... 
            reshape(wo_back{j,1},1,nOutput.*nInternal) wo_v{j,1}'];
            classic{j,:} = [classicQreal(j,:) pesos{j,1}];
        end
        parfor j=1:gap
           
               fitness (j,1) = feval(fun,classic{j,1} ,varargin{:},nInput,nOutput,posicoes{j,1});
               if isnan(fitness(j,1))
                  fitness(j,1)=inf;
              end
        end
    end   

        
    %ATUALIZA POPULA��O B(T)
    
    classic  = [best; classic];
    fitness  = [bestFitness; fitness];
    posicoes = [posicoesBest;posicoes]; 
    wo       = [bestWo; wo];
    wo_in    = [bestWo_in; wo_in];
    wo_back  = [bestWo_back; wo_back];
    
    
    [x,idx] = sort(fitness,'ascend');
      
    
    %REMOVE INDIV�DUOS MAIS FRACOS (EXCEDENTES)
    
    classic = classic(idx(1:numClassic),:);

    posicoes = posicoes(idx(1:numClassic),:);
    
    fitness = fitness(idx(1:numClassic),:);
    wo = wo(idx(1:numClassic),:);
    wo_in = wo_in(idx(1:numClassic),:);
    wo_back = wo_back(idx(1:numClassic),:);
       
    for i = 1:numClassic
       classicQreal(i,:)= classic{i,1}(:,1:6);
    end
    best = classic;
    bestQreal=classicQreal;
     
    fitness = fitness(1:numClassic,:);
    posicoesBest = posicoes;
    bestWo = wo;
    bestWo_in = wo_in;
    bestWo_back = wo_back;
    bestFitness = fitness;
    evobestFitness = [evobestFitness; bestFitness(1)];
    
   
    % GUARDA MELHORES INDIV�DUOS
   
    bestexperimento = classic{1,:};
    posicoesBestExperimento = posicoes(1,:);
    bestexperimentoQreal = classicQreal(1,:);
    
    clear posicoes pesos wo wo_in wo_back


    % SALVA A GERA��O, M�DIA DO FITNESS DE B(T) E O MELHOR FITNESS DE B(T)
        
    if g==1
        trace=[g,mean(bestFitness),bestFitness(1,1)];
        bestInd=bestInd;
        else
        trace= [trace;[g,mean(bestFitness),bestFitness(1,1)]];
        bestInd = [bestInd;best(1,:)];
    end
    
    % REPOSICIONA PULSOS DE ACORDO COM MELHORES INDIV�DUOS
    
    if (rem(g, genToWidth) == 0)
        
        % QUANTUM CROSSOVER
        tmpRow = randperm(numPulses);
         
       % ATUALIZA CENTROS DOS PULSOS
       tmpClassicQreal = classicQreal(tmpRow,:);
  
       tmpCenter = pulsesCenter;
        
       maskdx = rand(size(tmpCenter,1),numGenesQreal) < qCrossover;
        
       pulsesCenter(maskdx) = tmpCenter(maskdx) + (tmpClassicQreal(maskdx) - tmpCenter(maskdx)) * distance;
       
        % ATUALIZA INTERVALOS DOS PULSOS
 
       
        tmpMaxClassicQreal=zeros(1,numGenesQreal);
        
        for j=1:numGenesQreal
            for k=1:size(tmpClassicQreal,1)
                
                if maskdx(k,j)==1
                        if k==1
                            tmpMaxClassicQreal(1,j)=tmpClassicQreal(k,j);
                        else    
                            if tmpMaxClassicQreal(1,j)<tmpClassicQreal(k,j);
                               tmpMaxClassicQreal(1,j)=tmpClassicQreal(k,j);
                            end
                        end
                end
                 
            end
        end
        
        tmpMinClassicQreal=zeros(1,numGenesQreal);
        
        for j=1:numGenesQreal
            for k=1:size(tmpClassicQreal,1)
                
                if maskdx(k,j)==1
                    if k==1
                        tmpMinClassicQreal(1,j)=tmpClassicQreal(k,j);
                    else    
                        if tmpMinClassicQreal(1,j)>=tmpClassicQreal(k,j);
                           tmpMinClassicQreal(1,j)=tmpClassicQreal(k,j);
                        end
                    end
                end
                 
            end
        end
        
        
        for j=1:numPulses
            for k=1:numGenesQreal
            tmpMaxClassicQrealLINHA(j,k)=tmpMaxClassicQreal(1,k);
            tmpMinClassicQrealLINHA(j,k)=tmpMinClassicQreal(1,k);
            end
        end
        
        tmpWidth=pulsesWidth;
        pulsesWidth(maskdx) = tmpWidth(maskdx) + ((tmpMaxClassicQrealLINHA(maskdx)-tmpMinClassicQrealLINHA(maskdx)) - tmpWidth(maskdx)) * distance;

        % COLOCA CENTROS E INTERVALOS DENTRO DOS SEUS LIMITES CORRETOS
        allLower = pulsesCenter - pulsesWidth / 2;
        allUpper = pulsesCenter + pulsesWidth / 2;

        allLower = max(allLower,lowerverificaP); %CORRIGIDO, ESTAVA TROCADO MIN E MAX
        allUpper = min(allUpper,upperverificaP);

        pulsesCenter = (allUpper + allLower) / 2;
        pulsesWidth = (allUpper - allLower);

        
    end
    
    % IMPRIME STATUS DA EVOLU��O
    if (showEv == 1)
           fprintf('Exp: %d - \t Ger: %d - \t Best: %10.5f - \t Mean: %10.5f - \t AQreal: %10.5f  \t \n', experimento, g, bestFitness(1), mean(bestFitness), mean(mean(pulsesWidth)));
    end
    
  
 end
