function [bestexperimento, classic,  evobestFitness, trace] = aeiq_br_real(param_real, numPulses, numClassic, ...
    genToWidth, generations, crossover, qCrossover, showEv,  gap, experimento,seed,fun,limInf,limSup,varargin)




%[qreal,qbit]=loadbanco(bandodedados,param_real);

%n_variaveis = size(bancodedados,2); % número de variáveis a serem agrupadas 
% n_variaveis = 0;

% qreal=[repmat(-6,param_real,1),repmat(6,param_real,1)];
qreal = [limInf limSup];
%%%qbit=[repmat(1/sqrt(2),n_variaveis,1),repmat(1/sqrt(2),n_variaveis,1)];

% número de genes, obtido pela matriz de intervalos qreal
numGenesQreal = size(qreal,1);
%%%numGenesQbit  = size(qbit,1);

% o número de sorteios para os indivíduos clássicos deve ser um múltiplo do
% número indíviduos quânticos
if (rem(numClassic, numPulses) ~= 0)
   error('O número de indivíduos clássicos deve ser múltiplo do número de individuos quânticos');
end

% REPRESENTAÇÃO INDIVÍDUO REAL

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

% REPRESENTAÇÃO INDIVÍDUO BINÁRIO

%limitesuperior
%%%upperQbit=[qbit(1:end,2)]';


%limiteinferior
%%%lowerQbit=[qbit(1:end,1)]';

% LISTA DE INDIVÍDUOS QUANTICOS REAIS

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

% LISTA DE INDIVÍDUOS QUANTICOS BINÁRIOS

%quantumUpper
%%%for j=1:numPulses
   %%% for k=1:numGenesQbit;
%%%quantumUpper(j,k)=upperQbit(1,k);
   %%% end
%%%end

%quantumLower
%%%for j=1:numPulses
   %%% for k=1:numGenesQbit;
%%%quantumLower(j,k)=lowerQbit(1,k);
  %%%  end
%%%end

% PARÂMETROS DE INTERESSE

bestInd = [];
bestFitness = [];
best = [];
bestQreal = [];
%%%bestQbit = [];
%%%bestcontadorQbit = [];
evobestFitness = [];

% INICIALIZA LOOPING

for g = 1:generations,
    
    % SE O GAP DE INDIVÍDUOS PARA CROSSOVER FOR MAIOR QUE INDIVIDUOS
    % QUÂNTICOS, ENTÃO, ALGUNS INDIVIDUOS QUANTICOS SERÃO SELECIONADOS MAIS
    % VEZES ALEATORIAMENTE PARA CROSSOVER. CASO CONTRÁRIO, ALGUNS
    % INDIVIDUOS QUANTICOS DEIXARÃO DE SEREM SELECIONADOS PARA CROSSOVER.
    
    if (gap <= numPulses)
        thePulses = randperm(numPulses);
        thePulses = thePulses(1:gap);
    else
        thePulses = [repmat(1:numPulses,[1 floor(gap / numPulses)]) (1:mod(gap, numPulses))];
    end
    
    if (g == 1) 
        
        % INICIALIZA POPULAÇÃO CLÁSSICA. INDIVÍDUOS POSSÍVEIS TEM A MESMA
        % PROBABILIDADE DE SEREM GERADOS.
        
        % gerando indivíduos randômicos
        classicQreal = rand(numClassic, numGenesQreal);
        % corrigindo no domínio
        classicQreal= classicQreal.* repmat(initialWidth,numClassic,1) + repmat(lowerQreal,numClassic,1);
        % fazendo o qbit
        %%%classicQbit = round(rand(numClassic, numGenesQbit)); 
             
    else
        
        % GAP DE INDIVÍDUOS CLÁSSICOS SÃO GERADOS A PARTIR DA OBSERVAÇÃO DE
        % INDIVÍDUOS QUÂNTICOS PARA CROSSOVER COM GAP DE B(T)
        
        % REAIS
        for j=1:gap
            for k=1:numGenesQreal
            classicQreal(j,k) = rand .* pulsesWidth(thePulses(1,j),k) + (pulsesCenter(thePulses(1,j),k) - (pulsesWidth(thePulses(1,j),k) / 2));
            %%% sigma * (y-1/2 + mi) = sigma*y + (mi-sigma/2)
            % CERTIFICA-SE QUE SORTEIOS ESTÃO DENTRO DOS LIMITES
            classicQreal(j,k) = min(classicQreal(j,k),upperverifica(j,k));
            classicQreal(j,k) = max(classicQreal(j,k),lowerverifica(j,k));
            end
        end
       
        % BINÁRIOS
        %%%for j=1:gap
           %%% for k=1:numGenesQbit
            %%%ProbabHumQbit(j,k)=quantumUpper(thePulses(1,j),k)^2;
            %%%ProbabZeroQbit(j,k)=quantumLower(thePulses(1,j),k)^2;
            %%%end   
        %%%end
       
        %%%for j=1:gap
           %%% for k=1:numGenesQbit
              %%%  if rand <= ProbabHumQbit(j,k)   
                   %%%classicQbit(j,k) = 1;
                %%%else
                   %%%classicQbit(j,k) = 0;    
                %%%end
            %%%end
        %%%end
    
    end
    
    %EMBARALHA OS INDIVÍDUOS QUÂNTICOS GERADOS PARA CROSSOVER VISANDO AUMENTAR A DIVERSIDADE EM B(T)
    qwert = randperm(gap);
    classicQreal = classicQreal(qwert,:); 
    %%%classicQbit  = classicQbit(qwert,:); 

    %FAZ CROSSOVER DE INDIVÍDUOS OBSERVADOS COM GAP DE MELHORES INDIVÍDUOS
%     if g==1
%         contadorQbit=zeros(numClassic,numGenesQbit);
%     end
    
    if (g > 1)

        % o gap de indivíduos clássicos selecionados para crossover deve
        % ser menor ou igual que a população de indivíduos clássicos
        if (gap>numClassic)
           error('O gap de indivíduos clássicos selecionados para crossover é maior que a própria população clássica');
        end
             
        
        % MÁSCARA DE CROSSOVER     
        
        maskxQreal = rand(gap, numGenesQreal) <= crossover;
       %%% maskxQbit  = rand(gap, numGenesQbit)  <= crossover;
        new_Qreal = zeros(size(maskxQreal,1),size(maskxQreal,2));
        %%%new_Qbit = zeros(size(maskxQbit,1),size(maskxQbit,2));
        
        idxCrossover = 1:numClassic;
        tmpBestQreal = bestQreal(idxCrossover(1:gap),:);
        %%%tmpBestQbit  = bestQbit(idxCrossover(1:gap),:);
        

%         SOMENTE OS PESOS E FUNÇÕES DE ATIVAÇÃO CUJOS NEURONIOS ESTEJAM ATIVADOS SERÃO ATUALIZADOS COM CROSSOVER.
        for j=1:gap
            
            % parâmetros para o cruzamento
            distance = rand; % alfa do cruzamento aritmético
            escolhido_1=randi(gap); %  1º indivíduo escolhido (P(t))
            escolhido_2=randi(gap); %  2º indivíduo escolhido (B(t))
            mskReal = maskxQreal(j,1:numGenesQreal); % quais parâmetros reais serão alterados
           %%% mskBin = maskxQbit(j,1:numGenesQbit); % quais parâmetros binários serão alterados
                            
            % hora de cruzar
            %new_Qreal(j,:)=(tmpBestQreal(escolhido_1,:).*(1-mskReal)+(tmpBestQreal(escolhido_1,:).*mskReal)*distance)+(tmpBestQreal(escolhido_2,:).*(1-mskReal)+(tmpBestQreal(escolhido_2,:).*mskReal)*(1-distance));  % parte real
            new_Qreal (j,(mskReal == 1)) = tmpBestQreal(escolhido_2,(mskReal == 1)) + (classicQreal(escolhido_1,(mskReal == 1)) - tmpBestQreal(escolhido_2,(mskReal == 1))) * distance;
            new_Qreal (j,(mskReal == 0)) = tmpBestQreal(escolhido_2,(mskReal == 0));
            %%%for bin=1:size(maskxQbit,2) % parte binária
               %%% if mskBin(bin)==1
                  %%%  new_Qbit(j,bin)=tmpBestQbit(escolhido_1,bin);
                %%%else
                   %%% new_Qbit(j,bin)=tmpBestQbit(escolhido_2,bin);
                %%%end
           
        end
            
          classicQreal=new_Qreal;
          %%%classicQbit=new_Qbit;
          
%             wi_mask = new_maskQreal(1:(ni)*nh); 
%             wo_mask = new_maskQreal((ni+1)*nh+1:(ni+1)*nh+nh*no);
%             wi_mask = new_maskQreal;
%             wo_mask = new_maskQreal;
%             acth_mask = new_maskxQbit;
% %             acth_mask = new_maskxQbit(ni+nh+1:ni+nh+nh);  
%             
%             for k=(ni+1):(ni+nh)
%                if new_tmpBestQbit(1,k)==0
%                    acth_mask(1,k-ni)=0;
%                end
%             end
%             
%             wi_mask = reshape(wi_mask,ni,nh); % pesos da primeira camada, na linha neuronios de entrada, na coluna neuronios na intermediaria
%             wo_mask = reshape(wo_mask,nh,no); % matriz de dimensao neuronios na camada escondida por linha e uma coluna (somente um neuronio na saida)
%              
%             aiBinario_tmpBest = new_tmpBestQbit(1:ni);
%             ahBinario_tmpBest = new_tmpBestQbit(ni+1:ni+nh);            
%                        
%             for l=1:(ni)
%                 if aiBinario_tmpBest(1,l)==0
%                     for k=1:nh
%                     wi_mask(l,k)=0;
%                     end
%                 end
%             end
%             
%             for l=1:nh
%                 if ahBinario_tmpBest(1,l)==0
%                     for k=1:(ni)
%                     wi_mask(k,l)==0;
%                     end
%                 end
%             end
%             
%             for l=1:nh
%                 if ahBinario_tmpBest(1,l)==0
%                     for k=1:no
%                     wo_mask(l,k)=0;
%                     end
%                 end
%             end
%             
%             wi_mask = reshape(wi_mask,1,(ni)*nh);
%             wo_mask = reshape(wo_mask,1,nh*no); 
%             
%             new_maskQreal(1:(ni)*nh) = wi_mask;
%             new_maskQreal((ni+1)*nh+1:(ni+1)*nh+nh*no) = wo_mask ;
%             
%             new_maskxQbit(ni+nh+1:ni+nh+nh) = acth_mask;
%             
%             maskxQreal(j,1:numGenesQreal) = new_maskQreal;
%             maskxQbit(j,1:numGenesQbit) =  new_maskxQbit;
%             
%         end            
%         
%         classicQreal(maskxQreal == 1) = tmpBestQreal(maskxQreal == 1) + (classicQreal(maskxQreal == 1) - tmpBestQreal(maskxQreal == 1)) * distance;
%         classicQreal(maskxQreal == 0) = tmpBestQreal(maskxQreal == 0);
%         
%         wi_mask = new_maskQreal(1:(ni)*nh); 
%         wo_mask = new_maskQreal((ni+1)*nh+1:(ni+1)*nh+nh*no);
%  
%         classicQbit(maskxQbit == 1) = tmpBestQbit(maskxQbit == 1);
%         classicQbit(maskxQbit == 0) = classicQbit(maskxQbit == 0);
%         
%         % SE O NEURONIO ESTAVA ATIVO A MENOS QUE numRODADAS, E ESTE FOI
%         % DESATIVADO PELO CROSSOVER, ESTE SERA NOVAMENTE ATIVADO PERMITINDO QUE
%         % HAJAM MAIS ATUALIZAÇÕES DE PESOS
%         
%             tmpbestcontadorQbit=bestcontadorQbit(idxCrossover(1:gap),:);
%         
%             for j=1:size(tmpBestQbit,1)
%                 for k=1:(ni+nh)
%                 %   
%                     if tmpBestQbit(j,k)==1
%                         tmpbestcontadorQbit(j,k)=tmpbestcontadorQbit(j,k)+1;
%                         %
%                         if classicQbit(j,k)==0
%                            if tmpbestcontadorQbit(j,k)<=numRODADAS
%                                classicQbit(j,k)=tmpBestQbit(j,k);
%                            end
%                         end
%                         %
%                     elseif tmpBestQbit(j,k)==0
%                         %
%                            tmpbestcontadorQbit(j,k)=0;
%                         %
%                     end
%                 %    
%                 end
%             end
%             
%             contadorQbit=tmpbestcontadorQbit;
%         
%             for j=1:numClassic
%                 for k=1:numGenesQbit
%                     bestcontadorQbit(j,k)=bestcontadorQbit(j,k)+1;
%                 end
%             end
%                 
    end
    
    % AVALIA INDIVÍDUOS APÓS CRUZAMENTO
    
    fitness = [];
   %%% classic=[classicQreal,classicQbit];
    classic = [classicQreal];
    %corrigido=zeros(size(classic,1),size(classic,2));
    
    
    if g==1 
        parfor j=1:numClassic
              fitness (j,1) = feval(fun,classic(j,1:param_real) ,varargin{:});
              if isnan(fitness(j,1))
                  fitness(j,1)=inf;
              end
        end
    else
        parfor j=1:gap
               fitness (j,1) = feval(fun,classic(j,1:param_real) ,varargin{:});
               if isnan(fitness(j,1))
                  fitness(j,1)=inf;
               end
        end
    end   

        
    %ATUALIZA POPULAÇÃO B(T)
    
    classic = [best; classic];
    fitness = [bestFitness; fitness];
%%%    contadorQbit = [bestcontadorQbit; contadorQbit];
    %%%[x,idx] = sort(fitness,'descend');
    [x,idx] = sort(fitness,'ascend');
    
    classic = classic(idx,:);
    fitness = fitness(idx,:);
%%%    contadorQbit = contadorQbit(idx,:);
    
    %REMOVE INDIVÍDUOS MAIS FRACOS (EXCEDENTES)

    classic = classic(1:numClassic,:);
    classicQreal=classic(1:numClassic,1:numGenesQreal);
    %%%classicQbit=classic(1:numClassic,numGenesQreal+1:numGenesQreal+numGenesQbit);
%%%    contadorQbit=contadorQbit(1:numClassic,:);
    
    best = classic;
    bestQreal=classicQreal;
   %%% bestQbit=classicQbit;
%%%    bestcontadorQbit=contadorQbit;
    
    fitness = fitness(1:numClassic,:);
    bestFitness = fitness;
    evobestFitness = [evobestFitness; bestFitness(1)];

    % GUARDA MELHORES INDIVÍDUOS
   
    bestexperimento = classic(1,:);
    bestexperimentoQreal = classic(1,1:numGenesQreal);
   %%% bestexperimentoQbit = classic(1,numGenesQreal+1:numGenesQreal+numGenesQbit);

    % SALVA A GERAÇÃO, MÉDIA DO FITNESS DE B(T) E O MELHOR FITNESS DE B(T)
        
    if g==1
        trace=[g,mean(bestFitness),bestFitness(1,1)];
        bestInd=bestInd;
        else
        trace= [trace;[g,mean(bestFitness),bestFitness(1,1)]];
        bestInd = [bestInd;best(1,:)];
    end
    
    % REPOSICIONA PULSOS DE ACORDO COM MELHORES INDIVÍDUOS
    
    if (rem(g, genToWidth) == 0)
        
        % QUANTUM CROSSOVER
        tmpRow = randperm(numPulses);
         
       % ATUALIZA CENTROS DOS PULSOS
       tmpClassicQreal = classicQreal(tmpRow,:);
     %%%  tmpClassicQbit  = classicQbit(tmpRow,:);
       tmpCenter = pulsesCenter;
        
       maskdx = rand(size(tmpCenter,1),numGenesQreal) < qCrossover;
        
       
        % SOMENTE OS PESOS CUJOS NEURONIOS ESTEJAM ATIVADOS SERÃO UTILIZADOS PARA ATUALIZAR OS CENTROS.
            
%         for j=1:size(tmpCenter,1)
%             
%             new_maskdx = maskdx(j,1:numGenesQreal);
%             new_tmpClassicQbit = tmpClassicQbit(j,1:numGenesQbit);
% 
%             wi_maskdx = new_maskdx(1:(ni)*nh);
%             wo_maskdx = new_maskdx((ni+1)*nh+1:(ni+1)*nh+nh*no);
%             
%             wi_maskdx = reshape(wi_maskdx,ni,nh); % pesos da primeira camada, na linha neuronios de entrada, na coluna neuronios na intermediaria
%             wo_maskdx = reshape(wo_maskdx,nh,no); % matriz de dimensao neuronios na camada escondida por linha e uma coluna (somente um neuronio na saida)
%   
%             aiBinario_tmpClassicQbit = new_tmpClassicQbit(1:ni);
%             ahBinario_tmpClassicQbit = new_tmpClassicQbit(ni+1:ni+nh);            
%                        
%             for l=1:(ni)
%                 if aiBinario_tmpClassicQbit(1,l)==0
%                     for k=1:nh
%                     wi_maskdx(l,k)==0;
%                     end
%                 end
%             end
%             
%             for l=1:nh
%                 if ahBinario_tmpClassicQbit(1,l)==0
%                     for k=1:(ni)
%                     wi_maskdx(k,l)==0;
%                     end
%                 end
%             end
%             
%             for l=1:nh
%                 if ahBinario_tmpClassicQbit(1,l)==0
%                     for k=1:no
%                     wo_maskdx(l,k)==0;
%                     end
%                 end
%             end
%             
%             wi_maskdx = reshape(wi_maskdx,1,(ni)*nh);
%             wo_maskdx = reshape(wo_maskdx,1,nh*no); 
%             
%             new_maskdx(1:(ni)*nh) = wi_maskdx;
%             new_maskdx((ni+1)*nh+1:(ni+1)*nh+nh*no) = wo_maskdx ;
%             
%             maskdx(j,1:numGenesQreal) = new_maskdx;
%             
%         end 
        
        pulsesCenter(maskdx) = tmpCenter(maskdx) + (tmpClassicQreal(maskdx) - tmpCenter(maskdx)) * distance;
       
% ATUALIZA INTERVALOS DOS PULSOS
 
%    % ATUALIZAÇÃO DOS INTEVALOS QUÂNTICOS POR DISTÂNCIA RELATIVA,
%    % NORMALIZADA DO INDIVÍDUO COMPARA E VALOR OBJETIVO
%    if (g > 1)
%         if (options(2)==1)
%             compareValue = mean(bestFitness);
%         elseif (options(2)==2)
%             compareValue = median(bestFitness);
%         else
%         compareValue = bestFitness(1,1);
%         end
%     end  

%     % LARGURA DOS PULSOS OU É DECRESCIDA LINEARMENTE, OU É ATUALIZA DE
%     % ACORDO COM COMPARA
%    
%     if options(1)==1    
%         
%     % DECRESCIMENTO LINEAR
%       
%                for k=1:numPulses
%                   for j=1:numGenesQreal
%                   pulsesWidth(k,j) = pulsesWidth(k,j) - (initialWidth(1,j) ./ ((generations/genToWidth)-1));
%                   end
%                end
%     
%     else 
%          
%    % AMPLITUDE RELATIVA NORMALIZADA
%      
%         if compareValue>options(3)
%            options(3)=compareValue*2;
%         end              
%     
%         for k=1:numPulses
%         for j=1:numGenesQreal
%         pulsesWidth(k,j) = initialWidth(1,j) - ((initialWidth(1,j)*compareValue) ./ options(3));
%         end
%         end
%         
%     end   

        % DA MESMA FORMA, SOMENTE OS PESOS CUJOS NEURONIOS ESTEJAM ATIVADOS SERÃO
        % UTILIZADOS PARA ATUALIZAR AS AMPLITUDES
        
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
        
%         tmpMaxClassicQreal=max(tmpClassicQreal);
%         tmpMinClassicQreal=min(tmpClassicQreal);
        
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

        % Q-GATE
       %%% tmpClassicQbit = classicQbit(tmpRow,:);

        %%%for j=1:numPulses
        %%%    for k=1:numGenesQbit
         %%%       if tmpClassicQbit(j,k) == 1
          %%%          DeltaTeta(j,k)=(-1)*deltateta;
           %%%     else        
           %%%         DeltaTeta(j,k)=deltateta;
          %%%      end         
         %%%   end             
        %%%end

      %%%  quantumUpper_before=quantumUpper;
     %%%   quantumLower_before=quantumLower;
            
      %%%  for j=1:numPulses
        %%%    for k=1:numGenesQbit
         
        %%%    U=[[cos(DeltaTeta(j,k)) -sin(DeltaTeta(j,k))];[sin(DeltaTeta(j,k)) cos(DeltaTeta(j,k))]]; % 2 x 2
         %%%   atualiza=[quantumUpper(j,k)',quantumLower(j,k)']'; % 2 x 1
        %%%    atualiza=U*atualiza; % 2 x 1
          %%%  quantumUpper(j,k)=atualiza(1,1);
         %%%   quantumLower(j,k)=atualiza(2,1);

         %%%   end
        %%%end
        
      % SOMENTE AS FUNÇOES DE ATIVAÇOES CUJOS NEURONIOS ESTEJAM ATIVO SERAO
      % ATUALIZADAS NO CROSSOVER QUANTICO
      
%       for j=1:numPulses
%             for k=(ni+nh+1):(ni+nh+nh)
%                 if tmpClassicQbit(j,(k-ni-nh))==0
%                     quantumUpper(j,k)=quantumUpper_before(j,k);
%                     quantumLower(j,k)=quantumLower_before(j,k);
%                 end
%             end
%       end
%       
%       for j=1:numPulses
%             for k=(ni+nh+nh+1):(ni+nh+nh+no)
%                 if tmpClassicQbit(j,(k-ni-nh))==0
%                     quantumUpper(j,k)=quantumUpper_before(j,k);
%                     quantumLower(j,k)=quantumLower_before(j,k);
%                 end
%             end
%       end

        
        
    end
    
    % CALCULA A MÁXIMA PROBABILIDAE PARA TODOS OS GENES Q-BIT DE TODOS OS
    % INDIVÍVÍDUOS QUÂNTICOS
    
    %%%for j=1:numPulses
    %%%   for k=1:numGenesQbit
    %%%       MaxProbab(j,k)=max(quantumUpper(j,k)^2,quantumLower(j,k)^2);
    %%%   end
    %%%end

    % IMPRIME STATUS DA EVOLUÇÃO
    if (showEv == 1)
            %%%fprintf('Exp: %d - \t Ger: %d - \t Best: %10.5f - \t Mean: %10.5f - \t AQreal: %10.5f  \t AQbit: %10.5f \n', experimento, g, bestFitness(1), mean(bestFitness), mean(mean(pulsesWidth)), mean(mean(MaxProbab)));
            fprintf('Exp: %d - \t Ger: %d - \t Best: %10.5f - \t Mean: %10.5f - \t AQreal: %10.5f  \t \n', experimento, g, bestFitness(1), mean(bestFitness), mean(mean(pulsesWidth)));
    end
    
  
end
