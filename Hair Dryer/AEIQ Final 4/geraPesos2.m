function [wo_in,wo_back,wo,posicoes] = geraPesos2(nInternal,r,connectivity,nInput,nOutput)
  
  nInternal = ceil(nInternal);

  wo_in = 2.0 * rand(nInternal, nInput)- 1.0;
  
  wo_back = 2.0 * rand(nInternal, nOutput)- 1.0;

  success = 0 ;                                               
    while success == 0
    % following block might fail, thus we repeat until we obtain a valid
    % internalWeights matrix
    try,
        wo = 2*r*sprand(nInternal,nInternal,connectivity);
        wo(wo ~= 0) = ...
        wo(wo ~= 0)  - r;
        success = 1 ;
    catch,
        success = 0 ; 
    end
 
    posicoes = (wo~=0);
%     wo = full(wo(posicoes));
%     
%     pesos = [reshape(wo_in,1,nInput.*nInternal)... 
%          reshape(wo_back,1,nOutput.*nInternal) wo'];


end

