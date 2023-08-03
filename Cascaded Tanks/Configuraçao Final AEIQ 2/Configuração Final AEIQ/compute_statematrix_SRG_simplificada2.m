function stateCollectMat = ...
    compute_statematrix_SRG_simplificada2(inputSequence, outputSequence, esn, nForgetPoints, initialState,varargin)


    nDataPoints = length(outputSequence(:,1));

    stateCollectMat = ...
        zeros(nDataPoints - nForgetPoints,esn.nInternalUnits) ; 
    
%     totalstate = zeros(esn.nInputUnits + esn.nInternalUnits + esn.nOutputUnits, 1);
    totalstate = initialState';
%     zeroState = zeros(esn.nInternalUnits, 1);
    zeroState = initialState(1:esn.nInternalUnits)';
    typeSpecificArg = [];

    collectIndex = 0;
    for i = 1:nDataPoints
   
       in = esn.inputScaling .* inputSequence(i,:)' + esn.inputShift;  % in is column vector
      
       totalstate(esn.nInternalUnits+1:esn.nInternalUnits + esn.nInputUnits) = in;

       internalState = feval(esn.type, totalstate, esn, typeSpecificArg) ; 
    
       netOut = esn.teacherScaling .* outputSequence(i,:)' + esn.teacherShift;
  
       totalstate = [zeroState; in; netOut];
       if (i > nForgetPoints)
          collectIndex = collectIndex + 1;
          stateCollectMat(collectIndex,:) = [internalState']; 
       end % fim if
    end % fim loop datapoints
end