 function sortedCell = sortCell(cell,idx)




    tamanho = size(idx,1);

    for i=1:tamanho 
        
     sortedCell{i,1} = cell{idx(i),1};        
             
    end

    


 end