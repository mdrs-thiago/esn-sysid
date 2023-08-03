% function sortedCell sortCell(cell,idx)

clc
clear all
close all
    cell{1,1} = ones(1,56);
    cell{2,1} = ones(1,37);
    cell{3,1} = ones(4,26);
    cell{4,1} = ones(1,1);
    cell{5,1} = ones(1,75);
    cell{6,1} = ones(10,15);
    cell{7,1} = ones(4,35);
    cell{8,1} = ones(12,1);
    cell{9,1} = ones(7,5);
    idx = [9; 2; 4; 1; 7; 5; 3; 8; 6];



    tamanho = size(idx,1);

    for i=1:tamanho 
        
     sortedCell{i,1} = cell{idx(i),1};        
             
    end

    


% end