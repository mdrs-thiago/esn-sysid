
clc
clear all
close all

mskR = 1;

mskC = 0;


a(1,1) = 0;
a(1,2) = 1;
a(1,3) = 2;
a(2,1) = 0;
a(2,2) = 0;
a(2,3) = 2;
a(3,1) = 2;
a(3,2) = 0;
a(3,3) = 2;

a = sparse(a);

posicoesA = (a~=0);
    a_v = full(a(posicoesA));

b(1,1) = 4;
b(1,2) = 3;
% b(1,3) = 4;
b(2,1) = 0;
b(2,2) = 4;
% b(2,3) = 0;
% b(3,1) = 0;
% b(3,2) = 4;
% b(3,3) = 4;

b = sparse(b);

posicoesB = (b~=0);
    b_v = full(b(posicoesB));

[minN, posMinN] = min([size(a,1) size(b,1)]);
[maxN, posMaxN] = max([size(a,1) size(b,1)]);

posMatchM = logical(sparse(maxN,maxN));
posMatchM(1:minN,1:minN) = (a(1:minN,1:minN)~=0)&(b(1:minN,1:minN)~=0)
posMatchN = (a(1:minN,1:minN)~=0)&(b(1:minN,1:minN)~=0)

if(size(a,1)>=size(b,1))
    
   posMatchA = posMatchM;
   posMatchB = posMatchN;
   
else
    
   posMatchA = posMatchN;
   posMatchB = posMatchM; 
    
end


if (mskR==0) % tamanhoBest
    c = a;
    if (mskC==0) % mantém conectividade
        c(posMatchA) = a(posMatchA)-b(posMatchB);
    end
else
    c = b;
    if (mskC==0) % mantém conectividade
        c(posMatchB) = a(posMatchA)-b(posMatchB);
    end
end
    c
