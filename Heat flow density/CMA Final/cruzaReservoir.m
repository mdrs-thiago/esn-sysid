function [w,w_in,w_back,N,conectividade,posicoes] = cruzaReservoir(w_a, w_b, res_a, res_b,conect_a,conect_b,w_in_a,w_in_b,w_back_a,w_back_b,lambda)

msk = randi([0,1],1);

N_a = ceil(res_a);
N_b = ceil(res_b);

minN = min([N_a N_b]);
maxN = max([N_a N_b]);

posMatchM = logical(sparse(maxN,maxN));
posMatchM(1:minN,1:minN) = (w_a(1:minN,1:minN)~=0)&(w_b(1:minN,1:minN)~=0);
posMatchN = (w_a(1:minN,1:minN)~=0)&(w_b(1:minN,1:minN)~=0);


if(N_a>=N_b)
    
   posMatchA = posMatchM;
   posMatchB = posMatchN;
   
else
    
   posMatchA = posMatchN;
   posMatchB = posMatchM; 
    
end


if (msk==0) % tamanho B(t)
    w = w_a;
    w(posMatchA) = w_a(posMatchA)+ lambda * (w_a(posMatchA)-w_b(posMatchB));
    conectividade = conect_a;
    N = res_a;
    w_in = w_in_a;
    w_back = w_back_a;
    
else   % tamanho P(t)
    w = w_b;
    w(posMatchB) = w_a(posMatchA)+ lambda * (w_a(posMatchA)-w_b(posMatchB));
    conectividade = conect_b;
    N = res_b;
    w_in = w_in_b;
    w_back = w_back_b;
end
   
   posicoes = (w~=0);
   
   w_in(1:minN,:) = w_in_a(1:minN,:)+ lambda * (w_in_a(1:minN,:)- w_in_b(1:minN,:));
   
   w_back(1:minN,:) = w_back_a(1:minN,:)+ lambda * (w_back_a(1:minN,:)- w_back_b(1:minN,:));
   
   

end
