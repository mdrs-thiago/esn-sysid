clc
clear all
close all


a(1,1) = 2;
a(1,3) = 1;
a(1,4) = 3;
a(2,2) = 1;
a(2,3) = 5;
a(3,2) = 2;
a(3,4) = 3;
a(4,4) = 5;

a = sparse (a);
n_a = 4;
conec_a = 0.5;


b(1,1) = 1;
b(1,2) = 2;
b(2,3) = 1;
b(3,1) = 2;
b(4,4) = 4;

b = sparse(b);
n_b = 4;
conec_b = 0.44;


W_in_a = [1 5; 2 6; 3 7; 4 8];
W_in_b = [9 12 ; 10 13; 11 14; 15 16];
w_back_a = [1 2 3; 4 5 6; 7 8 9; 10 11 12];
w_back_b = [13 14 17; 15 16 18; 19 20 21; 22 23 24];

[w,w_in,w_back,N,conectividade,posicoes] = cruzaReservoir(a, b, n_a, n_b,conec_a,conec_b,W_in_a,W_in_b,w_back_a, w_back_b,0.3);

w_in 

w

w_back