% pre processamento dados secador

clear all
close all
clc

load Tank1.mat

u = u';
y1 = y(1,:)';
y2 = y(2,:)';


uTreino = [u(1:750)] ;
uValidacao = [u(751:1250)];
uTeste = [u(1251:end)];

uMin = min ([uTreino; uValidacao]);
uMin = uMin - 0.2*uMin;
%uMin = 0;
uMax = max ([uTreino; uValidacao]);
uMax = uMax + 0.2*uMax;
%uMax = 1;

trainInputSequence = normaliza  (uTreino,uMin,uMax);
valInputSequence = normaliza  (uValidacao,uMin,uMax);
testInputSequence = normaliza  (uTeste,uMin,uMax);

yTreino = [y1(1:750) y2(1:750)] ;
yValidacao = [y1(751:1250) y2(751:1250)];
yTeste = [y1(1251:end) y2(1251:end)];

yMin = min ([yTreino; yValidacao]);
yMin = yMin - 0.2*yMin;

yMax = max ([yTreino; yValidacao]);
yMax = yMax + 0.2*yMax;

% yMin = 0; % n�o normaliza
% yMax = 1; % n�o normaliza

trainOutputSequence = normaliza  (yTreino,yMin,yMax);
valOutputSequence = normaliza  (yValidacao,yMin,yMax);
testOutputSequence = normaliza  (yTeste,yMin,yMax);

clear u y

save tankNorm

% subplot(3,1,1);
% plot(uTreino,'b');
% hold on
% legend('Real','Previsto');
% xlabel('Amostras','FontSize',14);
% ylabel('u(k) ','FontSize',14);
% title('Treinamento','FontSize',14);
% subplot(3,1,2);
% plot(uValidacao,'b');
% legend('Real','Previsto');
% xlabel('Amostras','FontSize',14);
% ylabel('u(k) ','FontSize',14);
% title('Valida��o','FontSize',14);
% subplot(3,1,3);
% plot(uTeste,'b');
% hold on
% legend('Real','Previsto');
% xlabel('Amostras','FontSize',14);
% ylabel('u(k) ','FontSize',14);
% title('Teste','FontSize',14);
% 
% figure;
% subplot(3,1,1);
% plot(yTreino,'b');
% hold on
% legend('Real','Previsto');
% xlabel('Amostras','FontSize',14);
% ylabel('y(k) ','FontSize',14);
% title('Treinamento','FontSize',14);
% subplot(3,1,2);
% plot(yValidacao,'b');
% legend('Real','Previsto');
% xlabel('Amostras','FontSize',14);
% ylabel('y(k) ','FontSize',14);
% title('Valida��o','FontSize',14);
% subplot(3,1,3);
% plot(yTeste,'b');
% hold on
% legend('Real','Previsto');
% xlabel('Amostras','FontSize',14);
% ylabel('y(k) ','FontSize',14);
% title('Teste','FontSize',14);
