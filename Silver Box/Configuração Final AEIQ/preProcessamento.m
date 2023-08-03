% pre processamento dados secador

clear all
close all
clc

load SilverBox.mat

u = V1';
y = V2';


uTreino = [u(40001:91072)] ;
uValidacao = [u(91073:end)];
uTeste = [u(1:40000)];

uMin = min ([uTreino; uValidacao]);
uMin = uMin - 0.2*uMin;
%uMin = 0;
uMax = max ([uTreino; uValidacao]);
uMax = uMax + 0.2*uMax;
%uMax = 1;

trainInputSequence = normaliza  (uTreino,uMin,uMax);
valInputSequence = normaliza  (uValidacao,uMin,uMax);
testInputSequence = normaliza  (uTeste,uMin,uMax);

yTreino = [y(40001:91072)] ;
yValidacao = [y(91073:end)];
yTeste = [y(1:40000)];

yMin = min ([yTreino; yValidacao]);
yMin = yMin - 0.2*yMin;

yMax = max ([yTreino; yValidacao]);
yMax = yMax + 0.2*yMax;

% yMin = 0; % não normaliza
% yMax = 1; % não normaliza

trainOutputSequence = normaliza  (yTreino,yMin,yMax);
valOutputSequence = normaliza  (yValidacao,yMin,yMax);
testOutputSequence = normaliza  (yTeste,yMin,yMax);

clear u y

save silverNorm

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
% title('Validação','FontSize',14);
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
% title('Validação','FontSize',14);
% subplot(3,1,3);
% plot(yTeste,'b');
% hold on
% legend('Real','Previsto');
% xlabel('Amostras','FontSize',14);
% ylabel('y(k) ','FontSize',14);
% title('Teste','FontSize',14);
