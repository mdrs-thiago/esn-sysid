% pre processamento dados secador

clear all
close all
clc

load thermic_res_wall.dat

u1 = thermic_res_wall(:,1);
u2 = thermic_res_wall(:,2);

y1 = thermic_res_wall(:,3);


uTreino = [u1(1:200) u2(1:200)] ;
uValidacao = [u1(201:500) u2(201:500)];
uTeste = [u1(501:1500) u2(501:1500)];

uMin = min ([uTreino; uValidacao]);
uMin = uMin - 0.2*uMin;

uMax = max ([uTreino; uValidacao]);
uMax = uMax + 0.2*uMax;

trainInputSequence = normaliza  (uTreino,uMin,uMax);
valInputSequence = normaliza  (uValidacao,uMin,uMax);
testInputSequence = normaliza  (uTeste,uMin,uMax);

yTreino = [y1(1:200)] ;
yValidacao = [y1(201:500)];
yTeste = [y1(501:1500)];

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

save heatflowNorm

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
