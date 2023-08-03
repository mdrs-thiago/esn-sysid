% pre processamento dados secador

clear all
close all
clc

load CD_player_arm.dat

u1 = CD_player_arm(:,1);
u2 = CD_player_arm(:,2);

y1 = CD_player_arm(:,3);
y2 = CD_player_arm(:,4);




uTreino = [u1(1:200) u2(1:200)] ;
uValidacao = [u1(201:500) u2(201:500)];
uTeste = [u1(501:2000) u2(501:2000)];

uMin = min ([uTreino; uValidacao]);
uMin = uMin - 0.2*uMin;

uMax = max ([uTreino; uValidacao]);
uMax = uMax + 0.2*uMax;

trainInputSequence = normaliza  (uTreino,uMin,uMax);
valInputSequence = normaliza  (uValidacao,uMin,uMax);
testInputSequence = normaliza  (uTeste,uMin,uMax);

yTreino = [y1(1:200) y2(1:200)] ;
yValidacao = [y1(201:500) y2(201:500)];
yTeste = [y1(501:2000) y2(501:2000)];

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

save CDNorm

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
