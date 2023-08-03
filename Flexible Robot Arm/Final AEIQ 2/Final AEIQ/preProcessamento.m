% pre processamento dados secador

clear all
close all
clc

load robot_arm.dat

u = robot_arm(:,1);
y = robot_arm(:,2);


uTreino = [u(1:312)] ;
uValidacao = [u(313:512)];
uTeste = [u(513:end)];

uMin = min ([uTreino; uValidacao]);
uMin = uMin - 0.2*uMin;
%uMin = 0;
uMax = max ([uTreino; uValidacao]);
uMax = uMax + 0.2*uMax;
%uMax = 1;

trainInputSequence = normaliza  (uTreino,uMin,uMax);
valInputSequence = normaliza  (uValidacao,uMin,uMax);
testInputSequence = normaliza  (uTeste,uMin,uMax);

yTreino = [y(1:312)] ;
yValidacao = [y(313:512)];
yTeste = [y(513:end)];

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

save robotNorm

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
