% pre processamento dados secador

clear all
close all
clc

load steamgen.dat

u1 = steamgen(:,2);
u2 = steamgen(:,3);
u3 = steamgen(:,4);
u4 = steamgen(:,5);

y1 = steamgen(:,6);
y2 = steamgen(:,7);
y3 = steamgen(:,8);
y4 = steamgen(:,9);


uTreino = [u1(1:5600) u2(1:5600) u3(1:5600) u4(1:5600)] ;
uValidacao = [u1(5601:7600) u2(5601:7600) u3(5601:7600) u4(5601:7600)];
uTeste = [u1(7601:end) u2(7601:end) u3(7601:end) u4(7601:end)];

uMin = min ([uTreino; uValidacao; uTeste]);
%uMin = 0;
uMax = max ([uTreino; uValidacao; uTeste]);
%uMax = 1;

trainInputSequence = normaliza  (uTreino,uMin,uMax);
valInputSequence = normaliza  (uValidacao,uMin,uMax);
testInputSequence = normaliza  (uTeste,uMin,uMax);

yTreino = [y1(1:5600) y2(1:5600) y3(1:5600) y4(1:5600)] ;
yValidacao = [y1(5601:7600) y2(5601:7600) y3(5601:7600) y4(5601:7600)];
yTeste = [y1(7601:end) y2(7601:end) y3(7601:end) y4(7601:end)];

yMin = min ([yTreino; yValidacao; yTeste]);
yMax = max ([yTreino; yValidacao; yTeste]);

% yMin = 0; % não normaliza
% yMax = 1; % não normaliza

trainOutputSequence = normaliza  (yTreino,yMin,yMax);
valOutputSequence = normaliza  (yValidacao,yMin,yMax);
testOutputSequence = normaliza  (yTeste,yMin,yMax);

clear u y

save steamNorm

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
