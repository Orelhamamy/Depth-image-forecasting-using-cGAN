model_name  = 'cGAN_5pic_1y_train_1.4';

cd '/home/lab/orel_ws/project/model_training/'
files = strsplit(ls(model_name + "/losses*.mat"),'\n');
set(groot,'defaultAxesTickLabelInterpreter','latex');
last = load(files{end-1},'-mat');
size= [680.0000  697.3465  362.2047  271.6535]
figure(1)
plot(last.Gen_total_loss)
xlabel('Epoch','Interpreter', 'latex'); ylabel('${J_{G^*}}$','Interpreter', 'latex')
set(gcf,'Position', size)
grid on


figure(2)
plot(last.Gen_loss)
xlabel('Epoch','Interpreter', 'latex'); ylabel('${J_G}$','Interpreter', 'latex')
set(gcf,'Position', size)
grid on

figure(3)
plot(last.Gen_l1_loss)
xlabel('Epoch','Interpreter', 'latex'); ylabel('${J_{L1}}$','Interpreter', 'latex')
set(gcf,'Position', size)
grid on

figure(4)
plot(last.Disc_loss)
xlabel('Epoch','Interpreter', 'latex'); ylabel('${J_{D}}$','Interpreter', 'latex')
set(gcf,'Position', size)
grid on

figure(5)
plot(smooth(last.Reff_disc_loss,50))
xlabel('Epoch','Interpreter', 'latex'); ylabel('${J_{D_{reff}}}$','Interpreter', 'latex')
set(gcf,'Position', size)
grid on
