clear
model_name  = 'cGAN_5pic_2y_train_2.3';

cd '/home/lab/orel_ws/project/model_training/'
files = strsplit(ls(model_name + "/losses*.mat"),{'\n',' '});
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
% set(gca,'Xtick', 0:10:150)
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
% set(gca,'Xtick', 0:10:150)
grid on

% figure(5)
% plot(smooth(last.Reff_disc_loss,50))
% xlabel('Epoch','Interpreter', 'latex'); ylabel('${J_{D_{reff}}}$','Interpreter', 'latex')
% set(gcf,'Position', size)
% grid on

figure(6)
plot(last.Reff_disc_loss)
xlabel('Epoch','Interpreter', 'latex'); ylabel('${J_{D_{reff}}}$','Interpreter', 'latex')
set(gcf,'Position', size)
grid on


lr_files = strsplit(ls(model_name + "/lr_rates*.mat"),{'\n', ' '});
lr_rates = load(lr_files{end-1});
lr_rates.title = strjoin(strsplit(lr_files{end-1},'_'));
% lr_rates_2 = load(lr_files{end-2});
% lr_rates_2.title = strjoin(strsplit(lr_files{end-2},'_'));

figure(7)
plot(lr_rates.gen_lr)
hold on
plot(lr_rates.disc_lr)
legend('Gen LR','Disc LR'); grid on; %title(lr_rates.title);
set(gcf,'Position', size)
xlabel('Epoch','Interpreter', 'latex'); ylabel('Learning rate','Interpreter', 'latex')
hold off



% figure(8)
% plot(lr_rates_2.gen_lr)
% hold on
% plot(lr_rates_2.disc_lr)
% legend('Gen LR','Disc LR'); grid on; title(lr_rates_2.title);
% set(gcf,'Position', size)
% xlabel('Epoch','Interpreter', 'latex'); ylabel('Learning rate','Interpreter', 'latex')
% hold off