% This script plot the loss functions values and LR during the
% model training. NOTE that you need to fill in the model's name,
% and change the current folder to models folder.

clear
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
model_name  = 'cGAN_5pic_1y_train';
% cd '' % can use to load data from specific dir/ model
files = strsplit(ls(model_name + "/losses*.mat"),{'\n',' '});
set(groot,'defaultAxesTickLabelInterpreter','latex');
last = load(files{end-1},'-mat');
files{end-1}
win_size= [680.0000  697.3465  362.2047  271.6535];

figure(1)
plot(last.Gen_total_loss)
xlabel('Epoch','Interpreter', 'latex'); ylabel('${J_{G^*}}$','Interpreter', 'latex')
set(gcf,'Position', win_size)
grid on

figure(2)
plot(last.Gen_loss)
xlabel('Epoch','Interpreter', 'latex'); ylabel('${J_G}$','Interpreter', 'latex')
set(gcf,'Position', win_size)
% set(gca,'Xtick', 0:10:150)
grid on

figure(3)
plot(last.Gen_l1_loss)
xlabel('Epoch','Interpreter', 'latex'); ylabel('${J_{L1}}$','Interpreter', 'latex')
set(gcf,'Position', win_size)
grid on

figure(4)
plot(last.Disc_loss)
xlabel('Epoch','Interpreter', 'latex'); ylabel('${J_{D}}$','Interpreter', 'latex')
set(gcf,'Position', win_size)
% set(gca,'Xtick', 0:10:150)
grid on

figure(5)
plot(last.Reff_disc_loss)
xlabel('Epoch','Interpreter', 'latex'); ylabel('${J_{D_{reff}}}$','Interpreter', 'latex')
set(gcf,'Position', win_size)
grid on


lr_files = strsplit(ls(model_name + "/lr_rates*.mat"),{'\n', ' '});
lr_rates = load(lr_files{end-1});
lr_rates.title = strjoin(strsplit(lr_files{end-1},'_'));

figure(6)
plot(lr_rates.gen_lr)
hold on
plot(lr_rates.disc_lr)
legend('Gen LR','Disc LR'); grid on; %title(lr_rates.title);
set(gcf,'Position', win_size)
xlabel('Epoch','Interpreter', 'latex'); ylabel('Learning rate','Interpreter', 'latex')
hold off
