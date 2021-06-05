clear

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

model_name  = 'SM-3D_conv';
cd '/home/lab/orel_ws/project/model_training/'
files = strsplit(ls(model_name + "/losses*.mat"),{'\n',' '});
set(groot,'defaultAxesTickLabelInterpreter','latex');
last = load(files{end-1},'-mat');
files{end-1}
win_size= [680.0000  697.3465  362.2047  271.6535]

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

% figure(5)
% plot(smooth(last.Reff_disc_loss,50))
% xlabel('Epoch','Interpreter', 'latex'); ylabel('${J_{D_{reff}}}$','Interpreter', 'latex')
% set(gcf,'Position', size)
% grid on

figure(5)
plot(last.Reff_disc_loss)
xlabel('Epoch','Interpreter', 'latex'); ylabel('${J_{D_{reff}}}$','Interpreter', 'latex')
set(gcf,'Position', win_size)
grid on


lr_files = strsplit(ls(model_name + "/lr_rates*.mat"),{'\n', ' '});
lr_rates = load(lr_files{end-1});
lr_rates.title = strjoin(strsplit(lr_files{end-1},'_'));
% lr_rates_2 = load(lr_files{end-2});
% lr_rates_2.title = strjoin(strsplit(lr_files{end-2},'_'));

figure(6)
plot(lr_rates.gen_lr)
hold on
plot(lr_rates.disc_lr)
legend('Gen LR','Disc LR'); grid on; %title(lr_rates.title);
set(gcf,'Position', win_size)
xlabel('Epoch','Interpreter', 'latex'); ylabel('Learning rate','Interpreter', 'latex')
hold off

%% Plot all reff loss in one graph 

cd '/home/lab/orel_ws/project/model_training/'
win_size= [680 219 1108 750];
directory = dir();
models = directory(startsWith({directory(:).name},{'SM', 'ARM'}));
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
figure(7)
set(gcf,'Position', win_size); hold on;
for i=1:size(models,1)
    loss_mat = strsplit(ls(models(i).name + "/losses*.mat"),{'\n',' '});
    models(i).loss = load(loss_mat{end-1},'-mat');
    plot(models(i).loss.Reff_disc_loss);
    leg{i} = replace(models(i).name,'_',' ');
end    
grid on; legend(leg,'Location','northwest','NumColumns',2); 
xlabel('Epoch','Interpreter', 'latex'); ylabel('${J_{D_{reff}}}$','Interpreter', 'latex')
set(gca,'FontSize', 20); box on;
hold off;


% figure(8)
% plot(lr_rates_2.gen_lr)
% hold on
% plot(lr_rates_2.disc_lr)
% legend('Gen LR','Disc LR'); grid on; title(lr_rates_2.title);
% set(gcf,'Position', size)
% xlabel('Epoch','Interpreter', 'latex'); ylabel('Learning rate','Interpreter', 'latex')
% hold off