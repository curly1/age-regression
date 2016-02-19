clear, close all;

addpath([cd '/code'])
addpath([cd '/code/LSSVM/LSSVMlabv1_8_R2009b_R2011a'])
data1 = load('aGender_ProsodicParams.mat');
data2 = load('aGender_ivec.mat');

%% SETTINGS
minAge = 0;
maxAge = 80;
%group = (data2.females);
group = (data2.males | data2.females | data2.children);
kernel = 'RBF_kernel';

%% make sure ids of data1 = ids of data2

ids1 = cell2mat(data1.params(:,2));
ids2 = data2.ids(group);

% sort by id
[ids1,i1] = sort(ids1);
data1.features = data1.features(i1,:);
data1.labels = data1.labels(i1);

[ids2,i2] = sort(ids2);
data2.features = data2.features(i2,:);
data2.labels = data2.labels(group);
data2.labels = data2.labels(i2);

% same data in both files
[idx1, idx2] = compareMatrices(ids2, ids1);
X1 = data1.features(idx1,:);
Y1 = data1.labels(idx1);
X2 = data2.features(idx2,:);
Y2 = data2.labels(idx2);

%%
K = 10; 

folds = crossvalind('Kfold', size(X1,1), K);                 % randomowo czesc danych do testu

for k = 1:K

test_idx = (folds == k);
train_idx = (folds ~= k);

type = 'function estimation';

%% 1st model
%[gam1,sig21] = tunelssvm({X1(train_idx,:),Y1(train_idx),type,[],[], kernel},...
%    'simplex', 'leaveoneoutlssvm',{'mse'});
gam1 = 23.7135;     % RBF
sig21 = 36.4475;

[alpha1,b1] = trainlssvm({X1(train_idx,:),Y1(train_idx),type,gam1,sig21, kernel});

Y_pred1 = simlssvm({X1(train_idx,:),Y1(train_idx),type,gam1,sig21, kernel ,'preprocess'},...
    {alpha1,b1},X1(test_idx,:));
Y_pred1(Y_pred1 < minAge) = minAge;
Y_pred1(Y_pred1 > maxAge) = maxAge;

Y_true1 = Y1(test_idx);
[Y_true1, idx] = sort(Y_true1);

Y_pred1 = Y_pred1(idx);

MAE1 = 1/(length(Y_pred1))*sum(abs(Y_pred1-Y_true1));
MAE_std1 = std(abs(Y_pred1-Y_true1));

%% 2nd model
%[gam2,sig22] = tunelssvm({X2(train_idx,:),Y2(train_idx),type,[],[], kernel},...
%    'simplex', 'leaveoneoutlssvm',{'mse'});
gam2 = 7.760218;     % RBF
sig22 = 820.3196;

[alpha2,b2] = trainlssvm({X2(train_idx,:),Y2(train_idx),type,gam2,sig22, kernel});

Y_pred2 = simlssvm({X2(train_idx,:),Y2(train_idx),type,gam2,sig22, kernel ,'preprocess'},...
    {alpha2,b2},X2(test_idx,:));
Y_pred2(Y_pred2 < minAge) = minAge;
Y_pred2(Y_pred2 > maxAge) = maxAge;

Y_true2 = Y2(test_idx);
[Y_true2, idx] = sort(Y_true2);

Y_pred2 = Y_pred2(idx);

MAE2 = 1/(length(Y_pred2))*sum(abs(Y_pred2-Y_true2));
MAE_std2 = std(abs(Y_pred2-Y_true2));

%% fusion
Y_pred = (0.2*Y_pred1) + (0.8*Y_pred2);
if Y_true1 == Y_true2, Y_true = Y_true1; end
MAE_fused = 1/(length(Y_pred))*sum(abs(Y_pred-Y_true));
MAE_std_fused = std(abs(Y_pred-Y_true));

%% correlation coefficients
p1 = corrcoef([Y_true, Y_pred1]);
p2 = corrcoef([Y_true, Y_pred2]);
p = corrcoef([Y_true, Y_pred]);

[MAE_fused < MAE2 , p(2) > p2(2)]

p_ref = 1/(length(Y_pred)-1)*((Y_true-mean(Y_true))/std(Y_true))'*((Y_pred-mean(Y_pred))/std(Y_pred));

pearson(k) = p(2) - p2(2);
MAE_diff(k) = MAE2 - MAE_fused;

end

P = mean(pearson)
MAE = mean(MAE_diff)

%%
figure()
plot(Y_true1); hold on
plot(Y_pred1, 'or')
legend('True', 'Predicted')
ylabel('age'), xlabel('# speaker')
title('model 1')

figure()
plot(Y_true2); hold on
plot(Y_pred2, 'or')
legend('True', 'Predicted')
ylabel('age'), xlabel('# speaker')
title('model 2')

figure()
plot(Y_true); hold on
plot(Y_pred, 'or')
legend('True', 'Predicted')
ylabel('age'), xlabel('# speaker')
title('fused model')

rmpath([cd '/code'])
rmpath([cd '/code/LSSVM/LSSVMlabv1_8_R2009b_R2011a'])