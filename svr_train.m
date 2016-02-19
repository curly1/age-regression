%clear, close all;

addpath([cd '/code'])
addpath([cd '/code/LSSVM/LSSVMlabv1_8_R2009b_R2011a'])
data = load('aGender_PitchParams.mat');
load([cd '/data/fixed_folds_aGender_K15.mat']);
%load('hyperparams.mat');

% SETTINGS
minAge = 6;
maxAge = 80;
%group = (data.females);
%group = (data.males | data.females | data.children);
group = logical(data.labels);
folds = folds2;
who = who2;
%gam = ones(15,1).*gam;
%sig2 = ones(15,1).*sig2;

folds = folds(group);
%group = logical(data.labels);          % all data
kernel = 'RBF_kernel';
whiten = 0;
w = 0;                                  % wccn

X = data.features(group,:);
X = X(:,[1 2 5 6 9 10]);
Y = data.labels(group);

K = 15;
type = 'function estimation';
%[gam,sig2] = tunelssvm({X,Y,type,[],[], kernel},...
%   'simplex', 'leaveoneoutlssvm',{'mse'});
%folds = crossvalind('Kfold', size(X,1), K);   
%%
for k = 1:K
    k
    test_idx = (folds == k);
    train_idx = (folds ~= k);
    
    %[gam(k), sig2(k)] = tunelssvm({X(train_idx,:),Y(train_idx),type,[],[], kernel}, 'simplex', 'crossvalidatelssvm', {10,'mse'});      
    
    if whiten == 1
        m = mean(X(train_idx,:));
        X = bsxfun(@minus,X,m);
        W = calc_white_mat(cov(X(train_idx,:)));
        X = X*W;
    end
    
    features_wccn = zeros(size(data.features));
    if w == 1
     % Lf=wccn(data.features(data.females&train_idx',:), who(data.females&train_idx')');
     % Lm=wccn(data.features(data.males&train_idx',:), who(data.males&train_idx')');
     % Lc=wccn(data.features(data.children&train_idx',:), who(data.children&train_idx')');
      L=wccn(data.features(train_idx',:), who(train_idx')');
      FeaWCCN = data.features*L;
      %features_wccn(data.females,:) = data.features(data.females,:)*Lf;
      %features_wccn(data.males,:) = data.features(data.males,:)*Lm;
      %features_wccn(data.children,:) = data.features(data.children,:)*Lc;
      X = FeaWCCN(group,:);
      %m = mean(X(train_idx,:));
      %X = bsxfun(@minus,X,m);
    end
    
    %model = initlssvm(X(train_idx,:),Y(train_idx),'f',[],[],'RBF_kernel');
    %L_fold = 3;                                                 % 3 fold CV
    %model = tunelssvm(model,'simplex', 'rcrossvalidatelssvm',{L_fold,'mae'},'wmyriad');
    %model = robustlssvm(model);
    %Y_pred1 = simlssvm(model,X(test_idx,:));
    
    %gam = 20.53258;     % RBF
    %sig2 = 322.6538;
    %gam = 10580.058;     % RBF
    %sig2 = 322.668;
    %gam = 0.002755;     % linear
    %sig2 = 128.8632;
    %gam = 25301.0366;   % poly
    %sig2 = [208.7688 3];
    
    [alpha,b] = trainlssvm({X(train_idx,:),Y(train_idx),type,gam(k),sig2(k), kernel});
    
    Y_pred = simlssvm({X(train_idx,:),Y(train_idx),type,gam(k),sig2(k), kernel ,'preprocess'},...
        {alpha,b},X(test_idx,:));
    Y_pred(Y_pred < minAge) = minAge;
    Y_pred(Y_pred > maxAge) = maxAge;
    
    Y_true = Y(test_idx);
    
    %MAE1 = 1/(length(Y_pred1))*sum(abs(Y_pred1-Y_true));
    %MAE_std1 = std(abs(Y_pred1-Y_true));
    
    [Y_true, idx] = sort(Y_true);
    Y_pred = Y_pred(idx);
    
    Y_prior = ones(size(Y_true)).*mean(Y(train_idx));
    MAE_prior(k) = 1/(length(Y_prior))*sum(abs(Y_prior-Y_true));
    MAE(k) = 1/(length(Y_pred))*sum(abs(Y_pred-Y_true));
    MAE_std = std(abs(Y_pred-Y_true));
    
    p(k) = 1/(length(Y_pred)-1)*((Y_true-mean(Y_true))/std(Y_true))'*((Y_pred-mean(Y_pred))/std(Y_pred));
end

mean(MAE_prior)
mean(MAE)
mean(p)

figure()
plot(Y_true, '-o'); hold on
plot(Y_pred, 'or')
legend('True', 'Predicted')
ylabel('age'), xlabel('# speaker')

rmpath([cd '/code'])
rmpath([cd '/code/LSSVM/LSSVMlabv1_8_R2009b_R2011a'])