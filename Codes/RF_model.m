clear; clc; close all;
%% Load data:
X_train = readtable("X_train.csv");
y_train = readtable("y_train.csv");
X_train = normalize(X_train,"range"); %scale values to [0,1] interval

X_test = readtable("X_test.csv");
y_test = readtable("y_test.csv");
X_test = normalize(X_test,"range"); %scale values to [0,1] interval
%% Define Random forest model
rng('default')
mdl1 = TreeBagger(200, X_train,y_train,'PredictorSelection','curvature',...
    'OOBPrediction','On','OOBPredictorImportance', 'on', 'Method','classification');

%% plot errors
figure;
oobErrorBaggedEnsemble = oobError(mdl1);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

% Accuracy in percent
[y_pred1, scores1]  = predict(mdl1, X_test); % predict test labels
accuracy1 = ((sum(round(scores1(:,2)) == table2array(y_test))))/size(y_test,1)*100;
fprintf('Accuracy of Random Forest: %f \n',accuracy1);
% Confusion matrix
[conmat1,order1] = confusionmat((table2array(y_test)),round(scores1(:,2))');
figure("Name",'Confusion Chart')
cm1 = confusionchart(conmat1,order1);
cm1.RowSummary = 'row-normalized';
cm1.ColumnSummary = 'column-normalized';


%% Unbiased Predictor Importance Estimates
rng('default')
imp = mdl1.OOBPermutedPredictorDeltaError;
figure;
bar(imp);
title('Curvature Test');
ylabel('Predictor importance estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = mdl1.PredictorNames;
h.XTickLabelRotation = 90;
h.TickLabelInterpreter = 'none';

%% Feature Selection using Predictor Weights
idxvar = find(imp>mean(imp))

% model with reduced predictors
rng('default')
mdl2 = TreeBagger(200, X_train(:,idxvar),y_train,'OOBPrediction','On','OOBPredictorImportance', 'off',...
    'Method','classification');
% Errors
figure;
oobErrorBaggedEnsemble = oobError(mdl2);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

% Accuracy
[y_pred, scores]  = predict(mdl2, X_test(:,idxvar));
accuracy = ((sum(round(scores(:,2)) == table2array(y_test))))/size(y_test,1)*100;
fprintf('Accuracy of Random Forest: %f \n',accuracy);
% Confusion matrix
[conmat,order] = confusionmat((table2array(y_test)),round(scores(:,2))');
figure("Name",'Confusion Chart')
cm = confusionchart(conmat,order);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';