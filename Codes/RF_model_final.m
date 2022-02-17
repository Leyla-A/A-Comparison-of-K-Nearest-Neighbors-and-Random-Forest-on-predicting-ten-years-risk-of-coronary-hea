%% Read balanced Data
X_train = readtable("X_train_smote.csv");
y_train = readtable("y_train_smote.csv");
X_train = normalize(X_train,"range"); %scale values to [0,1] interval

X_test = readtable("X_test_smote.csv");
y_test = readtable("y_test_smote.csv");
X_test = normalize(X_test,"range"); %scale values to [0,1] interval
save('X_test_normalized.mat',"X_test")

%% Bayesian Optimization of parameters
rng('default'); %for reproducibility

minLS = optimizableVariable('minLS',[1,30],'Type','integer');
numPTS = optimizableVariable('numPTS',[1,size(X_train,2)],'Type','integer');
hyperparametersRF = [ minLS; numPTS];
results = bayesopt(@(params)oobErrRF(params,X_train,y_train),hyperparametersRF,...
    'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);

bestOOBErr = results.MinObjective
bestHyperparameters = results.XAtMinObjective

%% Fitting model with optimized parameters
rng('default')  %for reproducibility
RF_modelSMOTE_final = TreeBagger(200, X_train,y_train,...
    'Method','classification',...
    'OOBPrediction', 'on',...
    'MinLeafSize',bestHyperparameters.minLS,...
    'NumPredictorstoSample',bestHyperparameters.numPTS, ...
    'OOBPredictorImportance', 'on');

%% Predictor Importance Estimates
imp = RF_modelSMOTE_final.OOBPermutedPredictorDeltaError;
figure;
bar(imp);
title('Curvature Test');
ylabel('Predictor importance estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = RF_modelSMOTE_final.PredictorNames;
h.XTickLabelRotation = 90;
h.TickLabelInterpreter = 'none';

%% Analising performance of fitted model
% plotting out of bag Error of trees
figure;
oobErrorBaggedEnsemble = oobError(RF_modelSMOTE_final);
plot(oobErrorBaggedEnsemble);
xlabel('Number of grown trees');
ylabel('Out-of-bag classification error');

% average error
meanError = mean(oobErrorBaggedEnsemble);
fprintf('Average error of Random Forest: %f\n',meanError)

% AUC and ploting ROC curve
[yfit,scores] = oobPredict(RF_modelSMOTE_final);
[fpr,tpr,~,AUC] = perfcurve(RF_modelSMOTE_final.Y,scores(:,2),'1');
fprintf('AUC is %f\n',AUC) % Area under the ROC Curve
figure;
plot(fpr,tpr); %ROC Curve
xlabel('False Positive Rate');
ylabel('True Positive Rate');

% Confusion chart and accuracy, precision, recall, F1 score
figure;
[cm, order] = confusionmat(RF_modelSMOTE_final.Y,yfit);
c = confusionchart(cm,order);
c.RowSummary = 'row-normalized';
c.ColumnSummary = 'column-normalized';

TP = cm(2,2); TN = cm(1,1); FP =cm(1,2); FN = cm(2,1);
accuracy = (TP + TN)/(sum(sum(cm)));
fprintf('Accuracy is %f\n',accuracy);
prec = TP/(TP + FP);
fprintf('Precision is %f\n',prec);
rec = TP/(TP+FN);
fprintf('Recall is %f\n',rec);
F1 = 2*TP/(2*TP+FP+FN);
fprintf('F score is %f\n',F1);

%% saving model
rng('default') %for reproducibility
RF_smote_final = compact(RF_modelSMOTE_final); % for predict() function we need CompactTreeBagger model

save('RF_smote_final.mat','RF_smote_final')
%% Analising performance on testing data
fprintf('time passed for predicting test set')
tic
[yfitT, scoresT] = predict(RF_smote_final,X_test);
toc

figure;
[cmT, orderT] = confusionmat(table2array(y_test),round(scoresT(:,2)));
cT = confusionchart(cmT,orderT);
cT.RowSummary = 'row-normalized';
cT.ColumnSummary = 'column-normalized';

% Confusion chart and accuracy, precision, recall, F1 score
TPt = cmT(2,2); TNt = cmT(1,1); FPt =cmT(1,2); FNt = cmT(2,1);
accuracyT = (TPt + TNt)/(sum(sum(cmT)));
fprintf('Accuracy is %f\n',accuracyT);
precT = TPt/(TPt + FPt);
fprintf('Precision is %f\n',precT);
recT = TPt/(TPt+FNt);
fprintf('Recall is %f\n',recT);
F1T = 2*TPt/(2*TPt+FPt+FNt);
fprintf('F score is %f\n',F1T);

% comparing train and test set ROC curves
[fprT,tprT,~,AUCT] = perfcurve(y_test.TenYearCHD,scoresT(:,2),1);
fprintf('AUC is %f\n',AUCT) % Area under the ROC Curve for test set

figure;
plot(fpr,tpr)
hold on
plot (fprT,tprT)
legend('Train', 'Test')
xlabel('False Positive Rate');
ylabel('True Positive Rate');
hold off

%%  Function to calculate Objectives for each parameters chosen for Bayesian Optimization
function oobErr = oobErrRF(params,X_train,y_train)
randomForest = TreeBagger(200, X_train, y_train, 'Method', 'classification',...
    'OOBPrediction', 'on', 'MinLeafSize', params.minLS,...
    'NumPredictorstoSample', params.numPTS);
oobErr = oobError(randomForest, 'Mode','ensemble');
end