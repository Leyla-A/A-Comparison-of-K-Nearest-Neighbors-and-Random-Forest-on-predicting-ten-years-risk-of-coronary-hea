clear global; close all; clc
% Load models for KNN and Random forest models
load('knn_model_final.mat');
load('RF_smote_final.mat');

% Read Test data sets
load('X_test_normalized.mat'); % because Train set also normalized
load("X_test_feature_selected.mat"); % because we have done feature selection for KNN
y_test = readtable("y_test_smote.csv");


% Predicting labels and scors for test set, print run time
fprintf('KNN ')
tic
[yfit_knn, score_knn] = predict(knn_model_final,X_test_new );
toc
fprintf('Random Forest ')
tic
[yfit_rf_str, score_rf]= predict(RF_smote_final,X_test); %predict produces string labels so using score and rounding will produce numeric labels
toc
yfit_rf = round(score_rf(:,2)); % rounding second will give us predictid numeric values



% plotting ROC Curve for KNN and RF
[fprKNN,tprKNN,~,AUCknn] = perfcurve(y_test.TenYearCHD,score_knn(:,2),1);

[fprRF, tprRF,~,AUCrf] = perfcurve(y_test.TenYearCHD,score_rf(:,2),1);


figure;
plot(fprKNN,tprKNN); %ROC Curve
hold on
plot(fprRF,tprRF); %ROC Curve
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('KNN','Random Forest');

% Confusion chart for KNN 
% and accuracy, precision, recall, F1 score
figure;
[cmKNN, orderKNN] = confusionmat(y_test.TenYearCHD,yfit_knn);
c = confusionchart(cmKNN,orderKNN);
c.RowSummary = 'row-normalized';
c.ColumnSummary = 'column-normalized';
c.Title = 'Confusion chart for K Nearest Neighbor classification';

fprintf('For K Nearest Neighbor\n')
fprintf('AUC is %f\n',AUCknn) % Area under the ROC Curve
TPknn = cmKNN(2,2); TNknn = cmKNN(1,1); FPknn =cmKNN(1,2); FNknn = cmKNN(2,1);
accuracyKNN = (TPknn + TNknn)/(sum(sum(cmKNN)));
fprintf('Accuracy is %f\n',accuracyKNN);
precKNN = TPknn/(TPknn + FPknn);
fprintf('Precision is %f\n',precKNN);
recKNN = TPknn/(TPknn+FNknn);
fprintf('Recall is %f\n',recKNN);
F1knn = 2*TPknn/(2*TPknn+FPknn+FNknn);
fprintf('F score is %f\n',F1knn);

% Confusion chart for Random forest model
% and accuracy, precision, recall, F1 score
figure;
[cmRF, orderRF] = confusionmat(y_test.TenYearCHD,yfit_rf);
c = confusionchart(cmRF,orderRF);
c.RowSummary = 'row-normalized';
c.ColumnSummary = 'column-normalized';
c.Title = 'Confusion chart for Random Forest classification';

fprintf('\n')
fprintf('For Random Forest\n')
fprintf('AUC is %f\n',AUCrf) % Area under the ROC Curve
TPrf = cmRF(2,2); TNrf = cmRF(1,1); FPrf =cmRF(1,2); FNrf = cmRF(2,1);
accuracyRF = (TPrf + TNrf)/(sum(sum(cmRF)));
fprintf('Accuracy is %f\n',accuracyRF);
precRF = TPrf/(TPrf + FPrf);
fprintf('Precision is %f\n',precRF);
recRF = TPrf/(TPrf+FNrf);
fprintf('Recall is %f\n',recRF);
F1rf = 2*TPrf/(2*TPrf+FPrf+FNrf);
fprintf('F score is %f\n',F1rf);