warning('off')

NDVI = csvread('NDVI.csv');

NDVI_PhS = phasespace(NDVI,50,1);
error = 'RMSE';
MinRMSE = Inf;
Best_m = 0;
Best_model = [];
Best_tr = [];
Best_time = [];

for m=1:50

        Features = [NDVI_PhS(1:end-1,end-m+1:end)];
      
        Targets = NDVI_PhS(2:end,end);

        N = length(Targets);
 
        Train = round(0.70 * N);
        Val = round(0.15 * N);

        %Training data set
        XTrain = Features(1:Train,:);
        YTrain = Targets(1:Train);

        %Validation data set
        XVal = Features(Train+1:Train+1+Val,:);
        YVal = Targets(Train+1:Train+1+Val);
        
        %Test data set
        XTest = Features(Train+2+Val:end,:);
        YTest = Targets(Train+2+Val:end);
        
        Means = mean(XTrain);
        Mean = mean(YTrain);

        Stds = std(XTrain);
        Std = std(YTrain);

        XTrain = bsxfun(@minus, XTrain, Means);
        XTrain = bsxfun(@rdivide, XTrain, Stds);

        XVal = bsxfun(@minus, XVal, Means);
        XVal = bsxfun(@rdivide, XVal, Stds);
        
        XTest = bsxfun(@minus, XTest, Means);
        XTest = bsxfun(@rdivide, XTest, Stds);

        YTrain = (YTrain - Mean) / Std;
        YVal = (YVal - Mean) / Std;
        YTest = (YTest - Mean) / Std;
        
   
        [b,se,pval,opt_inmodel,stats,nextstep,history] = stepwisefit(XTrain, YTrain,'display','off','penter',0.05,'premove',0.05);

        XTrain = XTrain(:,opt_inmodel);
        XVal = XVal(:,opt_inmodel);
        XTest = XTest(:,opt_inmodel);
        
        c = size(XTrain,2);
       
        name = 'PCA';
           
        [mappedA, mapping] = compute_mapping(XTrain, name, c);

        XTrain = out_of_sample(XTrain, mapping);
        XVal = out_of_sample(XVal, mapping);
        XTest = out_of_sample(XTest, mapping);
        
        [model, trbest] = trainLRNN(XTrain', YTrain', XVal', YVal', XTest', YTest');
         
        y_pred = testFFNN(model, XTest');

        resultsRLR     = assessment(YTest*Std+Mean, y_pred*Std+Mean,'regress',size(XTest,2));
        fprintf('Test: RMSE: %f, MAPE: %f, MAE: %f, DS: %f, R: %f \n', resultsRLR.RMSE, resultsRLR.MAPE, resultsRLR.MAE, resultsRLR.DS, resultsRLR.R);

        if resultsRLR.RMSE < MinRMSE
            Best_m = m;
            Best_model = model;
            Best_tr = trbest;
        end
         
end

        Features = [NDVI_PhS(1:end-1,end-Best_m+1:end)];
      
        Targets = NDVI_PhS(2:end,end);

        N = length(Targets);
 
        Train = round(0.70 * N);
        Val = round(0.15 * N);

        %Training data set
        XTrain = Features(1:Train,:);
        YTrain = Targets(1:Train);

        %Validation data set
        XVal = Features(Train+1:Train+1+Val,:);
        YVal = Targets(Train+1:Train+1+Val);
        
        %Test data set
        XTest = Features(Train+2+Val:end,:);
        YTest = Targets(Train+2+Val:end);
        
        Means = mean(XTrain);
        Mean = mean(YTrain);

        Stds = std(XTrain);
        Std = std(YTrain);

        XTrain = bsxfun(@minus, XTrain, Means);
        XTrain = bsxfun(@rdivide, XTrain, Stds);

        XVal = bsxfun(@minus, XVal, Means);
        XVal = bsxfun(@rdivide, XVal, Stds);
        
        XTest = bsxfun(@minus, XTest, Means);
        XTest = bsxfun(@rdivide, XTest, Stds);

        YTrain = (YTrain - Mean) / Std;
        YVal = (YVal - Mean) / Std;
        YTest = (YTest - Mean) / Std;
        
   
        [b,se,pval,opt_inmodel1,stats,nextstep,history] = stepwisefit(XTrain, YTrain,'display','off','penter',0.05,'premove',0.05);

        XTrain = XTrain(:,opt_inmodel1);
        XVal = XVal(:,opt_inmodel1);
        XTest = XTest(:,opt_inmodel1);
        
        c = size(XTrain,2);
       
        name = 'PCA';
           
        [mappedA, mapping] = compute_mapping(XTrain, name, c);

        XTrain = out_of_sample(XTrain, mapping);
        XVal = out_of_sample(XVal, mapping);
        XTest = out_of_sample(XTest, mapping);

        y_pred = testLRNN(Best_model, XTest');
        resultsRLR     = assessment(YTest*Std+Mean, y_pred'*Std+Mean,'regress',size(XTest,2));
        fprintf('Test: RMSE: %f, MAPE: %f, MAE: %f, DS: %f, R: %f \n', resultsRLR.RMSE, resultsRLR.MAPE, resultsRLR.MAE, resultsRLR.DS, resultsRLR.R);

        