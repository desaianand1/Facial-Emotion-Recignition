%% Load Test Data
clear variables;
testData= 'G:\My Drive\Image Rec Project\Code and results\testDataAndBayesian\testingData.mat';
load(testData);

%% Loop through single emotion Nets and classify data.
directory = 'G:\My Drive\Image Rec Project\Code and results\oneEmotion\';
categories = {'neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'none', 'uncertain', 'non-face'};
numEmotions = size(categories,2);


figure();
hold on
Xpoints = zeros(4999, numEmotions);
Ypoints = zeros(4999, numEmotions);
for i=1:numEmotions
    trueClass= testLabels;
    clear  netTransfer
    load(strcat(directory,(string(categories(i))+ '.mat')))
    trueClass(trueClass~=string(categories(i))) = 'not';
    [Ypred, scores] = classify(netTransfer, testVector);

    if(Ypred(1) == "not" && scores(1,1) >=.5) || (Ypred(1) ~= "not" && scores(1,1) <=.5)
        predScores2 = scores(:,2);
        
    else
        predScores2 = scores(:,1);
    end
    [X,Y,T]= perfcurve(trueClass',predScores2,string(categories(i)));
         plot(X,Y)
    dataSize = length(X);
    Xpoints(1:dataSize, i) = X;
    Ypoints(1:dataSize,i) = Y;
            
end
%%
hold off
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC curve');
legend(categories);
savefig('roc_combined.png');