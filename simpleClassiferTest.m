clear variables;
filename = 'G:\My Drive\Image Rec Project\Code and results\testDataAndBayesian\testingData.mat';
load(filename);
[prediction, scores, index] = simpleClassifer(testVector);
predictionR = prediction';
correctClass = find(predictionR == testLabels); 
accuracy = length(correctClass) / length(testLabels);

%% Create the Confusion Matrix
categories = {'neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'none', 'uncertain', 'non-face', 'not'};

confMat = zeros(11, 11);
for expect = 1:11
    for ident = 1:11
        val = length(find((categories(expect) == predictionR) & (categories(ident) == testLabels)));
        confMat(expect, ident) = val;
    end
end

%% Very correct classifications
correctScores = scores(correctClass);
max1 = max(correctScores);
max1Index = find(correctScores == max1);
max1Label = testLabels(max1Index);
imshow(uint8(testVector(:,:,:,max1Index)));
%correctScores(max1Index) = 0;
